"""Reproducible, single-machine hyperparameter experiments for mini_param_agent.

The tool deliberately uses a small contract instead of trying to rewrite an
arbitrary training program.  A training script receives its trial values in
``ML_EXPERIMENT_PARAMS`` (a JSON object) and as ``ML_PARAM_<NAME>`` variables.
It reports a metrics JSON/CSV file at ``ML_EXPERIMENT_METRICS_PATH``.  A final
stdout line such as ``ML_METRICS: {\"val_accuracy\": 0.91}`` is also supported.
This works equally well for PyTorch, scikit-learn, and any other executable
Python training script.
"""

import asyncio
import csv
import json
import os
import pprint
import re
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult


class MLExperimentTool(Tool):
    """Run a baseline and Optuna trials against a user-provided metric."""

    _PARAM_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _MARKER_START = "# ML_EXPERIMENT_PARAMS_START"
    _MARKER_END = "# ML_EXPERIMENT_PARAMS_END"

    def __init__(self, workspace_dir: str = "."):
        # resolve() canonicalizes /var and /private/var on macOS, avoiding false
        # workspace-boundary failures for temporary directories and symlinked projects.
        self.workspace_dir = Path(workspace_dir).resolve()

    @property
    def name(self) -> str:
        return "run_ml_experiment"

    @property
    def description(self) -> str:
        return (
            "Run a reproducible baseline plus Optuna hyperparameter search for one Python "
            "training script or Jupyter notebook. The training program receives ML_EXPERIMENT_PARAMS "
            "(JSON), ML_PARAM_<NAME>, ML_EXPERIMENT_METRICS_PATH, and ML_EXPERIMENT_TRIAL_DIR environment "
            "variables. It must write a JSON/CSV metrics file to ML_EXPERIMENT_METRICS_PATH or print a final "
            "line 'ML_METRICS: {\"metric\": value}'. Use parameter_space entries such as "
            "{'lr': {'type':'float','low':1e-4,'high':1e-2,'log':true}} and a validation metric. "
            "All logs, trial parameters, CSV/JSON results, and the best-parameter report are saved under "
            ".mini_param_agent/experiments. Source files are never edited unless write_back_path is explicitly set."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "script_path": {"type": "string", "description": "Workspace-relative or absolute .py or .ipynb training file."},
                "metric_name": {"type": "string", "description": "Validation metric key to optimize, for example val_accuracy or val_loss."},
                "metric_mode": {"type": "string", "enum": ["maximize", "minimize"], "default": "maximize"},
                "parameter_space": {"type": "object", "description": "Optuna space: each value has type float/int/categorical and low/high or choices."},
                "baseline_params": {"type": "object", "description": "Optional reference values for the baseline run. Missing values use each parameter spec's baseline field, low bound, or first categorical choice."},
                "n_trials": {"type": "integer", "description": "Number of Optuna trials after the baseline (default 10). Set 0 to run only the baseline.", "default": 10},
                "train_command": {"type": "string", "description": "Optional command template. Tokens may use {python}, {script}, {metrics_path}, {trial_dir}, and parameter names, e.g. '{python} {script} --lr {lr}'. Default: '{python} {script}'."},
                "metrics_format": {"type": "string", "enum": ["auto", "json", "csv"], "default": "auto"},
                "timeout": {"type": "integer", "description": "Per-trial timeout in seconds, 1-3600 (default 600).", "default": 600},
                "seed": {"type": "integer", "description": "Seed supplied to Optuna and ML_EXPERIMENT_SEED (default 42).", "default": 42},
                "write_back_path": {"type": "string", "description": "Optional .json config to merge best parameters into, or .py file containing ML_EXPERIMENT_PARAMS marker comments. Never defaults to the source file."},
                "file": {"type": "string", "description": "Compatibility alias for script_path."},
                "metric": {"type": "string", "description": "Compatibility alias for metric_name."},
                "direction": {"type": "string", "enum": ["maximize", "minimize"], "description": "Compatibility alias for metric_mode."},
                "params": {"description": "Compatibility alias for parameter_space; accepts an object or a JSON string. min/max are accepted as low/high."},
            },
            # Both canonical names and aliases are accepted for smaller local
            # models, which often infer generic names such as file/metric.
            "required": [],
        }

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace_dir / candidate
        return candidate.resolve()

    def _require_workspace_path(self, path: Path) -> None:
        try:
            path.relative_to(self.workspace_dir)
        except ValueError as exc:
            raise ValueError(f"Path must be inside the workspace: {path}") from exc

    def _validate_space(self, parameter_space: dict[str, Any]) -> None:
        for name, spec in parameter_space.items():
            if not self._PARAM_NAME.match(name) or not isinstance(spec, dict):
                raise ValueError(f"Invalid parameter specification for '{name}'")
            kind = spec.get("type")
            if kind not in {"float", "int", "categorical"}:
                raise ValueError(f"Parameter '{name}' type must be float, int, or categorical")
            if kind == "categorical":
                if not isinstance(spec.get("choices"), list) or not spec["choices"]:
                    raise ValueError(f"Categorical parameter '{name}' needs a non-empty choices list")
            elif "low" not in spec or "high" not in spec:
                raise ValueError(f"Parameter '{name}' needs low and high values")

    @staticmethod
    def _normalize_parameter_space(parameter_space: dict[str, Any] | str | None) -> dict[str, Any]:
        """Accept canonical Optuna specs and common local-model shorthand."""
        if parameter_space is None:
            return {}
        if isinstance(parameter_space, str):
            try:
                parameter_space = json.loads(parameter_space)
            except json.JSONDecodeError as exc:
                raise ValueError("parameter_space/params must be a JSON object or valid JSON string") from exc
        if not isinstance(parameter_space, dict):
            raise ValueError("parameter_space/params must be an object")

        normalized: dict[str, Any] = {}
        for name, raw_spec in parameter_space.items():
            if not isinstance(raw_spec, dict):
                raise ValueError(f"Parameter '{name}' must be an object")
            spec = dict(raw_spec)
            if "min" in spec and "low" not in spec:
                spec["low"] = spec.pop("min")
            if "max" in spec and "high" not in spec:
                spec["high"] = spec.pop("max")
            if "type" not in spec:
                if "choices" in spec:
                    spec["type"] = "categorical"
                elif "step" in spec or (isinstance(spec.get("low"), int) and isinstance(spec.get("high"), int)):
                    spec["type"] = "int"
                else:
                    spec["type"] = "float"
            normalized[name] = spec
        return normalized

    def _prepare_script(self, source: Path, experiment_dir: Path) -> Path:
        if source.suffix == ".py":
            return source
        if source.suffix != ".ipynb":
            raise ValueError("script_path must end in .py or .ipynb")
        converter = shutil.which("jupyter-nbconvert")
        jupyter = shutil.which("jupyter")
        if not converter and not jupyter:
            raise RuntimeError("Notebook conversion needs nbconvert. Install the 'ml' extra or jupyter/nbconvert.")
        converted_dir = experiment_dir / "notebook_source"
        converted_dir.mkdir(parents=True, exist_ok=True)
        command = ([converter] if converter else [jupyter, "nbconvert"]) + ["--to", "script", "--output-dir", str(converted_dir), str(source)]
        result = __import__("subprocess").run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        converted = converted_dir / f"{source.stem}.py"
        if result.returncode != 0 or not converted.exists():
            message = result.stderr.strip() or result.stdout.strip() or "unknown conversion error"
            raise RuntimeError(f"Notebook conversion failed: {message}")
        return converted

    @staticmethod
    def _format_command(template: str, values: dict[str, Any]) -> list[str]:
        try:
            command = template.format(**values)
        except KeyError as exc:
            raise ValueError(f"train_command refers to an unknown placeholder: {exc.args[0]}") from exc
        args = shlex.split(command)
        if not args:
            raise ValueError("train_command cannot be empty")
        return args

    @staticmethod
    def _read_metrics(path: Path, metrics_format: str, stdout: str) -> dict[str, Any] | None:
        if path.exists() and path.stat().st_size:
            content = path.read_text(encoding="utf-8").strip()
            formats = [metrics_format] if metrics_format != "auto" else ["json", "csv"]
            for fmt in formats:
                try:
                    if fmt == "json":
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            return parsed
                    elif fmt == "csv":
                        rows = list(csv.DictReader(content.splitlines()))
                        if rows:
                            return dict(rows[-1])
                except (json.JSONDecodeError, csv.Error):
                    continue

        for line in reversed(stdout.splitlines()):
            payload = line.split("ML_METRICS:", 1)[1].strip() if "ML_METRICS:" in line else line.strip()
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    async def _run_trial(
        self,
        script: Path,
        experiment_dir: Path,
        trial_id: int,
        params: dict[str, Any],
        train_command: str,
        metrics_format: str,
        timeout: int,
        seed: int,
    ) -> dict[str, Any]:
        trial_dir = experiment_dir / f"trial_{trial_id:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = trial_dir / f"metrics.{ 'csv' if metrics_format == 'csv' else 'json' }"
        # Quote every template value before shlex splits the command. This permits workspace
        # paths and categorical values containing spaces without invoking a shell.
        raw_values = {
            "python": sys.executable,
            "script": str(script),
            "metrics_path": str(metrics_path),
            "trial_dir": str(trial_dir),
            **params,
        }
        values = {key: shlex.quote(str(value)) for key, value in raw_values.items()}
        command = self._format_command(train_command, values)
        environment = os.environ.copy()
        environment.update(
            {
                "ML_EXPERIMENT_PARAMS": json.dumps(params),
                "ML_EXPERIMENT_METRICS_PATH": str(metrics_path),
                "ML_EXPERIMENT_TRIAL_DIR": str(trial_dir),
                "ML_EXPERIMENT_SEED": str(seed),
            }
        )
        for key, value in params.items():
            environment[f"ML_PARAM_{key.upper()}"] = str(value)

        started_at = datetime.now(timezone.utc).isoformat()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.workspace_dir),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
                return {"trial": trial_id, "params": params, "status": "timeout", "error": f"Timed out after {timeout}s", "started_at": started_at}
        except OSError as exc:
            return {"trial": trial_id, "params": params, "status": "failed", "error": str(exc), "started_at": started_at}

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        (trial_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (trial_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        record: dict[str, Any] = {"trial": trial_id, "params": params, "started_at": started_at, "command": command}
        if process.returncode != 0:
            record.update({"status": "failed", "return_code": process.returncode, "error": stderr[-2000:] or "Training command failed"})
            return record
        metrics = self._read_metrics(metrics_path, metrics_format, stdout)
        if metrics is None:
            record.update({"status": "failed", "return_code": 0, "error": "No metrics found. Write ML_EXPERIMENT_METRICS_PATH or print ML_METRICS: {...}."})
            return record
        record.update({"status": "completed", "return_code": 0, "metrics": metrics})
        return record

    @staticmethod
    def _metric_value(record: dict[str, Any], metric_name: str) -> float:
        metrics = record.get("metrics", {})
        try:
            return float(metrics[metric_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Trial {record.get('trial')} did not report numeric metric '{metric_name}'") from exc

    def _write_back(self, path: Path, params: dict[str, Any]) -> None:
        self._require_workspace_path(path)
        if path.suffix == ".json":
            existing: dict[str, Any] = {}
            if path.exists() and path.read_text(encoding="utf-8").strip():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("write_back_path JSON must contain an object")
                existing = loaded
            existing.update(params)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return
        if path.suffix == ".py" and path.exists():
            source = path.read_text(encoding="utf-8")
            start, end = source.find(self._MARKER_START), source.find(self._MARKER_END)
            if start < 0 or end < 0 or end <= start:
                raise ValueError("Python write-back requires # ML_EXPERIMENT_PARAMS_START and # ML_EXPERIMENT_PARAMS_END markers")
            # A Python source file needs Python literals (True/False/None),
            # not JSON literals (true/false/null).
            replacement = f"{self._MARKER_START}\nML_EXPERIMENT_PARAMS = {pprint.pformat(params, sort_dicts=True)}\n"
            after_marker = end + len(self._MARKER_END)
            path.write_text(source[:start] + replacement + self._MARKER_END + source[after_marker:], encoding="utf-8")
            return
        raise ValueError("write_back_path must be a .json file or an existing marked .py file")

    async def execute(
        self,
        script_path: str | None = None,
        metric_name: str | None = None,
        metric_mode: str = "maximize",
        parameter_space: dict[str, Any] | str | None = None,
        baseline_params: dict[str, Any] | None = None,
        n_trials: int = 10,
        train_command: str | None = None,
        metrics_format: str = "auto",
        timeout: int = 600,
        seed: int = 42,
        write_back_path: str | None = None,
        **legacy_kwargs: Any,
    ) -> ToolResult:
        try:
            # Local models frequently generate generic names despite the JSON
            # schema. Keep these aliases at the execution boundary rather than
            # failing an otherwise valid and potentially expensive experiment.
            script_path = script_path or legacy_kwargs.pop("file", None) or legacy_kwargs.pop("path", None)
            metric_name = metric_name or legacy_kwargs.pop("metric", None)
            metric_mode = legacy_kwargs.pop("direction", metric_mode)
            if parameter_space is None:
                parameter_space = legacy_kwargs.pop("params", None)
            if legacy_kwargs:
                raise ValueError(f"Unsupported ML experiment arguments: {sorted(legacy_kwargs)}")
            if not script_path or not metric_name:
                raise ValueError("script_path (or file) and metric_name (or metric) are required")
            if metric_mode not in {"maximize", "minimize"} or metrics_format not in {"auto", "json", "csv"}:
                raise ValueError("metric_mode or metrics_format is invalid")
            if not 0 <= n_trials <= 1_000 or not 1 <= timeout <= 3_600:
                raise ValueError("n_trials must be 0-1000 and timeout must be 1-3600 seconds")
            parameter_space = self._normalize_parameter_space(parameter_space)
            self._validate_space(parameter_space)
            if n_trials and not parameter_space:
                raise ValueError("parameter_space is required when n_trials is greater than 0")
            baseline_params = baseline_params or {}
            unknown_baseline = set(baseline_params) - set(parameter_space)
            if unknown_baseline:
                raise ValueError(f"baseline_params contains parameters not in parameter_space: {sorted(unknown_baseline)}")
            reference_params = dict(baseline_params)
            for name, spec in parameter_space.items():
                if name not in reference_params:
                    if "baseline" in spec:
                        reference_params[name] = spec["baseline"]
                    elif spec["type"] == "categorical":
                        reference_params[name] = spec["choices"][0]
                    else:
                        reference_params[name] = spec["low"]
            source = self._resolve_path(script_path)
            self._require_workspace_path(source)
            if not source.is_file():
                raise FileNotFoundError(f"Training file not found: {script_path}")

            run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + source.stem
            experiment_dir = self.workspace_dir / ".mini_param_agent" / "experiments" / run_name
            experiment_dir.mkdir(parents=True, exist_ok=False)
            script = self._prepare_script(source, experiment_dir)
            command_template = train_command or "{python} {script}"
            records: list[dict[str, Any]] = []

            baseline = await self._run_trial(script, experiment_dir, 0, reference_params, command_template, metrics_format, timeout, seed)
            records.append(baseline)

            if n_trials:
                try:
                    import optuna
                except ImportError as exc:
                    raise RuntimeError("Optuna is required for n_trials > 0. Install with: pip install 'mini_param_agent[ml]' or pip install optuna") from exc

                sampler = optuna.samplers.TPESampler(seed=seed)
                study = optuna.create_study(direction=metric_mode, sampler=sampler)

                def objective(trial):
                    params: dict[str, Any] = {}
                    for name, spec in parameter_space.items():
                        if spec["type"] == "float":
                            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
                        elif spec["type"] == "int":
                            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1), log=bool(spec.get("log", False)))
                        else:
                            params[name] = trial.suggest_categorical(name, spec["choices"])
                    record = asyncio.run(self._run_trial(script, experiment_dir, trial.number + 1, params, command_template, metrics_format, timeout, seed))
                    records.append(record)
                    if record["status"] != "completed":
                        raise optuna.TrialPruned(record.get("error", "Training failed"))
                    try:
                        return self._metric_value(record, metric_name)
                    except ValueError as exc:
                        record.update({"status": "failed", "error": str(exc)})
                        raise optuna.TrialPruned(str(exc)) from exc

                # This tool is async but trials are deliberately serial: GPU memory and local training
                # code are commonly not safe to run concurrently.
                await asyncio.to_thread(study.optimize, objective, n_trials=n_trials)

            completed = [item for item in records if item.get("status") == "completed"]
            valid: list[tuple[float, dict[str, Any]]] = []
            for item in completed:
                try:
                    valid.append((self._metric_value(item, metric_name), item))
                except ValueError as exc:
                    item.update({"status": "failed", "error": str(exc)})
            if not valid:
                raise RuntimeError(f"No completed trial reported numeric metric '{metric_name}'. See {experiment_dir}")
            best_score, best = (max if metric_mode == "maximize" else min)(valid, key=lambda item: item[0])
            baseline_score: float | None = None
            try:
                baseline_score = self._metric_value(records[0], metric_name)
            except ValueError:
                pass
            improvement = None
            if baseline_score is not None:
                improvement = best_score - baseline_score if metric_mode == "maximize" else baseline_score - best_score
            report = {
                "source": str(source), "executable_script": str(script), "metric_name": metric_name,
                "metric_mode": metric_mode, "best_trial": best["trial"], "best_score": best_score,
                "baseline_score": baseline_score, "improvement_over_baseline": improvement,
                "best_params": best["params"], "best_metrics": best["metrics"], "records": records,
            }
            fieldnames = ["trial", "status", "params", "metrics", "error", "return_code"]
            with (experiment_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for record in records:
                    row = record.copy()
                    row["params"] = json.dumps(row.get("params", {}), ensure_ascii=False)
                    row["metrics"] = json.dumps(row.get("metrics", {}), ensure_ascii=False)
                    writer.writerow(row)
            if write_back_path:
                self._write_back(self._resolve_path(write_back_path), best["params"])
                report["write_back_path"] = str(self._resolve_path(write_back_path))
            (experiment_dir / "best_params.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

            return ToolResult(success=True, content=json.dumps({
                "experiment_dir": str(experiment_dir), "best_trial": best["trial"], "best_score": best_score,
                "baseline_score": baseline_score, "improvement_over_baseline": improvement,
                "best_params": best["params"], "report": str(experiment_dir / "best_params.json"),
                "results_csv": str(experiment_dir / "results.csv"), "write_back_path": report.get("write_back_path"),
            }, ensure_ascii=False, indent=2))
        except Exception as exc:
            return ToolResult(success=False, content="", error=f"ML experiment failed: {exc}")
