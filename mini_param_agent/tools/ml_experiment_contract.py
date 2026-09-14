"""Static preflight and versioned evidence for single-script ML experiments."""

import ast
import hashlib
from importlib import metadata
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter
import sys
from typing import Any

from pydantic import BaseModel, Field


class ExperimentContract(BaseModel):
    schema_version: int = 1
    source: str
    source_sha256: str
    metric_name: str
    metric_mode: str
    parameter_space: dict[str, Any]
    baseline_params: dict[str, Any]
    n_trials: int
    seed: int
    timeout_seconds: int
    metrics_format: str
    train_command: str
    static_check: dict[str, Any]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EvidenceBundle(BaseModel):
    schema_version: int = 1
    contract_path: str
    source_sha256: str
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=platform.platform)
    dependency_versions: dict[str, str] = Field(default_factory=lambda: installed_versions())
    baseline: dict[str, Any]
    trials: list[dict[str, Any]]
    best_trial: int | None
    best_score: float | None
    best_params: dict[str, Any] | None
    baseline_score: float | None
    improvement_over_baseline: float | None
    failure_facts: list[dict[str, Any]]
    artifacts: dict[str, str]


def inspect_training_script(source: Path, parameters: dict[str, Any], metric: str, command: str, n_trials: int, strict: bool = True) -> dict[str, Any]:
    """Inspect syntax and literal references without importing or running user code.

    This establishes evidence of *references*, not data flow into a model or
    actual metric production; runtime validation remains authoritative.
    """
    warnings: list[str] = []
    errors: list[str] = []
    try:
        if source.suffix == ".ipynb":
            notebook = json.loads(source.read_text(encoding="utf-8"))
            cells = notebook.get("cells", [])
            code = "\n".join("".join(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "code")
            if not code.strip():
                errors.append("Notebook contains no code cells")
        else:
            code = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot read training source for static inspection: {exc}") from exc

    try:
        tree = ast.parse(code, filename=str(source))
        literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        parse_status = "parsed"
    except SyntaxError as exc:
        if source.suffix == ".py":
            errors.append(f"Python syntax error at line {exc.lineno}: {exc.msg}")
        else:
            warnings.append("Notebook code could not be parsed statically; nbconvert/runtime will verify it")
        literals = set()
        parse_status = "unparsed"

    placeholders = {field.split(".")[0].split("[")[0] for _, field, _, _ in Formatter().parse(command) if field}
    parameter_checks = {}
    for name in parameters:
        channels = []
        if name in literals:
            channels.append("literal_key")
        if f"ML_PARAM_{name.upper()}" in literals:
            channels.append("individual_environment_variable")
        if name in placeholders:
            channels.append("command_placeholder")
        if f"--{name.replace('_', '-')}" in literals or f"--{name}" in literals:
            channels.append("cli_option")
        parameter_checks[name] = channels
        if not channels:
            message = f"Parameter '{name}' has no visible literal/env/command reference; static inspection cannot confirm it reaches training"
            # Dynamic lookup and notebook magics can make a valid reference
            # unprovable; prefer an explicit warning in those cases.
            if parse_status == "parsed" and n_trials and strict:
                errors.append(message)
            else:
                warnings.append(message)

    metric_channels = []
    if metric in literals:
        metric_channels.append("literal_key")
    if "ML_EXPERIMENT_METRICS_PATH" in literals or "{metrics_path}" in command:
        metric_channels.append("metrics_file")
    if "ML_METRICS:" in code:
        metric_channels.append("stdout_marker")
    if metric not in literals:
        warnings.append(f"Metric key '{metric}' is not visible as a literal; runtime must verify it")
    if not {"metrics_file", "stdout_marker"} & set(metric_channels):
        warnings.append("No metrics file or ML_METRICS output channel is visible; runtime must verify it")

    return {"status": "error" if errors else ("warning" if warnings else "ok"),
            "parse_status": parse_status, "parameter_checks": parameter_checks,
            "metric_channels": metric_channels, "warnings": warnings, "errors": errors,
            "limitation": "References are syntactic clues; model use and metric validity are checked by trial execution."}


def source_digest(source: Path) -> str:
    return hashlib.sha256(source.read_bytes()).hexdigest()


def installed_versions() -> dict[str, str]:
    names = ("optuna", "scikit-learn", "torch", "nbconvert", "numpy", "scipy")
    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def failure_fact(record: dict[str, Any], metric_name: str) -> dict[str, Any] | None:
    if record.get("status") == "completed":
        return None
    error = str(record.get("error") or "Training failed")
    if record.get("status") == "timeout":
        kind, suggestion = "timeout", "Increase timeout or reduce training workload"
    elif error.startswith("No metrics found"):
        kind, suggestion = "missing_metrics", "Write metrics to ML_EXPERIMENT_METRICS_PATH or print ML_METRICS: {...}"
    elif "did not report numeric metric" in error:
        kind, suggestion = "invalid_metric", f"Output a finite numeric '{metric_name}' metric"
    elif record.get("return_code") not in (None, 0):
        kind, suggestion = "training_exit", "Inspect stderr.log and the training command"
    else:
        kind, suggestion = "execution_error", "Inspect the failure message and training environment"
    return {"trial": record.get("trial"), "params": record.get("params", {}),
            "error_type": kind, "message": error[-2000:], "return_code": record.get("return_code"),
            "stdout_log": record.get("stdout_log"), "stderr_log": record.get("stderr_log"),
            "suggested_action": suggestion, "recoverable": kind in {"timeout", "missing_metrics", "invalid_metric"}}


def finite_metric(value: Any, metric_name: str, trial: int) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Trial {trial} did not report numeric metric '{metric_name}'") from exc
    if not math.isfinite(score):
        raise ValueError(f"Trial {trial} did not report finite numeric metric '{metric_name}'")
    return score
