"""The ML contract rejects obvious mistakes and preserves auditable failures."""

import hashlib
import json
from pathlib import Path

import pytest

from mini_param_agent.tools.ml_experiment_contract import failure_fact, finite_metric
from mini_param_agent.tools.ml_experiment_tool import MLExperimentTool


@pytest.mark.asyncio
async def test_static_preflight_rejects_unused_parameter_before_execution(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import json, os\n"
        "open('should_not_run', 'w').write('bad')\n"
        "open(os.environ['ML_EXPERIMENT_METRICS_PATH'], 'w').write(json.dumps({'val_accuracy': 0.5}))\n"
    )
    result = await MLExperimentTool(str(tmp_path)).execute(
        script_path="train.py", metric_name="val_accuracy",
        parameter_space={"learning_rate": {"type": "float", "low": 0.001, "high": 0.1}}, n_trials=1)
    assert not result.success
    assert "learning_rate" in result.error
    assert "Static preflight failed" in result.error
    assert not (tmp_path / "should_not_run").exists()


@pytest.mark.asyncio
async def test_invalid_search_bounds_rejected_before_training(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('no training should run')\n")
    result = await MLExperimentTool(str(tmp_path)).execute(
        script_path="train.py", metric_name="val_accuracy", n_trials=1,
        parameter_space={"C": {"type": "float", "low": 10.0, "high": 0.1}})
    assert not result.success
    assert "low <= high" in result.error
    assert not (tmp_path / ".mini_param_agent").exists()


@pytest.mark.asyncio
async def test_evidence_contract_and_reproducible_baseline(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import json, os\n"
        "params = json.loads(os.environ['ML_EXPERIMENT_PARAMS'])\n"
        "with open(os.environ['ML_EXPERIMENT_METRICS_PATH'], 'w') as stream:\n"
        "    json.dump({'val_accuracy': float(params.get('C', 1.0))}, stream)\n"
    )
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    result = await MLExperimentTool(str(tmp_path)).execute(
        script_path="train.py", metric_name="val_accuracy", n_trials=0,
        parameter_space={"C": {"type": "float", "low": 0.1, "high": 2.0}},
        baseline_params={"C": 1.0}, seed=7)
    assert result.success, result.error
    output = json.loads(result.content)
    contract = json.loads(Path(output["contract"]).read_text())
    evidence = json.loads(Path(output["evidence"]).read_text())
    assert contract["source_sha256"] == digest
    assert contract["seed"] == 7
    assert contract["metric_name"] == "val_accuracy"
    assert contract["static_check"]["parameter_checks"]["C"] == ["literal_key"]
    assert evidence["best_trial"] == 0
    assert evidence["best_score"] == 1.0
    assert evidence["baseline"]["command"][0]
    assert evidence["source_sha256"] == digest
    assert evidence["failure_facts"] == []
    assert Path(output["failure_facts"]).read_text().strip() == "[]"
    assert Path(output["report_md"]).exists()
    assert Path(output["report"]).exists()


@pytest.mark.asyncio
async def test_failed_training_still_writes_evidence_and_actionable_fact(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("import sys\nprint('broken training', file=sys.stderr)\nsys.exit(3)\n")
    result = await MLExperimentTool(str(tmp_path)).execute(
        script_path="train.py", metric_name="val_accuracy", n_trials=0)
    assert not result.success
    assert "training_exit" in result.error
    evidence_path = Path(result.error.split("evidence: ", 1)[1].split("; failure facts:", 1)[0])
    evidence = json.loads(evidence_path.read_text())
    fact = evidence["failure_facts"][0]
    assert fact["error_type"] == "training_exit"
    assert fact["return_code"] == 3
    assert "broken training" in Path(fact["stderr_log"]).read_text()
    assert Path(evidence["artifacts"]["results_csv"]).exists()
    assert evidence["best_trial"] is None


@pytest.mark.asyncio
async def test_missing_and_nonfinite_metric_are_failures(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("import json, os\nopen(os.environ['ML_EXPERIMENT_METRICS_PATH'], 'w').write(json.dumps({'val_accuracy': 'nan'}))\n")
    result = await MLExperimentTool(str(tmp_path)).execute(script_path="train.py", metric_name="val_accuracy", n_trials=0)
    assert not result.success
    assert "invalid_metric" in result.error
    with pytest.raises(ValueError, match="finite"):
        finite_metric(float("inf"), "val_accuracy", 1)
    assert failure_fact({"trial": 2, "status": "timeout", "error": "Timed out"}, "val_accuracy")["error_type"] == "timeout"


@pytest.mark.asyncio
async def test_wrong_metric_key_is_reported_in_preflight_and_failure_facts(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("import json, os\nopen(os.environ['ML_EXPERIMENT_METRICS_PATH'], 'w').write(json.dumps({'train_accuracy': 0.9}))\n")
    result = await MLExperimentTool(str(tmp_path)).execute(script_path="train.py", metric_name="val_accuracy", n_trials=0)
    assert not result.success
    evidence_path = Path(result.error.split("evidence: ", 1)[1].split("; failure facts:", 1)[0])
    evidence = json.loads(evidence_path.read_text())
    contract = json.loads(Path(evidence["contract_path"]).read_text())
    assert contract["static_check"]["status"] == "warning"
    assert "val_accuracy" in contract["static_check"]["warnings"][0]
    assert evidence["failure_facts"][0]["error_type"] == "invalid_metric"


@pytest.mark.asyncio
async def test_warn_mode_allows_dynamic_parameter_lookup(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import json, os\n"
        "params = json.loads(os.environ['ML_EXPERIMENT_PARAMS'])\n"
        "score = float(next(iter(params.values())))\n"
        "open(os.environ['ML_EXPERIMENT_METRICS_PATH'], 'w').write(json.dumps({'val_accuracy': score}))\n"
    )
    result = await MLExperimentTool(str(tmp_path)).execute(
        script_path="train.py", metric_name="val_accuracy", n_trials=1,
        parameter_space={"C": {"type": "float", "low": 0.1, "high": 1.0}},
        static_check_mode="warn")
    assert result.success, result.error
    output = json.loads(result.content)
    assert output["static_check"]["status"] == "warning"
    assert output["best_score"] is not None


@pytest.mark.asyncio
async def test_failed_baseline_is_returned_with_successful_trials(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import json, os, sys\n"
        "params = json.loads(os.environ['ML_EXPERIMENT_PARAMS'])\n"
        "if params['x'] == 0:\n"
        "    print('baseline failed', file=sys.stderr)\n"
        "    sys.exit(4)\n"
        "open(os.environ['ML_EXPERIMENT_METRICS_PATH'], 'w').write(json.dumps({'val_accuracy': 0.8}))\n"
    )
    result = await MLExperimentTool(str(tmp_path)).execute(
        script_path="train.py", metric_name="val_accuracy", n_trials=2,
        parameter_space={"x": {"type": "categorical", "choices": [1]}},
        baseline_params={"x": 0})
    assert result.success, result.error
    output = json.loads(result.content)
    assert output["baseline_score"] is None
    assert output["best_score"] == 0.8
    assert output["failed_trials"] == 1
    assert output["failures"][0]["error_type"] == "training_exit"
