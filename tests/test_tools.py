"""Test cases for tools."""

import asyncio
import ast
import json
import tempfile
from pathlib import Path

import pytest

from mini_param_agent.tools import BashTool, EditTool, MLExperimentTool, ReadTool, WriteTool


@pytest.mark.asyncio
async def test_read_tool():
    """Test read file tool."""
    print("\n=== Testing ReadTool ===")

    # Create a temp file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello, World!")
        temp_path = f.name

    try:
        tool = ReadTool()
        result = await tool.execute(path=temp_path)

        assert result.success, f"Read failed: {result.error}"
        # ReadTool now returns content with line numbers in format: "LINE_NUMBER|LINE_CONTENT"
        assert "Hello, World!" in result.content, f"Content mismatch: {result.content}"
        assert "|Hello, World!" in result.content, f"Expected line number format: {result.content}"
        print("✅ ReadTool test passed")
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_write_tool():
    """Test write file tool."""
    print("\n=== Testing WriteTool ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.txt"

        tool = WriteTool()
        result = await tool.execute(path=str(file_path), content="Test content")

        assert result.success, f"Write failed: {result.error}"
        assert file_path.exists(), "File was not created"
        assert file_path.read_text() == "Test content", "Content mismatch"
        print("✅ WriteTool test passed")


@pytest.mark.asyncio
async def test_edit_tool():
    """Test edit file tool."""
    print("\n=== Testing EditTool ===")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello, World!")
        temp_path = f.name

    try:
        tool = EditTool()
        result = await tool.execute(
            path=temp_path, old_str="World", new_str="Agent"
        )

        assert result.success, f"Edit failed: {result.error}"
        content = Path(temp_path).read_text()
        assert content == "Hello, Agent!", f"Content mismatch: {content}"
        print("✅ EditTool test passed")
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_bash_tool():
    """Test bash command tool."""
    print("\n=== Testing BashTool ===")

    tool = BashTool()

    # Test successful command
    result = await tool.execute(command="echo 'Hello from bash'")
    assert result.success, f"Bash failed: {result.error}"
    assert "Hello from bash" in result.content, f"Output mismatch: {result.content}"
    print("✅ BashTool test passed")

    # Test failed command
    result = await tool.execute(command="exit 1")
    assert not result.success, "Command should have failed"
    print("✅ BashTool error handling test passed")


@pytest.mark.asyncio
async def test_ml_experiment_baseline_and_write_back():
    """The ML tool can run a standard script and preserve a reproducible report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        script = workspace / "train.py"
        script.write_text(
            "import json, os\n"
            "with open(os.environ['ML_EXPERIMENT_METRICS_PATH'], 'w') as f:\n"
            "    json.dump({'val_accuracy': 0.75, 'train_loss': 0.2}, f)\n",
            encoding="utf-8",
        )
        config = workspace / "best_config.json"
        tool = MLExperimentTool(workspace_dir=tmpdir)
        result = await tool.execute(
            script_path="train.py",
            metric_name="val_accuracy",
            n_trials=0,
            write_back_path="best_config.json",
        )

        assert result.success, result.error
        payload = json.loads(result.content)
        assert payload["best_score"] == 0.75
        assert payload["baseline_score"] == 0.75
        assert payload["improvement_over_baseline"] == 0.0
        assert Path(payload["report"]).exists()
        assert Path(payload["results_csv"]).exists()
        assert json.loads(config.read_text()) == {}


@pytest.mark.asyncio
async def test_ml_experiment_accepts_local_model_argument_aliases():
    """Small local models often call generic file/metric/params names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "train.py").write_text(
            "import json, os\n"
            "with open(os.environ['ML_EXPERIMENT_METRICS_PATH'], 'w') as f:\n"
            "    json.dump({'val_accuracy': 0.8}, f)\n",
            encoding="utf-8",
        )
        result = await MLExperimentTool(workspace_dir=tmpdir).execute(
            file="train.py",
            metric="val_accuracy",
            direction="maximize",
            params='{"C": {"type": "float", "min": 0.1, "max": 1.0}}',
            n_trials=0,
        )

        assert result.success, result.error
        assert json.loads(result.content)["best_score"] == 0.8


@pytest.mark.asyncio
async def test_ml_experiment_python_write_back_produces_valid_python(tmp_path):
    """A boolean categorical optimum must be written as True, not JSON true."""
    script = tmp_path / "train.py"
    script.write_text(
        "import json, os\n"
        "# ML_EXPERIMENT_PARAMS_START\n"
        "ML_EXPERIMENT_PARAMS = {'use_feature': False}\n"
        "# ML_EXPERIMENT_PARAMS_END\n"
        "params = json.loads(os.environ['ML_EXPERIMENT_PARAMS'])\n"
        "with open(os.environ['ML_EXPERIMENT_METRICS_PATH'], 'w') as f:\n"
        "    json.dump({'val_accuracy': float(params['use_feature'])}, f)\n",
        encoding="utf-8",
    )
    result = await MLExperimentTool(workspace_dir=str(tmp_path)).execute(
        script_path="train.py",
        metric_name="val_accuracy",
        parameter_space={"use_feature": {"type": "categorical", "choices": [False, True], "baseline": True}},
        n_trials=0,
        write_back_path="train.py",
    )

    assert result.success, result.error
    tree = ast.parse(script.read_text(encoding="utf-8"))
    assigned = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ML_EXPERIMENT_PARAMS" for target in node.targets)
    )
    assert ast.literal_eval(assigned) == {"use_feature": True}
    assert json.loads(result.content)["write_back_path"] == str(script)


@pytest.mark.asyncio
async def test_ml_experiment_optuna_search():
    """When the optional dependency is installed, Optuna trials select the best score."""
    pytest.importorskip("optuna")
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "train.py").write_text(
            "import json, os\n"
            "value = int(json.loads(os.environ['ML_EXPERIMENT_PARAMS']).get('x', 0))\n"
            "with open(os.environ['ML_EXPERIMENT_METRICS_PATH'], 'w') as f:\n"
            "    json.dump({'val_score': -(value - 3) ** 2}, f)\n",
            encoding="utf-8",
        )
        result = await MLExperimentTool(workspace_dir=tmpdir).execute(
            script_path="train.py",
            metric_name="val_score",
            parameter_space={"x": {"type": "int", "low": 1, "high": 5}},
            n_trials=8,
            seed=7,
        )

        assert result.success, result.error
        payload = json.loads(result.content)
        assert payload["best_score"] == 0.0
        assert payload["best_params"] == {"x": 3}


async def main():
    """Run all tool tests."""
    print("=" * 80)
    print("Running Tool Tests")
    print("=" * 80)

    await test_read_tool()
    await test_write_tool()
    await test_edit_tool()
    await test_bash_tool()

    print("\n" + "=" * 80)
    print("All tool tests passed! ✅")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
