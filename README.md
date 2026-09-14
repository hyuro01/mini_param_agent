# mini_param_agent

English | [中文](README_CN.md)

mini_param_agent is a local CLI agent that can run reproducible hyperparameter experiments on a Python training script or Jupyter notebook. It also retains the existing file, shell, note, skill, MCP, and ACP capabilities. The project name does not imply automatic tuning of arbitrary code: the training program must accept trial parameters and report a validation metric.

## What it does

- Runs an agent loop with tool calls, retries, logging, and context summarization.
- Configurable recent-history retention, bounded summaries, optional `recall_context` keyword search, and resumable snapshots with a previous-generation backup. Shared POSIX filesystem mounts are supported; no distributed storage service is deployed. See [context configuration](docs/PRODUCTION_GUIDE.md#21-advanced-context-management).
- Reads, writes, and edits workspace files; runs shell commands; records and recalls session notes.
- Loads bundled skills and optional MCP tools, and exposes an ACP server for compatible editors.
- Runs a baseline plus Optuna trials for a single `.py` or convertible `.ipynb` training program. Trials receive parameters through `ML_EXPERIMENT_PARAMS` and report JSON/CSV metrics or `ML_METRICS` output. Results include trial logs, `results.csv`, and `best_params.json`. Source write-back is optional and explicit.
- Uses either the Anthropic or OpenAI client selected in configuration, including compatible local model endpoints when correctly configured.

## Available tools

The CLI registers these tools (individual groups can be disabled in `config.yaml`):

- `read_file`, `write_file`, `edit_file`: safely inspect and modify files inside the selected workspace.
- `bash`, `bash_output`, `bash_kill`: run foreground or background shell commands and inspect or stop background jobs.
- `record_note`, `recall_notes`: persist and retrieve session notes in the workspace.
- `recall_context`: enabled with `context.enable_recall: true`; keyword search over this session's compacted history, separate from manually recorded notes.
- `get_skill`: load the full instructions for one of the bundled skills on demand; skill metadata is injected into the system prompt.
- `run_ml_experiment`: run a baseline and Optuna trials for a `.py`/`.ipynb` training program, collect metrics, and optionally write back the best parameters.
- Configured MCP tools: external tools loaded from an `mcp.json` file when MCP is enabled.

The exact tool schemas are exposed to the model at runtime. File tools check workspace paths, but shell commands and training programs are not sandboxed: execute trusted code only. Context storage can explicitly target a shared directory outside the workspace; restrict its access permissions.

## MCP example configuration

[`mini_param_agent/config/mcp-example.json`](mini_param_agent/config/mcp-example.json) is a template, not an active configuration. It defines two optional MCP servers, both disabled by default:

- `minimax_search` starts an external web-search server through `uvx`; enable it only after supplying the required API keys.
- `memory` starts the official MCP knowledge-graph memory server through `npx`.

Copy it to `mcp.json` in the configured search path, set `disabled` to `false` only for servers you trust, and ensure `uvx`/`npx` and their dependencies are available. Never commit secrets in this file.

## Install and run

From this project directory, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --extra ml
uv run mini_param_agent --workspace .
```

For an editable command available outside the project directory:

```bash
uv tool install --editable '.[ml]'
mini_param_agent --workspace /path/to/project
```

The base installation can omit `--extra ml` or `[ml]`, but Optuna trials require the optional ML dependencies. A fresh installation may need network access to resolve packages. The executable names are `mini_param_agent` and `mini_param_agent_acp`; `python -m mini_param_agent.cli` also works. Old executable aliases are not installed by this project.

## Configure

The CLI reads `mini_param_agent/config/config.yaml` in development mode, then `~/.mini_param_agent/config/config.yaml`, then the packaged config. Copy the template into the development path if needed:

```bash
cp mini_param_agent/config/config-example.yaml mini_param_agent/config/config.yaml
```

Set `api_key`, `api_base`, `model`, and `provider` (`anthropic` or `openai`) in that file. You can also set `max_steps`, `workspace_dir`, retry options, and `tools` switches such as `enable_ml_experiment`. New user configuration and logs use `~/.mini_param_agent`; new workspace experiments use `.mini_param_agent/experiments`. Existing workspace experiments were moved to this path without changing their contents. Do not commit a real API key.

## Run an ML experiment

The training script must read `ML_EXPERIMENT_PARAMS` and write a metric to `ML_EXPERIMENT_METRICS_PATH` (or print a final `ML_METRICS: {...}` line). See [the RBF SVC example](examples/ml/tune_rbf_svc_moons.py) and the [production and ML guide](docs/PRODUCTION_GUIDE.md).

```text
Call run_ml_experiment on examples/ml/tune_rbf_svc_moons.py.
Optimize val_accuracy (maximize). Search C and gamma as log-scale floats
from 0.01 to 100 and 0.01 to 10. Use baseline C=1.0, gamma=1.0,
20 trials, seed 42, and a 60-second per-trial timeout.
Report the baseline, best score, best parameters, report path, and CSV path.
Do not write back to the source file.
```

To write the best parameter dictionary to a marked Python source file or a JSON config, explicitly supply `write_back_path`. The tool does not rewrite arbitrary model code or the original notebook. Experiments execute the supplied training program, so use files you trust.

## Other usage

```bash
uv run mini_param_agent --task "List the available tools" --workspace .
uv run python -m mini_param_agent.cli
uv run mini_param_agent --version
uv run mini_param_agent log
```

The [examples directory](examples/README.md) contains basic tools, simple and full agents, session notes, provider selection, tool schemas, and ML training examples. For feature details, ML experiments, troubleshooting, upgrade directions, and deployment, see the [production guide](docs/PRODUCTION_GUIDE.md).

## Test

```bash
uv run pytest -q tests/test_tools.py tests/test_agent_empty_response.py tests/test_openai_compat.py
uv run pytest -q
```

Some integration tests require external services or a writable home-directory log path.

## License

[MIT](LICENSE).
