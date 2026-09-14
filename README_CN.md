# mini_param_agent

[English](README.md) | 中文

mini_param_agent 是一个在本地运行的 CLI Agent，可对单个 Python 训练脚本或 Jupyter Notebook 执行可复现的超参数实验。项目保留文件、Shell、笔记、Skills、MCP 和 ACP 等已有能力。它不会凭名称自动调任意代码：训练程序必须接收 trial 参数并报告验证指标。

## 已实现的能力

- Agent 循环、工具调用、重试、日志与上下文摘要。
- 工作区文件读取、写入、编辑、Shell 命令以及会话笔记记录和检索。
- 加载随包提供的 Skills 和可选 MCP 工具，并提供 ACP 服务供兼容编辑器使用。
- 对单个 `.py` 或可转换的 `.ipynb` 先运行基线，再使用 Optuna 搜索。训练程序通过 `ML_EXPERIMENT_PARAMS` 接收参数，以 JSON/CSV 或 `ML_METRICS` 输出指标；保存 trial 日志、`results.csv` 和 `best_params.json`。回写源文件须显式指定。
- 通过配置选择 Anthropic 或 OpenAI 客户端；正确配置后可接入兼容 API 的本地模型服务。

## 当前可用的 Tools

CLI 会注册以下工具（可在 `config.yaml` 中关闭对应工具组）：

- `read_file`、`write_file`、`edit_file`：在选定工作区内读取和修改文件。
- `bash`、`bash_output`、`bash_kill`：执行前台或后台 Shell 命令，查看或终止后台任务。
- `record_note`、`recall_notes`：在工作区保存和检索会话笔记。
- `get_skill`：按需加载随包提供的某个 Skill 的完整说明；Skill 元数据会注入系统提示词。
- `run_ml_experiment`：对 `.py`/`.ipynb` 训练程序运行基线和 Optuna trial，收集指标，并可选回写最佳参数。
- 配置的 MCP 工具：启用 MCP 且存在 `mcp.json` 时，从外部 MCP 服务加载工具。

工具的精确参数模式会在运行时提供给模型。工具都以工作区为边界；ML 实验会执行你传入的训练程序，请只使用可信代码。

## `mcp-example.json` 是什么？

[`mini_param_agent/config/mcp-example.json`](mini_param_agent/config/mcp-example.json) 是 MCP 配置模板，不是启用中的配置。它定义了两个默认禁用的可选服务器：

- `minimax_search`：通过 `uvx` 启动外部网页搜索服务，启用前需要填入相应 API Key。
- `memory`：通过 `npx` 启动官方 MCP 知识图谱记忆服务。

如需使用，可复制为搜索路径中的 `mcp.json`，只把信任的服务器 `disabled` 改为 `false`，并确保已安装 `uvx`/`npx` 及其依赖。不要把密钥提交到 GitHub。

## 安装与运行

在项目目录安装 [uv](https://docs.astral.sh/uv/) 后运行：

```bash
uv sync --extra ml
uv run mini_param_agent --workspace .
```

若希望在其他目录也能使用命令，可在项目目录执行：

```bash
uv tool install --editable '.[ml]'
mini_param_agent --workspace /path/to/project
```

基础安装可以不加 `--extra ml` 或 `[ml]`，但 Optuna trial 需要 ML 可选依赖。首次安装可能需要联网解析依赖。新命令为 `mini_param_agent` 和 `mini_param_agent_acp`，也可运行 `python -m mini_param_agent.cli`；本项目不再安装旧命令别名。

## 配置

CLI 依次搜索开发目录的 `mini_param_agent/config/config.yaml`、`~/.mini_param_agent/config/config.yaml` 和包内配置。如需从模板开始：

```bash
cp mini_param_agent/config/config-example.yaml mini_param_agent/config/config.yaml
```

在文件中设置 `api_key`、`api_base`、`model`、`provider`（`anthropic` 或 `openai`）；还可以设置 `max_steps`、`workspace_dir`、重试参数，以及 `enable_ml_experiment` 等 `tools` 开关。新的用户配置和日志使用 `~/.mini_param_agent`，工作区实验使用 `.mini_param_agent/experiments`；已有工作区实验已原样移动到新目录。请勿提交真实 API Key。本次改名不会改动已有的本地模型配置。

## 运行调参实验

训练脚本必须读取 `ML_EXPERIMENT_PARAMS`，并向 `ML_EXPERIMENT_METRICS_PATH` 写入指标；也可以在最后输出 `ML_METRICS: {...}`。参见 [RBF SVC 示例](examples/ml/tune_rbf_svc_moons.py)和[完整调参说明](docs/ML_EXPERIMENT_TOOL.md)。

```text
请调用 run_ml_experiment，训练文件为 examples/ml/tune_rbf_svc_moons.py。
优化 val_accuracy，方向 maximize。搜索 C（0.01～100）和 gamma（0.01～10），
两者都是 log 尺度的 float；基线 C=1.0、gamma=1.0。
运行 20 次 trial，seed 42，单次 timeout 60 秒。
报告基线分数、最佳分数、最佳参数、报告路径和 CSV 路径。不要回写源文件。
```

如果希望把最佳参数写进带标记的 Python 文件或 JSON 配置，须显式给出 `write_back_path`。工具不会重写任意模型代码，也不会回写原始 Notebook。实验会执行传入的训练文件，因此只使用可信代码。

## 其他用法

```bash
uv run mini_param_agent --task "列出可用工具" --workspace .
uv run python -m mini_param_agent.cli
uv run mini_param_agent --version
uv run mini_param_agent log
```

[示例目录](examples/README_CN.md)包含基础工具、简单与完整 Agent、会话笔记、模型服务选择、工具模式和 ML 训练示例。配置与扩展见[开发指南](docs/DEVELOPMENT_GUIDE_CN.md)；ACP 和部署见[生产指南](docs/PRODUCTION_GUIDE_CN.md)。

## 测试

```bash
uv run pytest -q tests/test_tools.py tests/test_agent_empty_response.py tests/test_openai_compat.py
uv run pytest -q
```

部分集成测试需要外部服务或对主目录日志路径的写入权限。

## 许可证

[MIT](LICENSE)。
