# mini_param_agent 的 MLExperimentTool

关于如何写不同模型的调参 prompt、参数是否自动写回 `.py`、以及 Optuna 是否从最小值逐渐搜索到最大值，请参阅 [调参与写回常见问题](ML_EXPERIMENT_TOOL.md)。

`run_ml_experiment` 面向可重复执行的单文件训练程序（`.py`）及可转换的
Jupyter notebook（`.ipynb`）。它先运行基线，然后用 Optuna 的 TPE 搜索参数，
在工作区的 `.mini_param_agent/experiments/<时间>_<脚本名>/` 兼容目录保存每个 trial 的命令、
参数、标准输出、标准错误、指标、`results.csv` 与 `best_params.json`。

安装可选依赖：

```bash
uv sync --extra ml
```

训练程序需要通过下列任一方式报告指标：

1. 写入 `ML_EXPERIMENT_METRICS_PATH` 指向的 JSON 对象或单行 CSV；
2. 在 stdout 最后一行输出 `ML_METRICS: {"val_accuracy": 0.91}`。

每个 trial 自动提供以下环境变量：

- `ML_EXPERIMENT_PARAMS`：本次参数 JSON。
- `ML_PARAM_<参数名>`：单个参数，例如 `ML_PARAM_LR`。
- `ML_EXPERIMENT_METRICS_PATH`：指标文件应写入的位置。
- `ML_EXPERIMENT_TRIAL_DIR`：本次 trial 的结果目录。
- `ML_EXPERIMENT_SEED`：指定给实验的随机种子。

仓库还附带了可立即测试的 [scikit-learn 示例](../examples/ml/tune_logistic_regression.py)。
它不需要下载数据集，使用合成二分类数据，优化 `val_accuracy`。
若需要明显观察调参效果，请使用非线性的 [RBF SVC 示例](../examples/ml/tune_rbf_svc_moons.py)：
它的 `C` 和 `gamma` 会显著影响模型边界。

例如，训练脚本可以这样读取参数并报告验证集准确率：

```python
import json
import os

params = json.loads(os.environ["ML_EXPERIMENT_PARAMS"])
# 用 params.get("lr", 1e-3) 配置 PyTorch 或 scikit-learn 模型并训练
metrics = {"val_accuracy": validation_accuracy}
with open(os.environ["ML_EXPERIMENT_METRICS_PATH"], "w", encoding="utf-8") as f:
    json.dump(metrics, f)
```

调用工具时提供参数空间，例如：

```json
{
  "script_path": "train.py",
  "metric_name": "val_accuracy",
  "metric_mode": "maximize",
  "parameter_space": {
    "lr": {"type": "float", "low": 0.0001, "high": 0.01, "log": true, "baseline": 0.001},
    "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
    "epochs": {"type": "int", "low": 3, "high": 15}
  },
  "n_trials": 20,
  "train_command": "{python} {script} --lr {lr} --batch-size {batch_size}",
  "timeout": 900,
  "write_back_path": "configs/best_params.json"
}
```

工具的标准参数名是 `script_path`、`metric_name`、`metric_mode`、`parameter_space`。
为兼容本地小模型，也接受 `file`、`metric`、`direction`、`params`；`params` 可为 JSON
字符串，且 `min`/`max` 会自动转换为 `low`/`high`。

基线会使用 `baseline_params`（若提供），然后是每个参数定义中的 `baseline`；若两者
均未指定，则使用范围下界或类别列表的第一个值。这样，即使 `train_command` 使用了
`{lr}` 一类的模板变量，基线仍是一个可执行、可比较的参考配置。

`write_back_path` 只会在显式传入时写入。它可以是 JSON 配置文件（最佳参数会合并
进去），或者已经包含以下标记的 Python 文件：

```python
# ML_EXPERIMENT_PARAMS_START
ML_EXPERIMENT_PARAMS = {}
# ML_EXPERIMENT_PARAMS_END
```

为避免测试集泄漏，应使用验证集指标作为 `metric_name`；获得最佳参数后，单独在测试集
执行一次最终评估。工具按顺序执行 trial，避免常见的单 GPU 显存冲突。
