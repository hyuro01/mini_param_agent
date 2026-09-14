# MLExperimentTool：怎样选择参数、理解搜索、保存最优配置

本文是 mini_param_agent 的 [工具使用说明](ML_EXPERIMENT_TOOL_CN.md) 常见问题补充。`run_ml_experiment` 运行一次基线和 `n_trials` 次 Optuna 搜索；每次搜索都会重新运行训练脚本，并以指定的**验证集指标**比较结果。

## 怎样写调参 prompt？

先确认训练脚本真的读取 `ML_EXPERIMENT_PARAMS`，并把验证指标写到 `ML_EXPERIMENT_METRICS_PATH`。参数名必须与脚本读取的键一致；只在 prompt 里写一个脚本没有使用的参数，不会改变训练。

可以直接用仓库里的 RBF SVC 示例观察 `C` 与 `gamma` 的影响：

```text
请调用 run_ml_experiment。
script_path: examples/ml/tune_rbf_svc_moons.py
metric_name: val_accuracy
metric_mode: maximize
parameter_space:
  C: {type: float, low: 0.01, high: 100, log: true, baseline: 1.0}
  gamma: {type: float, low: 0.01, high: 10, log: true, baseline: 1.0}
n_trials: 20
seed: 42
timeout: 60
先不要回写训练脚本。完成后给出基线分数、最佳验证分数、提升幅度、最佳参数、报告和 CSV 路径。
```

对自己的文件，按所用模型选择参数。不要把下面所有参数同时放进一个不支持它们的脚本：

| 模型 | 适合尝试的参数 | 搜索空间例子 |
| --- | --- | --- |
| 逻辑回归 / RBF SVC | `C`，RBF SVC 另有 `gamma` | `C: {type: float, low: 0.001, high: 100, log: true}` |
| 随机森林 | `max_depth`、`n_estimators`、`min_samples_leaf` | `max_depth: {type: int, low: 2, high: 20}`；`n_estimators: {type: int, low: 50, high: 500, step: 50}` |
| PyTorch 模型 | 学习率 `lr`、`weight_decay`、`dropout`、`batch_size` | `lr: {type: float, low: 0.0001, high: 0.01, log: true}`；`batch_size: {type: categorical, choices: [32, 64, 128]}` |

例如要调自己的 PyTorch 脚本，可以把上面的首段 prompt 换成：

```text
请调用 run_ml_experiment，训练文件是 train.py，优化 val_accuracy（maximize）。
参数空间：
lr: {type: float, low: 0.0001, high: 0.01, log: true, baseline: 0.001}
weight_decay: {type: float, low: 0.000001, high: 0.01, log: true, baseline: 0.0001}
dropout: {type: float, low: 0.0, high: 0.5, baseline: 0.2}
batch_size: {type: categorical, choices: [32, 64, 128], baseline: 64}
先做基线，再运行 20 次 trial；seed 42，单次 timeout 600 秒。报告基线与最佳分数、最佳参数、报告路径。不要回写源文件。
```

这要求 `train.py` 已从环境变量读取上述四个参数，并对每个配置训练、报告同一验证集指标。请把示例里的 `baseline` 改成你当前脚本实际使用的值，才有可信的“相比基线提升”。如果原脚本没有这个接口，先让 Agent 检查并接入，再单独发出上面的调参请求。`n_trials=20` 表示基线以外再运行 20 次训练，因此成本约为 21 次训练。搜索结果也不保证高于基线。

若训练的是随机森林，把文件与参数换成：

```text
请调用 run_ml_experiment，script_path 为 train_random_forest.py。
优化验证集 val_accuracy，metric_mode 为 maximize。
parameter_space:
  max_depth: {type: int, low: 2, high: 20, baseline: 6}
  n_estimators: {type: int, low: 50, high: 500, step: 50, baseline: 100}
  min_samples_leaf: {type: int, low: 1, high: 10, baseline: 1}
n_trials: 20，seed: 42，timeout: 300。
不要回写源文件，报告基线与最佳验证分数和结果文件路径。
```

## 最好的参数会自动写进 `.py` 吗？

**默认不会。** 无论最优配置来自基线还是搜索 trial，工具都会写 `best_params.json` 和 `results.csv`，但不会改源文件。

如果希望明确写回，请在 prompt 中加入 `write_back_path`。写回 JSON 配置文件时，最佳参数会合并到该文件；写回 `.py` 时，该文件必须已有以下标记：

```python
# ML_EXPERIMENT_PARAMS_START
ML_EXPERIMENT_PARAMS = {"C": 1.0, "gamma": 1.0}
# ML_EXPERIMENT_PARAMS_END
```

并且训练代码要在没有实验环境变量时使用这个字典，例如：

```python
params = (
    json.loads(os.environ["ML_EXPERIMENT_PARAMS"])
    if "ML_EXPERIMENT_PARAMS" in os.environ
    else ML_EXPERIMENT_PARAMS
)
```

[RBF SVC 示例](../examples/ml/tune_rbf_svc_moons.py) 已准备好这些标记。确认想改它时，在前述 prompt 中追加：

```text
本次搜索结束后，请把最佳参数写回原训练文件：
write_back_path: examples/ml/tune_rbf_svc_moons.py
```

工具只替换两个标记之间的参数字典，不会重写模型训练逻辑。写回发生在实验**结束后**；本次各 trial 仍读取各自的 `ML_EXPERIMENT_PARAMS`。下次直接运行脚本时会使用新字典。若下次再调用调参工具，基线仍由 `baseline_params` 或参数空间里的 `baseline` 决定；均未设置时使用下界或第一个类别值，不会自动读取文件里上次写回的最优值。要以上次最优值作为下次基线，请显式提供 `baseline_params`。

对于 `.ipynb`，工具先转换成脚本运行，但不会把最优参数写回 Notebook 本体。可以把最优参数写入单独的 `.json` 配置文件。

## 会从 `min` 一路增加到 `max` 吗？

**不会。** `low/high`（也兼容 `min/max`）只定义允许的范围。工具先用 `baseline_params`、参数定义中的 `baseline`，或默认下界/首个类别值运行一次基线；之后每个 trial 由 Optuna 的 `TPESampler(seed=seed)` 给出一组参数，并按顺序训练。基线不计入 Optuna 的 `n_trials`，但会参与最终最佳结果比较，所以 `best_trial: 0` 表示基线获胜或与最优 trial 并列。

当前工具没有自定义 `n_startup_trials`。Optuna 默认先随机采样 10 个已完成的 trial，再利用已有指标进行 TPE 采样。因此当 `n_trials=10` 时，搜索部分通常都是初始随机探索；想让 TPE 根据历史结果继续提出配置，可试 `n_trials=20` 或更多。它不是网格搜索，也不保证取到上下界或全局最优。[Optuna TPESampler 文档](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)

`log: true` 表示在对数尺度搜索正数参数，适合学习率、`C`、`gamma`、`weight_decay` 等跨数量级的参数；`int` 的 `step` 限定可取的离散值；`categorical` 只从 `choices` 中选择。固定 `seed` 有助于复现参数建议；模型训练本身还需在脚本中设置相关随机种子。
