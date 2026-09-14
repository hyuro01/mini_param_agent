# mini_param_agent 生产环境指南

> 当前实现的部署注意事项

## 目录

- [1. 功能实现](#1-功能实现)
- [2. 可升级方向](#2-可升级方向)
- [3. 生产部署](#3-生产部署)

---

## 1. Agent 功能概述

mini_param_agent 提供本地 Agent 循环和单脚本机器学习实验，并非托管训练平台。生产使用还需额外控制训练代码、依赖、资源和数据。


### 1.1 项目结构

```text
mini_param_agent/
├── cli.py                  # 交互式 CLI、斜杠命令与取消处理
├── acp/                    # 面向编辑器客户端的 Agent Client Protocol 适配层
├── agent.py                # 工具调用循环、续答与 ML 结果摘要
├── context.py              # Token 预算、压缩、归档与会话快照
├── config.py               # YAML 配置模型与配置文件搜索
├── config/                 # 安全模板、系统提示词与 MCP 示例
├── llm/                    # Anthropic/OpenAI 兼容模型客户端
├── schema/                 # Pydantic 消息、工具调用与用量模型
├── tools/                  # 文件、Shell、笔记、Skills、MCP、ML 与召回工具
├── skills/                 # 随包提供的 Skill 说明与第三方声明
└── logger.py / retry.py    # 运行日志与可配置重试策略
tests/                      # 单元、协议、工具与上下文回归测试
docs/                       # 生产与运维指南
examples/                   # ML 训练与实验示例
```

典型调用链是 `cli.py` 或 `acp/server.py` → `Agent` → `LLMClient` 与已注册的 `Tool`。`context.py` 位于每次模型请求之前，因此交互式 CLI 和 ACP 会话共享相同的预算、压缩与持久化行为。配置模板会随包发布；个人 `config.yaml`、日志、实验产物和上下文快照保留在本地并被 Git 忽略。

### 1.2 Agent 实现的功能

| 功能           | Agent 实现                                                                                                   |
| -------------- | ----------------------------------------------------------------------------------------------------------- |
| **上下文管理** | ✅ 最近完整轮次保留、有界摘要、Token 预算与校准、可选会话快照/恢复/备份和关键词召回；原有笔记工具仍可用。 |
| **工具调用**   | ✅ Read/Write/Edit/Bash、笔记、Skills、可选 MCP 与机器学习实验。                                              |
| **错误处理**   | ✅ 异常处理与可配置的 LLM 重试。                                                                             |
| **日志**       | ✅ 每次运行的日志文件与实验产物。                                                                           |
| **ML 调参**       | ✅ 读取参数范围，`run_ml_experiment`对 py / notebook 文件进行运行和训练，将输出最佳结果的参数保存在`.mini_param_agent/experiments/`，同时可选择直接写入源文件。                                                                           |

### 1.3 高级上下文管理

以下能力已实现。核心位于 `mini_param_agent/context.py`，由 `agent.py` 在每次模型调用前检查预算；`tools/context_recall.py` 提供召回工具，`config.py`、CLI 和 ACP 负责配置接入。

#### 如何起作用

1. 对消息的 JSON 内容、角色、工具参数/结果和工具定义进行 Token 估算。已知模型使用 tiktoken 对应编码；未知模型（包括当前 Qwen/Ollama）回退到 `cl100k_base`，附加安全余量，并用服务端 `prompt_tokens` 向上校准。不使用包含输出的 `total_tokens` 作为历史大小。它不是 Qwen 原生或多模态精确计数器。
2. 超过 `token_limit - reserve_tokens` 时，保留系统提示、当前用户请求及至少最近 N 条消息，按完整用户轮次划分，因此实际保留数可能大于 N；工具调用和结果不会被拆开。
3. `summary` 策略请求模型保留目标、约束、决策、指标、参数、路径、错误和待办；输入和输出均有长度限制。摘要失败或返回空内容时使用有界原文摘录。`recent` 策略不额外调用模型，只保留最近轮次及较早内容的有界摘录。摘要和摘录都是有损的，不保证每个参数都保留；精确指标以实验 JSON/CSV 为准。
4. 被压缩消息保存在本会话 archive 中。开启 `enable_recall` 后，模型可调用 `recall_context(query, limit)` 搜索这部分历史。当前是关键词匹配（中文按字符），不是向量检索，不跨会话搜索；返回最多 10 条，每条最多 1600 字符。历史结果仅是参考数据，不是新的指令。
5. 开启持久化后，在运行步骤边界和结束时保存有效消息及 archive。崩溃可能丢失尚未完成的步骤；恢复不会自动重放未完成的工具调用。`/clear` 创建独立的新会话，旧文件仍留在磁盘。

若系统提示、工具定义或受保护的当前轮次本身已经超预算，会明确停止并提示调整预算/输入，而不是静默删除当前请求。可降低 `keep_recent_messages`、缩小工具输出，或在服务端确实支持时增大窗口。

#### 配置与测试方法

将以下内容加入实际使用的 `config.yaml`（完整字段见 `config-example.yaml`）：

```yaml
context:
  token_limit: 8192       # 示例，不代表本地模型实际配置；应与服务端窗口匹配
  reserve_tokens: 2048   # 留给输出；不会替你修改服务端生成长度
  keep_recent_messages: 4
  strategy: summary     # 或 recent，避免额外摘要模型调用
  summary_chars: 2000
  summary_input_chars: 12000
  safety_margin: 1.2
  persistence: true
  storage_dir: .mini_param_agent/context
  session_id: demo_context
  resume: false
  enable_recall: true
```

首次运行 `uv run python -m mini_param_agent.cli`，启动时会显示会话 ID 和快照路径。连续交谈多个轮次，达到预算时会显示 `Context compressed: ...`。不要用一条超长消息测试“旧历史压缩”，因为当前轮次受到保护。

退出后将 `resume` 改成 `true`，保持同一 workspace、storage_dir 和 session_id，再启动即可恢复；恢复使用当前系统提示，不重新执行历史工具。首次会话不存在时不能设置 `resume: true`；新会话请换 session_id 或设为 null，避免覆盖已有会话。ACP 的 newSession 总是分配独立 ID，不通过此开关恢复已有 ACP 会话，协议 loadSession 尚未实现。

压缩发生后可问：“请调用 recall_context，query 为 report.json，找出之前的实验报告记录。”未被压缩的消息还在直接上下文中，不在 archive 搜索结果中。关闭持久化仍可做本次进程内压缩和召回，但退出后无法恢复。

#### 共享文件系统、备份与边界

`storage_dir` 可改成绝对路径，例如 `/mnt/shared/mini_param_agent/context`。需要由部署者先挂载 NFS/CephFS 等，并验证挂载支持 POSIX `flock` 与同目录原子 rename。代码只实现共享文件系统客户端存储，不创建分布式集群、复制或高可用服务；当前锁实现适用于 macOS/Linux。

目录结构为 `<storage_dir>/<workspace绝对路径哈希>/<session_id>/`，包含 `context.json`、`context.previous.json` 和锁文件。跨机器恢复需要同一 workspace 绝对路径和共享挂载。临时文件写入并 fsync 后原子替换，保留上一有效版本；主快照损坏时回退上一版。版本校验拒绝另一写入者覆盖已更新的会话；同一会话应只有一个运行实例。

备份是同一存储上的上一代副本，不是异地灾备。目录不是权限隔离系统：只共享给可信用户，限制挂载权限，备份文件也含完整历史、工具输出甚至敏感数据；没有自动脱敏、加密或过期清理。archive 随会话增长，当前快照与关键词扫描适合中小规模会话。不要把会话目录提交到 Git；自定义存储路径也需要加入忽略规则。

后续可升级：Qwen 原生 tokenizer / 服务端计数接口、向量或混合召回、索引与归档分片、保留周期、加密和访问控制，以及真正的异地备份与 ACP 会话恢复。


## 2. 升级与拓展方向


### 2.1 模型回退机制

配置可选择一个模型及 Anthropic/OpenAI 客户端；当前没有跨模型自动回退机制。

- **建立模型池**：配置多个模型账号，建立模型池以提高服务可用性。
- **引入高可用策略**：为模型池引入自动健康检测、故障节点切换、熔断等高可用策略。

### 2.2 模型幻觉的检测与修正

各工具会分别校验调用参数，但项目尚无通用的模型输出验证机制。

- **输入参数安全检查**：对部分工具的调用参数进行安全性检查，防止执行高危操作。
- **输出结果合理性检查**：对部分工具的调用结果进行反思（Self-reflection），检查其合理性。

## 3. 生产环境部署

### 3.1 容器化部署建议

我们推荐使用 Kubernetes 或 Docker 环境来部署 Agent。容器化部署具有以下优势：

- **资源隔离**：每个 Agent 实例运行在独立的容器中，互不干扰。
- **弹性扩展**：根据负载自动调整实例数量。
- **版本管理**：便于快速回滚和灰度发布。
- **环境一致性**：开发、测试、生产环境完全一致。

### 3.2 资源限制

#### 3.2.1 CPU 与内存限制

为防止 Agent 实例占用过多资源而影响宿主机，您必须为其设置 CPU 和内存的限制：

**Docker 配置示例**：
```yaml
# docker-compose.yml
services:
  agent:
    image: mini_param_agent:latest
    deploy:
      resources:
        limits:
          cpus: '2.0'      # 最多使用 2 个 CPU 核心
          memory: 2G       # 最多使用 2GB 内存
        reservations:
          cpus: '0.5'      # 保证至少 0.5 个核心
          memory: 512M     # 保证至少 512MB
```

#### 3.2.2 磁盘限制

Agent 运行过程中可能会产生大量的临时文件和日志，因此需要限制其磁盘使用量：

**Docker Volume 配置**：
```yaml
# docker-compose.yml
services:
  agent:
    volumes:
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 1G         # 临时文件最多 1GB
      - type: volume
        source: agent-data
        target: /app/data
        volume:
          driver_opts:
            size: 5G       # 数据卷最多 5GB
```


### 3.3 Linux 账户权限限制

#### 3.3.1 最小权限原则

**请勿使用 root 用户运行 Agent**，这会带来严重的安全风险。

**Dockerfile 最佳实践**：
```dockerfile
FROM python:3.11-slim

# 安装必要的系统工具
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# 创建非特权用户
RUN groupadd -r agent && useradd -r -g agent agent

# 设置工作目录
WORKDIR /app

# 将当前项目检出目录复制进镜像
COPY --chown=agent:agent . /app

# 切换到非特权用户后安装依赖
USER agent

# 使用 uv 同步依赖
RUN uv sync

# 启动应用
CMD ["uv", "run", "mini_param_agent"]
```

#### 3.3.2 文件系统权限

您应限制 Agent 只能访问必要的目录：

```bash
# 创建受限的工作目录
mkdir -p /app/workspace
chown agent:agent /app/workspace
chmod 750 /app/workspace  # 所有者读写执行，组只读执行

# 限制敏感目录的访问
chmod 700 /etc/agent      # 配置目录只有所有者能访问
chmod 600 /etc/agent/*.yaml  # 配置文件只有所有者能读写
```
