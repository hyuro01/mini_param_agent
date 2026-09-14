# mini_param_agent Production Guide

> Deployment considerations for the current implementation

## Table of Contents

- [1. Features](#1-features)
- [2. Upgrade Directions](#2-upgrade-directions)
- [3. Production Deployment](#3-production-deployment)

---

## 1. Agent Features

mini_param_agent provides a local agent loop and single-script ML experiments. It is not a managed training platform; production use needs additional controls for the training code, dependencies, resources, and data.


### 1.1 Project Structure

```text
mini_param_agent/
├── cli.py                  # Interactive CLI, slash commands and cancellation
├── acp/                    # Agent Client Protocol adapter for editor clients
├── agent.py                # Tool-calling loop, continuation and ML summaries
├── context.py              # Token budget, compression, archive and snapshots
├── config.py               # YAML models and configuration discovery
├── config/                 # Safe templates, system prompt and MCP example
├── llm/                    # Anthropic/OpenAI-compatible provider clients
├── schema/                 # Pydantic message, tool-call and usage models
├── tools/                  # File, shell, notes, Skills, MCP, ML and recall tools
├── skills/                 # Bundled skill instructions and notices
└── logger.py / retry.py    # Run logging and configurable retry policy
tests/                      # Unit, protocol, tool and context regression tests
docs/                       # Production and operational guides
examples/                   # ML training and experiment examples
```

The normal path is `cli.py` or `acp/server.py` → `Agent` → `LLMClient` and registered `Tool` implementations. `context.py` sits immediately before each model request, so the same budget and persistence behavior is shared by interactive CLI and ACP sessions. Configuration templates are packaged, while personal `config.yaml`, logs, experiment outputs and context snapshots remain local and ignored by Git.

### 1.2 What We've Implemented

| Feature                | Demo Implementation                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Context Management** | ✅ Recent complete rounds, bounded summaries, calibrated token estimates, optional resumable snapshots/backups and keyword recall; existing notes remain available |
| **Tool Calling**       | ✅ Read/Write/Edit/Bash, notes, Skills, optional MCP, and ML experiments                                                |
| **Error Handling**     | ✅ Exceptions and configurable LLM retries                                                                             |
| **Logging**            | ✅ Per-run log files and experiment artifacts                                                                          |
| **ML experiment capability**            | ✅ Read parameter range, `run_ml_experiment` runs and trains the py / notebook file, and save best parameters in `.mini_param_agent/experiments/`, or can choose to write directly to the source file                                                                          |


### 1.3 Advanced Context Management

Implemented in `mini_param_agent/context.py`, integrated before model calls in `agent.py`, configured through `config.py`, CLI and ACP. `tools/context_recall.py` provides optional archive retrieval.

#### Behavior

- The token estimator includes serialized messages, roles, tool arguments/results and tool definitions. Known models use their tiktoken encoding; unknown models, including Qwen/Ollama, fall back to `cl100k_base`. A safety margin and upward calibration from API `prompt_tokens` improve budgeting. This is not provider-exact Qwen or multimodal token counting; output-inclusive `total_tokens` is not treated as history size.
- Compression triggers above `token_limit - reserve_tokens`. The system prompt, active user request, and at least the most recent N messages survive. Retention expands to complete user rounds so tool calls and results stay together.
- `summary` uses a bounded summarization request preserving goals, constraints, decisions, parameters, metrics, paths, errors and pending work. Empty/failed summaries fall back to bounded excerpts. `recent` uses excerpts without an extra model call. Both are lossy: use experiment JSON/CSV artifacts for exact results.
- Compacted messages enter a session-local archive. Optional `recall_context(query, limit)` performs keyword matching (character matching for Chinese), not vector search or cross-session retrieval. It returns at most 10 excerpts of 1600 characters each. History is reference data, not new instructions.
- Opt-in snapshots save active messages and the archive at step boundaries and run completion. A crash can lose the unfinished step. Resume does not replay unfinished tool calls. `/clear` starts an isolated session while keeping old files on disk.
- If protected recent history, system instructions or tools still exceed the budget, execution stops with an actionable context error rather than silently dropping the active request. Reduce retention/input/tool output, or increase the budget only when the server supports it.

#### Configuration and smoke test

Add this to the active `config.yaml`; see `config-example.yaml` for all fields:

```yaml
context:
  token_limit: 8192       # Example only: match the server's actual context window
  reserve_tokens: 2048   # Budget headroom; does not change server generation settings
  keep_recent_messages: 4
  strategy: summary      # Or recent: no extra summarization request
  summary_chars: 2000
  summary_input_chars: 12000
  safety_margin: 1.2
  persistence: true
  storage_dir: .mini_param_agent/context
  session_id: demo_context
  resume: false
  enable_recall: true
```

Start `uv run python -m mini_param_agent.cli`; the session ID and snapshot path are displayed. Exchange multiple rounds until `Context compressed: ...` appears. One huge active message is not a good compression test: the active round is protected.

After exiting, set `resume: true` with the same workspace, storage directory and session ID. Resume keeps current system instructions and does not re-execute historical tools. A missing session cannot be resumed. Use a new ID or null for a new session rather than overwriting existing history. ACP newSession always assigns a fresh ID; protocol loadSession remains unsupported.

After compression, ask: “Call recall_context with query report.json to find the earlier experiment report.” Only compacted history is searched; recent messages remain directly visible. Persistence defaults to false; in-memory compression and recall work without persistence but cannot survive restart. Recall defaults to false too.

#### Shared storage and operational limits

An absolute `storage_dir`, such as `/mnt/shared/mini_param_agent/context`, can target a pre-mounted NFS/CephFS volume. Operators must provision the mount and verify POSIX flock and same-directory atomic rename semantics. This code is a shared-filesystem client, not a distributed storage cluster, replication service or HA system. The lock implementation currently targets macOS/Linux.

Snapshots live under `<storage_dir>/<hash-of-absolute-workspace>/<session_id>/`: `context.json`, `context.previous.json`, and a lock file. Cross-host resume requires the same absolute workspace path and shared mount. Writes use a flushed/fsynced temporary file and atomic replacement, retaining one valid previous generation. Corrupt primary snapshots fall back to that generation. Revision checks reject conflicting writers; use one running instance per session.

The previous generation is a same-storage backup, not off-site disaster recovery. Workspace namespacing is not access control. Snapshots and backups contain raw conversations and tool output, potentially including secrets; restrict mount access to trusted users. There is no automatic redaction, encryption or retention cleanup. Archives grow over time; full snapshots and linear keyword scans target small/medium sessions. Do not commit context directories; ignore custom storage paths too.

Future work: native Qwen/server token counting, vector/hybrid retrieval, indexed/sharded archives, retention policies, encryption/access controls, off-site backups and ACP session loading.


## 2. Upgrade Directions

### 2.1 Model Fallback Mechanism

The config selects one model and either an Anthropic or OpenAI client. It does not implement automatic failover across models.

- Introduce a model pool by configuring multiple model accounts to improve availability
- Introduce automatic health checks, failure removal, circuit breaker strategies for the model pool

### 2.2 Model Hallucination Detection and Correction

Tool arguments are validated by individual tools, but the project does not provide a general model-output verification system.

- Perform security checks on input parameters for certain tool calls to prevent high-risk actions
- Perform reflection on results from certain tool calls to check if they are reasonable

## 3. Production Deployment

### 3.1 Container Deployment Recommendations

We recommend using K8s/Docker environments for Agent deployment. Containerized deployment has the following advantages:

- **Resource Isolation**: Each Agent instance runs in an independent container without interference
- **Elastic Scaling**: Automatically adjust instance count based on load
- **Version Management**: Easy rollback and canary releases
- **Environment Consistency**: Development, testing, and production environments are completely consistent

### 3.2 Resource Limit Configuration

#### 3.2.1 CPU and Memory Limits

To prevent the Agent from consuming excessive CPU/Memory resources and affecting the host, CPU and memory limits must be set:

**Docker Configuration Example**:
```yaml
# docker-compose.yml
services:
  agent:
    image: mini_param_agent:latest
    deploy:
      resources:
        limits:
          cpus: '2.0'      # Maximum 2 CPU cores
          memory: 2G       # Maximum 2GB memory
        reservations:
          cpus: '0.5'      # Guarantee at least 0.5 cores
          memory: 512M     # Guarantee at least 512MB
```

#### 3.2.2 Disk Limits

Agents may generate large amounts of temporary files and log files, so disk usage needs to be limited:

**Docker Volume Configuration**:
```yaml
# docker-compose.yml
services:
  agent:
    volumes:
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 1G         # Maximum 1GB for temporary files
      - type: volume
        source: agent-data
        target: /app/data
        volume:
          driver_opts:
            size: 5G       # Maximum 5GB for data volume
```


### 3.3 Linux Account Permission Restrictions

#### 3.3.1 Principle of Least Privilege

**Never run the Agent as root user**, as this poses serious security risks.

**Dockerfile Best Practices**:
```dockerfile
FROM python:3.11-slim

# Install necessary system tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Create non-privileged user
RUN groupadd -r agent && useradd -r -g agent agent

# Set working directory
WORKDIR /app

# Copy this project checkout into the image
COPY --chown=agent:agent . /app

# Switch to non-privileged user before installing dependencies
USER agent

# Sync dependencies using uv
RUN uv sync

# Start the application
CMD ["uv", "run", "mini_param_agent"]
```

#### 3.3.2 File System Permissions

Restrict the Agent to only access necessary directories:

```bash
# Create restricted workspace directory
mkdir -p /app/workspace
chown agent:agent /app/workspace
chmod 750 /app/workspace  # Owner: read/write/execute, Group: read/execute

# Restrict access to sensitive directories
chmod 700 /etc/agent      # Config directory only accessible by owner
chmod 600 /etc/agent/*.yaml  # Config files only readable/writable by owner
```
