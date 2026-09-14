"""Offline context budget, persistence and retrieval regressions."""

import json
import pytest

from mini_param_agent.agent import Agent
from mini_param_agent.config import Config
from mini_param_agent.context import ContextConfig, ContextStore, ConversationContext, TokenCounter
from mini_param_agent.schema import Message, LLMResponse, ToolCall, FunctionCall
from mini_param_agent.tools.context_recall import ContextRecallTool


class Client:
    model = "qwen3.5:4b"

    async def generate(self, messages, tools=None):
        return LLMResponse(content="保留结果：accuracy=0.9; report.json", finish_reason="stop")


def test_snapshot_backup_and_conflict(tmp_path):
    config = ContextConfig(persistence=True, session_id="test")
    store = ContextStore(config, tmp_path)
    first = [Message(role="user", content="first")]
    store.save(first, [])
    competing = ContextStore(config, tmp_path)
    competing.load()
    store.save([Message(role="user", content="second")], first)
    with pytest.raises(RuntimeError, match="changed elsewhere"):
        competing.save(first, [])
    store.path.write_text("broken")
    recovered = ContextStore(config, tmp_path)
    assert recovered.load()[0] == first
    recovered.save(first, [])
    assert json.loads(store.path.read_text())["version"] == 1


def test_workspace_isolation_and_validation(tmp_path):
    config = ContextConfig(persistence=True, storage_dir=str(tmp_path / "shared"), session_id="test")
    assert ContextStore(config, tmp_path / "a").path != ContextStore(config, tmp_path / "b").path
    with pytest.raises(ValueError):
        ContextConfig(session_id="../escape")
    with pytest.raises(ValueError):
        ConversationContext(ContextConfig(resume=True), tmp_path, "unknown")


def test_token_count_tools_and_calibration(tmp_path):
    counter = TokenCounter(ContextConfig(), "qwen3.5:4b")
    messages = [Message(role="user", content="中文 <|endoftext|>")]
    context = ConversationContext(ContextConfig(), tmp_path, "unknown")
    tools = [ContextRecallTool(context)]
    assert counter.count(messages, tools) > counter.count(messages)
    counter.observe(1000, messages, tools)
    assert counter.count(messages, tools) >= 1000


@pytest.mark.parametrize("strategy", ["summary", "recent"])
async def test_compression_and_recall(tmp_path, strategy):
    config = ContextConfig(token_limit=1800, keep_recent_messages=1,
                           summary_chars=300, strategy=strategy, enable_recall=True)
    agent = Agent(Client(), "system", [], workspace_dir=str(tmp_path), context_config=config)
    agent.messages.extend([Message(role="user", content="train report.json"),
                           Message(role="assistant", content="old result " * 4000),
                           Message(role="user", content="latest request")])
    await agent._summarize_messages()
    assert agent.messages[-1].content == "latest request"
    assert agent.messages[0].role == "system"
    assert agent.messages[1].role == "assistant"
    assert agent._estimate_tokens() < config.token_limit
    result = await agent.tools["recall_context"].execute("report.json")
    assert "report.json" in result.content
    assert not (await agent.tools["recall_context"].execute("x", 99)).success


async def test_resume_clear_and_final_answer_preserved(tmp_path):
    config = ContextConfig(persistence=True, session_id="saved", enable_recall=True)
    agent = Agent(Client(), "system", [], workspace_dir=str(tmp_path), context_config=config)
    agent.logger.log_dir = tmp_path / "logs"
    agent.logger.log_dir.mkdir()
    agent.add_user_message("hello")
    answer = await agent.run()
    assert agent.messages[-1].content == answer
    restored = Agent(Client(), "new system", [], workspace_dir=str(tmp_path),
                     context_config=config.model_copy(update={"resume": True}))
    assert restored.messages[-1].content == answer
    assert "new system" in restored.messages[0].content
    restored.clear_history()
    assert len(restored.messages) == 1
    assert restored.context.store.session_id != "saved"


def test_cut_preserves_tool_group(tmp_path):
    context = ConversationContext(ContextConfig(keep_recent_messages=2), tmp_path, "unknown")
    messages = [Message(role="system", content="s"), Message(role="user", content="old"),
                Message(role="assistant", content="done"), Message(role="user", content="new"),
                Message(role="assistant", content="", tool_calls=[ToolCall(id="a", type="function", function=FunctionCall(name="read_file", arguments={}))]),
                Message(role="tool", content="result", tool_call_id="a")]
    assert context.cut(messages) == 3


async def test_summary_failure_bounded_and_active_round_guard(tmp_path):
    class Broken(Client):
        async def generate(self, **kwargs):
            raise RuntimeError("offline")
    agent = Agent(Broken(), "system", [], workspace_dir=str(tmp_path),
                  context_config=ContextConfig(token_limit=500, summary_chars=128))
    assert len(await agent._create_summary([Message(role="user", content="a" * 10000)], 1)) <= 128
    agent.add_user_message("long data " * 3000)
    with pytest.raises(ValueError, match="active round"):
        await agent._summarize_messages()


def test_yaml_settings_and_offline_tokenizer(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("api_key: ollama\ncontext:\n  strategy: recent\n  keep_recent_messages: 3\n")
    assert Config.from_yaml(path).agent.context.keep_recent_messages == 3
    def unavailable(*args):
        raise OSError("offline")
    monkeypatch.setattr("mini_param_agent.context.tiktoken.encoding_for_model", unavailable)
    with pytest.warns(RuntimeWarning, match="Tokenizer unavailable"):
        counter = TokenCounter(ContextConfig(), "unknown")
    assert counter.count([Message(role="user", content="中文")]) > 6


def test_resume_discards_only_unfinished_tool_step(tmp_path):
    config = ContextConfig(persistence=True, session_id="partial")
    store = ContextStore(config, tmp_path)
    user = Message(role="user", content="read a file")
    call = Message(role="assistant", content="", tool_calls=[
        ToolCall(id="a", type="function", function=FunctionCall(name="read_file", arguments={}))])
    store.save([Message(role="system", content="s"), user, call], [])
    agent = Agent(Client(), "system", [], workspace_dir=str(tmp_path),
                  context_config=config.model_copy(update={"resume": True}))
    assert agent.messages[-1] == user
    agent.messages.extend([call, Message(role="tool", content="done", tool_call_id="a")])
    agent._remove_unfinished_tool_step()
    assert agent.messages[-1].role == "tool"


async def test_acp_context_compression_and_checkpoint(tmp_path, capsys):
    from types import SimpleNamespace
    from mini_param_agent.acp import MiniMaxACPAgent, SessionState
    from mini_param_agent.config import AgentConfig, LLMConfig, ToolsConfig

    class Connection:
        async def sessionUpdate(self, payload):
            pass

    settings = ContextConfig(token_limit=1800, keep_recent_messages=1,
                             persistence=True, session_id="acp", summary_chars=300)
    agent = Agent(Client(), "system", [], workspace_dir=str(tmp_path), context_config=settings)
    agent.messages.extend([Message(role="user", content="old request"),
                           Message(role="assistant", content="history " * 5000),
                           Message(role="user", content="new request")])
    config = Config(llm=LLMConfig(api_key="test"), agent=AgentConfig(), tools=ToolsConfig())
    adapter = MiniMaxACPAgent(Connection(), config, Client(), [], "system")
    assert await adapter._run_turn(SessionState(agent=agent), "acp") == "end_turn"
    assert agent.context.archive
    assert agent.context.store.path.exists()
    assert "Context compressed" not in capsys.readouterr().out
