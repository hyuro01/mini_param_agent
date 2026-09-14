"""Regression tests for OpenAI-compatible local model request formatting."""

from mini_param_agent.llm.openai_client import OpenAIClient
from mini_param_agent.schema import FunctionCall, Message, ToolCall


def _client(is_minimax_endpoint: bool) -> OpenAIClient:
    """Build only the conversion portion; no SDK client or network is needed."""
    client = OpenAIClient.__new__(OpenAIClient)
    client.is_minimax_endpoint = is_minimax_endpoint
    client.is_ollama_endpoint = not is_minimax_endpoint
    return client


def test_ollama_empty_assistant_content_is_not_omitted():
    """Ollama rejects assistant messages whose content is serialized as null."""
    messages = [
        Message(role="user", content="hello"),
        Message(
            role="assistant",
            content="",
            thinking="internal reasoning from a prior turn",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="example", arguments={}),
                )
            ],
        ),
    ]

    _, converted = _client(False)._convert_messages(messages)

    assert converted[1]["content"] == ""
    assert "reasoning_details" not in converted[1]
    assert converted[1]["tool_calls"][0]["id"] == "call_1"


def test_minimax_keeps_its_reasoning_extension():
    _, converted = _client(True)._convert_messages(
        [Message(role="assistant", content="", thinking="reasoning")]
    )

    assert converted[0]["content"] == ""
    assert converted[0]["reasoning_details"] == [{"text": "reasoning"}]


def test_explicit_ollama_ml_tool_request_forces_that_tool():
    choice = _client(False)._get_explicit_ollama_tool_choice(
        [{"role": "user", "content": "Please use run_ml_experiment now."}],
        [{"type": "function", "function": {"name": "run_ml_experiment"}}],
    )

    assert choice == "run_ml_experiment"


def test_ollama_does_not_repeat_forced_tool_after_tool_result():
    choice = _client(False)._get_explicit_ollama_tool_choice(
        [
            {"role": "user", "content": "Please use run_ml_experiment now."},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "completed"},
        ],
        [{"type": "function", "function": {"name": "run_ml_experiment"}}],
    )

    assert choice is None
