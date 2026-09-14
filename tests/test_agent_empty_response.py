"""Regression coverage for local models that emit only reasoning."""

from mini_param_agent.agent import Agent
from mini_param_agent.schema import LLMResponse, Message
from mini_param_agent.tools.base import ToolResult


class _ReasoningOnlyThenAnswerClient:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages: list[Message], tools=None) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                thinking="I should formulate an answer.",
                finish_reason="stop",
            )
        return LLMResponse(content="Hello!", finish_reason="stop")


async def test_agent_continues_after_reasoning_only_response(tmp_path):
    client = _ReasoningOnlyThenAnswerClient()
    agent = Agent(
        llm_client=client,
        system_prompt="You are helpful.",
        tools=[],
        max_steps=3,
        workspace_dir=str(tmp_path),
    )
    agent.logger.log_dir = tmp_path / "logs"
    agent.logger.log_dir.mkdir()
    agent.add_user_message("hello")

    result = await agent.run()

    assert result == "Hello!"
    assert client.calls == 2
    assert "previous response contained analysis" in agent.messages[-2].content
    assert "'hello'" in agent.messages[-2].content


class _GetToolThenAnswerClient:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages: list[Message], tools=None) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            from mini_param_agent.schema import FunctionCall, ToolCall

            return LLMResponse(
                content="I will inspect the tool.",
                tool_calls=[
                    ToolCall(id="call_1", type="function", function=FunctionCall(name="get_tool", arguments={}))
                ],
                finish_reason="tool_calls",
            )
        return LLMResponse(content="Recovered.", finish_reason="stop")


async def test_agent_recovers_from_local_model_get_tool_hallucination(tmp_path):
    agent = Agent(
        llm_client=_GetToolThenAnswerClient(),
        system_prompt="You are helpful.",
        tools=[type("ExperimentTool", (), {"name": "run_ml_experiment"})()],
        max_steps=3,
        workspace_dir=str(tmp_path),
    )
    agent.logger.log_dir = tmp_path / "logs"
    agent.logger.log_dir.mkdir()
    agent.add_user_message("Tune a model")

    result = await agent.run()

    assert result == "Recovered."
    assert "There is no get_tool" in agent.messages[-2].content


def test_explicit_ml_request_narrows_available_tools(tmp_path):
    ml_tool = type("ExperimentTool", (), {"name": "run_ml_experiment"})()
    other_tool = type("OtherTool", (), {"name": "bash"})()
    agent = Agent(
        llm_client=_ReasoningOnlyThenAnswerClient(),
        system_prompt="You are helpful.",
        tools=[ml_tool, other_tool],
        workspace_dir=str(tmp_path),
    )
    agent.add_user_message("Please call run_ml_experiment for this file.")

    assert agent._select_tools_for_next_call() == [ml_tool]


class _SuccessfulMLTool:
    name = "run_ml_experiment"

    async def execute(self, **kwargs):
        return ToolResult(
            success=True,
            content=(
                '{"best_trial": 2, "best_score": 0.9, "best_params": {"lr": 0.001}, '
                '"report": "/tmp/best_params.json", "results_csv": "/tmp/results.csv"}'
            ),
        )


class _MLToolCallingClient:
    async def generate(self, messages: list[Message], tools=None) -> LLMResponse:
        from mini_param_agent.schema import FunctionCall, ToolCall

        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_ml",
                    type="function",
                    function=FunctionCall(name="run_ml_experiment", arguments={}),
                )
            ],
            finish_reason="tool_calls",
        )


async def test_agent_returns_ml_tool_summary_without_a_second_model_call(tmp_path):
    agent = Agent(
        llm_client=_MLToolCallingClient(),
        system_prompt="You are helpful.",
        tools=[_SuccessfulMLTool()],
        max_steps=3,
        workspace_dir=str(tmp_path),
    )
    agent.logger.log_dir = tmp_path / "logs"
    agent.logger.log_dir.mkdir()
    agent.add_user_message("Call run_ml_experiment")

    result = await agent.run()

    assert "最佳分数：0.9" in result
    assert "best_params.json" in result
