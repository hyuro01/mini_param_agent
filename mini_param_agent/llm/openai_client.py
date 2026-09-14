"""OpenAI LLM client implementation."""

import json
import logging
from typing import Any
from urllib.parse import urlparse

from openai import AsyncOpenAI

from ..retry import RetryConfig, async_retry
from ..schema import FunctionCall, LLMResponse, Message, TokenUsage, ToolCall
from .base import LLMClientBase

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClientBase):
    """LLM client using OpenAI's protocol.

    This client uses the official OpenAI SDK and supports:
    - Reasoning content (via reasoning_split=True)
    - Tool calling
    - Retry logic
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/v1",
        model: str = "MiniMax-M2.5",
        retry_config: RetryConfig | None = None,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: API key for authentication
            api_base: Base URL for the API (default: MiniMax OpenAI endpoint)
            model: Model name to use (default: MiniMax-M2.5)
            retry_config: Optional retry configuration
        """
        super().__init__(api_key, api_base, model, retry_config)
        # ``reasoning_split`` and ``reasoning_details`` are MiniMax extensions.
        # Passing them to a generic OpenAI-compatible server (notably Ollama)
        # is both unnecessary and can produce incompatible message handling.
        self.is_minimax_endpoint = any(
            host in api_base for host in ("api.minimax.io", "api.minimaxi.com")
        )
        parsed_base_url = urlparse(api_base)
        self.is_ollama_endpoint = parsed_base_url.port == 11434 or parsed_base_url.hostname in {
            "localhost",
            "127.0.0.1",
            "::1",
        }

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    async def _make_api_request(
        self,
        api_messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> Any:
        """Execute API request (core method that can be retried).

        Args:
            api_messages: List of messages in OpenAI format
            tools: Optional list of tools

        Returns:
            OpenAI ChatCompletion response (full response including usage)

        Raises:
            Exception: API call failed
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }

        # Enable MiniMax's non-standard reasoning support only for MiniMax.
        if self.is_minimax_endpoint:
            params["extra_body"] = {"reasoning_split": True}
        elif self.is_ollama_endpoint:
            # Qwen reasoning models served by Ollama can spend their complete
            # response budget in the reasoning field when presented with a large
            # tool schema. Ask for direct answers; the Agent has a continuation
            # fallback below for servers that still emit reasoning.
            params["extra_body"] = {"think": False}

        if tools:
            converted_tools = self._convert_tools(tools)
            params["tools"] = converted_tools
            forced_tool = self._get_explicit_ollama_tool_choice(api_messages, converted_tools)
            if forced_tool:
                params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": forced_tool},
                }

        # Use OpenAI SDK's chat.completions.create
        response = await self.client.chat.completions.create(**params)
        # Return full response to access usage info
        return response

    def _get_explicit_ollama_tool_choice(
        self,
        api_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> str | None:
        """Force a tool only when the user explicitly names it.

        Qwen-sized local models can state that they will invoke a function but
        finish with ordinary text instead. Ollama supports OpenAI's
        ``tool_choice`` object, so honor an explicit tool-name request without
        changing normal autonomous tool selection.
        """
        if not self.is_ollama_endpoint:
            return None
        # A tool result means the explicitly requested operation has already
        # begun; do not force it again on the summarizing follow-up request.
        if not api_messages or api_messages[-1].get("role") != "user":
            return None
        last_user_content = api_messages[-1].get("content", "")
        if not isinstance(last_user_content, str):
            return None
        available_names = {
            item.get("function", {}).get("name")
            for item in tools
            if item.get("type") == "function"
        }
        # Keep this deliberately narrow: it is a user-provided explicit action,
        # not a keyword heuristic for all ML-related questions.
        if "run_ml_experiment" in last_user_content and "run_ml_experiment" in available_names:
            return "run_ml_experiment"
        return None

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to OpenAI format.

        Args:
            tools: List of Tool objects or dicts

        Returns:
            List of tools in OpenAI dict format
        """
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                # If already a dict, check if it's in OpenAI format
                if "type" in tool and tool["type"] == "function":
                    result.append(tool)
                else:
                    # Assume it's in Anthropic format, convert to OpenAI
                    result.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "description": tool["description"],
                                "parameters": tool["input_schema"],
                            },
                        }
                    )
            elif hasattr(tool, "to_openai_schema"):
                # Tool object with to_openai_schema method
                result.append(tool.to_openai_schema())
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")
        return result

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to OpenAI format.

        Args:
            messages: List of internal Message objects

        Returns:
            Tuple of (system_message, api_messages)
            Note: OpenAI includes system message in the messages array
        """
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                # OpenAI includes system message in messages array
                api_messages.append({"role": "system", "content": msg.content})
                continue

            # For user messages
            if msg.role == "user":
                api_messages.append({"role": "user", "content": msg.content})

            # For assistant messages
            elif msg.role == "assistant":
                # OpenAI allows null assistant content for tool calls, but Ollama
                # rejects it with "invalid message content type: <nil>". Always
                # send a concrete string for an empty assistant turn.
                if isinstance(msg.content, str):
                    assistant_content: str | list[dict[str, Any]] = msg.content
                else:
                    assistant_content = msg.content
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_content}

                # Add tool calls if present
                if msg.tool_calls:
                    tool_calls_list = []
                    for tool_call in msg.tool_calls:
                        tool_calls_list.append(
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": json.dumps(tool_call.function.arguments),
                                },
                            }
                        )
                    assistant_msg["tool_calls"] = tool_calls_list

                # IMPORTANT: Add reasoning_details if thinking is present
                # This is CRITICAL for Interleaved Thinking to work properly!
                # The complete response_message (including reasoning_details) must be
                # preserved in Message History and passed back to the model in the next turn.
                # This ensures the model's chain of thought is not interrupted.
                if msg.thinking and self.is_minimax_endpoint:
                    assistant_msg["reasoning_details"] = [{"text": msg.thinking}]

                api_messages.append(assistant_msg)

            # For tool result messages
            elif msg.role == "tool":
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )

        return None, api_messages

    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare the request for OpenAI API.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            Dictionary containing request parameters
        """
        _, api_messages = self._convert_messages(messages)

        return {
            "api_messages": api_messages,
            "tools": tools,
        }

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI response into LLMResponse.

        Args:
            response: OpenAI ChatCompletion response (full response object)

        Returns:
            LLMResponse object
        """
        # Get message from response
        message = response.choices[0].message

        # Extract text content
        text_content = message.content or ""

        # Extract thinking content from MiniMax's reasoning_details and common
        # OpenAI-compatible fields used by local reasoning models such as Qwen.
        thinking_content = ""
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            # reasoning_details is a list of reasoning blocks
            for detail in message.reasoning_details:
                if hasattr(detail, "text"):
                    thinking_content += detail.text
        if not thinking_content:
            thinking_content = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
            )

        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                # Parse arguments from JSON string
                arguments = json.loads(tool_call.function.arguments)

                tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        type="function",
                        function=FunctionCall(
                            name=tool_call.function.name,
                            arguments=arguments,
                        ),
                    )
                )

        # Extract token usage from response
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return LLMResponse(
            content=text_content,
            thinking=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason="stop",  # OpenAI doesn't provide finish_reason in the message
            usage=usage,
        )

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        """Generate response from OpenAI LLM.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            LLMResponse containing the generated content
        """
        # Prepare request
        request_params = self._prepare_request(messages, tools)

        # Make API request with retry logic
        if self.retry_config.enabled:
            # Apply retry logic
            retry_decorator = async_retry(config=self.retry_config, on_retry=self.retry_callback)
            api_call = retry_decorator(self._make_api_request)
            response = await api_call(
                request_params["api_messages"],
                request_params["tools"],
            )
        else:
            # Don't use retry
            response = await self._make_api_request(
                request_params["api_messages"],
                request_params["tools"],
            )

        # Parse and return response
        return self._parse_response(response)
