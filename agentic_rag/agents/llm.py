"""
llm.py — LiteLLM client wrapper.

Provides a single async method for tool-enabled completions.
Handles retries via tenacity and surfaces a clean interface
so the agent doesn't depend on litellm directly.

Supports any provider litellm supports:
    "openai/gpt-4o"
    "anthropic/claude-sonnet-4-5"
    "openai/Qwen3-8B"           ← OpenAI-compatible local/cloud endpoint
    "ollama/llama3"             ← local Ollama
"""

import logging
from typing import Any, Optional

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


class LLMClient:
    """
    Thin async wrapper around litellm.acompletion.

    Usage:
        client = LLMClient(
            model="openai/gpt-4o",
            api_key="sk-...",             # optional, falls back to env var
            api_base="http://localhost:11434",  # optional, for local endpoints
        )
        content, tool_calls = await client.agentic_complete(messages, tools)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        retry_attempts: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Build retry decorator with caller-supplied config
        self._retry = retry(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(min=retry_min_wait, max=retry_max_wait),
            retry=retry_if_exception_type(Exception),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True,
        )

    async def agentic_complete(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, Optional[list[dict[str, Any]]]]:
        """
        Run one LLM completion turn.
        Returns (content, tool_calls) where tool_calls is None if no tools were called.
        """
        return await self._retry(self._complete)(messages, tools)

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, Optional[list[dict[str, Any]]]]:
        # Import here so the module is importable even if litellm isn't installed
        try:
            from litellm import acompletion
        except ImportError:
            raise ImportError("litellm is required: pip install litellm")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = await acompletion(**kwargs)
        message = response.choices[0].message
        content = message.content or ""

        raw_tool_calls = getattr(message, "tool_calls", None)
        if not raw_tool_calls:
            return content, None

        # Normalise to plain dicts so the agent doesn't depend on litellm types
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in raw_tool_calls
        ]
        return content, tool_calls
