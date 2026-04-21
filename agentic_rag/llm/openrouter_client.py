import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterConfig:
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = field(default_factory=lambda: os.getenv("OPENROUTER_LLM_MODEL", ""))
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_cost_tracking: bool = False


class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.total_tokens_used = 0

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": "https://github.com/yourusername/agentic-rag",
            "X-Title": "Agentic RAG System",
            "Content-Type": "application/json",
        }

    async def _make_request(
        self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs
    ) -> Any:
        await self._ensure_session()

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": stream,
        }

        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]

        payload = {k: v for k, v in payload.items() if v is not None}

        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload,
                    headers=self._build_headers(),
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status == 200:
                        if stream:
                            return response
                        else:
                            return await response.json()

                    elif response.status == 429:
                        wait_time = self.config.retry_delay * (2**attempt)
                        logger.warning(f"Rate limited. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue

                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"API error {response.status}: {error_text}")

            except asyncio.TimeoutError as e:
                logger.error(
                    f"Timeout on attempt {attempt + 1}/{self.config.max_retries}"
                )
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2**attempt))

            except aiohttp.ClientError as e:
                logger.error(f"Client error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2**attempt))

    def _track_usage(self, response: Dict[str, Any]):
        if not self.config.enable_cost_tracking:
            return

        if "usage" in response:
            input_tokens = response["usage"].get("prompt_tokens", 0)
            output_tokens = response["usage"].get("completion_tokens", 0)
            self.total_tokens_used += input_tokens + output_tokens

            logger.debug(f"Usage: {input_tokens} in, {output_tokens} out")

    async def complete(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        response = await self._make_request(messages, stream=False, **kwargs)
        self._track_usage(response)

        return response["choices"][0]["message"]["content"]

    async def stream(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> AsyncGenerator[str, None]:
        response = await self._make_request(messages, stream=True, **kwargs)

        async for line in response.content:
            line = line.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue

            if line.startswith("data: "):
                line = line[6:]

            if line == "[DONE]":
                break

            try:
                chunk = json.loads(line)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]
            except json.JSONDecodeError:
                continue

    async def agentic_complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        logger.debug(f"POST {self.config.base_url}/chat/completions")
        logger.debug(f"Model: '{self.config.model}'")
        logger.debug(f"Messages count: {len(messages)}")

        await self._ensure_session()

        payload_kwargs = kwargs.copy()
        if tools:
            payload_kwargs["tools"] = tools

        response = await self._make_request(messages, stream=False, **payload_kwargs)
        self._track_usage(response)

        message = response["choices"][0]["message"]
        content = message.get("content", "")
        tool_calls = message.get("tool_calls")

        return content, tool_calls

    def get_usage_summary(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens_used,
        }

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None


async def get_openrouter_response(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs,
) -> str:
    if model is None:
        model = os.getenv("OPENROUTER_LLM_MODEL")
    if not model:
        raise ValueError(
            "Model must be specified via OPENROUTER_LLM_MODEL env var or passed explicitly"
        )

    config = OpenRouterConfig(model=model, temperature=temperature, **kwargs)

    async with OpenRouterClient(config) as client:
        return await client.complete(messages)
