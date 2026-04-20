"""
Async OpenRouter LLM client with streaming, retries, and cost tracking.
"""

import os
import json
import time
import asyncio
import aiohttp
import logging
from typing import Optional, AsyncGenerator, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Available model families via OpenRouter."""
    OPENAI_TURBO = "openai/gpt-3.5-turbo"
    OPENAI_4 = "openai/gpt-4"
    OPENAI_4_TURBO = "openai/gpt-4-turbo-preview"
    OPENAI_4O = "openai/gpt-4o"
    OPENAI_4O_MINI = "openai/gpt-4o-mini"
    
    ANTHROPIC_OPUS = "anthropic/claude-3-opus"
    ANTHROPIC_SONNET = "anthropic/claude-3.5-sonnet"
    ANTHROPIC_HAIKU = "anthropic/claude-3-haiku"
    
    LLAMA_70B = "meta-llama/llama-3.1-70b-instruct"
    LLAMA_8B = "meta-llama/llama-3.1-8b-instruct"
    MISTRAL_LARGE = "mistralai/mistral-large"
    MIXTRAL = "mistralai/mixtral-8x7b-instruct"
    
    # Budget-friendly
    NEURAL_CHAT = "teknium/openhermes-2.5-mistral-7b"
    OPENCHAT = "openchat/openchat-7b"


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter client."""
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    base_url: str = "https://openrouter.io/api/v1"
    model: ModelProvider = ModelProvider.OPENAI_4_TURBO
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_cost_tracking: bool = True


class OpenRouterClient:
    """Async client for OpenRouter API with streaming and retry logic."""
    
    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self._token_prices = self._get_token_prices()
    
    @staticmethod
    def _get_token_prices() -> Dict[str, Tuple[float, float]]:
        """
        Returns (input_price, output_price) per 1K tokens.
        Prices in USD.
        """
        return {
            ModelProvider.OPENAI_4_TURBO.value: (0.01, 0.03),
            ModelProvider.OPENAI_4.value: (0.03, 0.06),
            ModelProvider.OPENAI_4O.value: (0.005, 0.015),
            ModelProvider.OPENAI_4O_MINI.value: (0.00015, 0.0006),
            ModelProvider.OPENAI_TURBO.value: (0.0005, 0.0015),
            ModelProvider.ANTHROPIC_OPUS.value: (0.015, 0.075),
            ModelProvider.ANTHROPIC_SONNET.value: (0.003, 0.015),
            ModelProvider.ANTHROPIC_HAIKU.value: (0.00025, 0.00125),
            ModelProvider.LLAMA_70B.value: (0.00012, 0.0003),
            ModelProvider.LLAMA_8B.value: (0.00002, 0.00005),
            ModelProvider.MISTRAL_LARGE.value: (0.0024, 0.0072),
            ModelProvider.MIXTRAL.value: (0.00024, 0.00024),
            ModelProvider.NEURAL_CHAT.value: (0.0, 0.0),  # Free
            ModelProvider.OPENCHAT.value: (0.0, 0.0),  # Free
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure session is initialized."""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": "https://github.com/yourusername/agentic-rag",
            "X-Title": "Agentic RAG System",
            "Content-Type": "application/json",
        }
    
    async def _make_request(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> Any:
        """Make request with retry logic."""
        
        await self._ensure_session()
        
        payload = {
            "model": self.config.model.value if isinstance(self.config.model, ModelProvider) else self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": stream,
        }
        
        # Add tools if provided
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        
        # Remove None values
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
                    
                    elif response.status == 429:  # Rate limit
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"API error {response.status}: {error_text}")
            
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout on attempt {attempt + 1}/{self.config.max_retries}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            
            except aiohttp.ClientError as e:
                logger.error(f"Client error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
    
    def _track_usage(self, response: Dict[str, Any]):
        """Track token usage and cost."""
        if not self.config.enable_cost_tracking:
            return
        
        if "usage" in response:
            input_tokens = response["usage"].get("prompt_tokens", 0)
            output_tokens = response["usage"].get("completion_tokens", 0)
            self.total_tokens_used += input_tokens + output_tokens
            
            # Calculate cost
            model = self.config.model.value if isinstance(self.config.model, ModelProvider) else self.config.model
            prices = self._token_prices.get(model, (0.0, 0.0))
            cost = (input_tokens * prices[0] + output_tokens * prices[1]) / 1000
            self.total_cost += cost
            
            logger.debug(
                f"Usage: {input_tokens} in, {output_tokens} out. "
                f"Cost: ${cost:.6f} (Total: ${self.total_cost:.6f})"
            )
    
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """
        Non-streaming completion.
        
        Args:
            messages: Chat messages in OpenAI format
            **kwargs: Override temperature, max_tokens, etc.
        
        Returns:
            Generated text
        """
        response = await self._make_request(messages, stream=False, **kwargs)
        self._track_usage(response)
        
        return response["choices"][0]["message"]["content"]
    
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Streaming completion.
        
        Yields:
            Text chunks as they arrive
        """
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
        **kwargs
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Completion with tool use support (for ReAct agents).
        
        Args:
            messages: Chat messages
            tools: Tool definitions (optional)
            **kwargs: Override params
        
        Returns:
            (response_text, tool_calls) where tool_calls is None if no tools
        """
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
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return {
            "total_tokens": self.total_tokens_used,
            "total_cost_usd": self.total_cost,
            "avg_cost_per_1k_tokens": (
                self.total_cost / (self.total_tokens_used / 1000)
                if self.total_tokens_used > 0
                else 0
            ),
        }
    
    async def close(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None


# Convenience function
async def get_openrouter_response(
    messages: List[Dict[str, Any]],
    model: ModelProvider = ModelProvider.OPENAI_4_TURBO,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """One-shot completion with OpenRouter."""
    config = OpenRouterConfig(model=model, temperature=temperature, **kwargs)
    async with OpenRouterClient(config) as client:
        return await client.complete(messages)