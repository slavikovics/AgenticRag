"""
config.py — application settings loaded from environment / .env file.
"""

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_grpc_port: int = Field(default=6334)

    # OpenRouter — used for both LLM and embeddings
    openrouter_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")

    # Embedding model (Qwen3-Embedding via OpenRouter)
    embedding_model: str = Field(default="qwen/qwen3-embedding-8b")

    # LLM (litellm model string, routed through OpenRouter)
    llm_model: str = Field(default="openrouter/qwen/qwen3.5-9b")

    # Tavily
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    # Agent
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=2048)
    max_iterations: int = Field(default=10)
    memory_size: int = Field(default=20)
    agent_timeout: float = Field(default=60.0)

    # General collection name (aggregated stats)
    general_collection: str = Field(default="general")

    log_level: str = Field(default="info")


settings = Settings()
