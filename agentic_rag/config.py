"""
Configuration management for Agentic RAG system.
Uses Qdrant vector database and OpenRouter for LLM and embeddings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Qdrant settings
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection_name: str = Field(default="documents")
    
    # OpenRouter settings (for both LLM and embeddings)
    openrouter_api_key: str = Field(...)
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_llm_model: str = Field(default="openai/gpt-4-turbo-preview")
    openrouter_embedding_model: str = Field(default="openai/text-embedding-3-small")
    
    # Model settings
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    
    # Agent settings
    max_iterations: int = Field(default=10)
    memory_size: int = Field(default=20)
    
    # Application settings
    log_level: str = Field(default="info")


# Global settings instance
settings = Settings()
