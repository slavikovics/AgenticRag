"""
Configuration management for Agentic RAG system.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Weaviate settings
    weaviate_url: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    weaviate_class_name: str = Field(default="Document", env="WEAVIATE_CLASS_NAME")
    
    # OpenRouter settings (for both LLM and embeddings)
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.io/api/v1", env="OPENROUTER_BASE_URL")
    openrouter_llm_model: str = Field(default="openai/gpt-4-turbo-preview", env="OPENROUTER_LLM_MODEL")
    openrouter_embedding_model: str = Field(default="openai/text-embedding-3-small", env="OPENROUTER_EMBEDDING_MODEL")
    
    # Model settings
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    
    # Agent settings
    max_iterations: int = Field(default=10, env="MAX_ITERATIONS")
    memory_size: int = Field(default=20, env="MEMORY_SIZE")
    
    # Application settings
    log_level: str = Field(default="info", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()