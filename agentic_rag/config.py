import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection_name: str = Field(default="documents")
    
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_llm_model: str = Field(default="openai/gpt-4-turbo-preview")
    openrouter_embedding_model: str = Field(default="openai/text-embedding-3-small")
    
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    
    max_iterations: int = Field(default=10)
    memory_size: int = Field(default=20)
    
    log_level: str = Field(default="info")


settings = Settings()
