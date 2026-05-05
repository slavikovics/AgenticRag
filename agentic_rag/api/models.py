"""
models.py — Pydantic request/response models.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    max_iterations: Optional[int] = None  # overrides agent default if set
    collection_hint: Optional[str] = None  # optional: hint the agent which collection


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict[str, Any]] = []
    iterations: int = 0


class SearchRequest(BaseModel):
    query: str
    collection: str  # required: which university collection
    limit: int = Field(default=5, ge=1, le=50)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    content: str
    source: str
    title: str = ""
    score: float


class SearchResponse(BaseModel):
    query: str
    collection: str
    results: list[SearchResult]
    count: int


class CollectionInfo(BaseModel):
    name: str
    points_count: int
    status: str
    vector_size: int


class HealthResponse(BaseModel):
    status: str
    collections: list[CollectionInfo] = []
    llm_model: str
    web_search_enabled: bool
    version: str = "2.0.0"
