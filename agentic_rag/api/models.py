"""
Pydantic models for Agentic RAG API.
"""

from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request body for RAG query."""

    query: str
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    include_sources: Optional[bool] = True
    max_iterations: Optional[int] = 10


class QueryResponse(BaseModel):
    """Response body for RAG query."""

    query: str
    answer: str
    sources: Optional[list[dict]] = None
    iterations: int
    cost_usd: Optional[float] = None


class SearchRequest(BaseModel):
    """Request for knowledge base search."""

    query: str
    limit: Optional[int] = 10
    alpha: Optional[float] = 0.5


class SearchResult(BaseModel):
    """Single search result."""

    content: str
    source: str
    chunk_id: int
    score: float


class SearchResponse(BaseModel):
    """Response for search."""

    query: str
    results: list[SearchResult]
    count: int


class FileUploadResponse(BaseModel):
    """Response for file upload."""

    status: str
    filename: str
    documents_indexed: int
    chunks: int


class FilesUploadResponse(BaseModel):
    """Response for multiple files upload."""

    status: str
    files_processed: int
    total_documents_indexed: int
    details: list[dict]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    qdrant: dict
    llm_client: str = "openrouter"
    version: str = "1.0.0"
