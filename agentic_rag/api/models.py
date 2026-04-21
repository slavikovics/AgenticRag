from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    include_sources: Optional[bool] = True
    max_iterations: Optional[int] = 10


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: Optional[list[dict]] = None
    iterations: int
    cost_usd: Optional[float] = None


class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    alpha: Optional[float] = 0.5


class SearchResult(BaseModel):
    content: str
    source: str
    chunk_id: int
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    count: int


class FileUploadResponse(BaseModel):
    status: str
    filename: str
    documents_indexed: int
    chunks: int


class FilesUploadResponse(BaseModel):
    status: str
    files_processed: int
    total_documents_indexed: int
    details: list[dict]


class HealthResponse(BaseModel):
    status: str
    qdrant: dict
    llm_client: str = "openrouter"
    version: str = "1.0.0"
