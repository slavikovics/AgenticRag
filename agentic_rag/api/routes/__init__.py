"""
Route modules for Agentic RAG API.
"""

from .documents import router as documents_router
from .files import router as files_router
from .health import router as health_router
from .query import router as query_router
from .search import router as search_router

__all__ = [
    "documents_router",
    "files_router",
    "health_router",
    "query_router",
    "search_router",
]
