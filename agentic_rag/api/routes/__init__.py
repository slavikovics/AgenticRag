"""API route modules."""

from .health import router as health_router
from .query import router as query_router
from .search import router as search_router

__all__ = ["health_router", "query_router", "search_router"]
