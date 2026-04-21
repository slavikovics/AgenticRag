import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_rag.config import settings

from .dependencies import cleanup_resources, get_llm_client, get_qdrant_manager
from .routes import (
    documents_router,
    files_router,
    health_router,
    query_router,
    search_router,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Agentic RAG API...")
    try:
        llm = await get_llm_client()
        qdrant = await get_qdrant_manager()
        logger.info("Initialization complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    logger.info("Shutting down...")
    await cleanup_resources()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Agentic RAG API",
    description="RAG system with agents using Qdrant and OpenRouter",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(search_router, prefix="/api/v1", tags=["Search"])
app.include_router(query_router, prefix="/api/v1", tags=["Query"])
app.include_router(documents_router, prefix="/api/v1", tags=["Documents"])
app.include_router(files_router, prefix="/api/v1", tags=["Files"])


@app.get("/")
async def root():
    """API root."""
    return {
        "name": "Agentic RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /api/v1/health",
            "stats": "GET /api/v1/stats",
            "search": "POST /api/v1/search",
            "query": "POST /api/v1/query",
            "websocket": "WS /api/v1/ws/query",
            "docs": "GET /docs",
            "upload_file": "POST /api/v1/files/upload",
            "upload_batch": "POST /api/v1/files/upload-batch",
            "delete_file": "DELETE /api/v1/files/{filename}",
            "index_documents": "POST /api/v1/documents/index",
            "delete_documents": "DELETE /api/v1/documents/{source}",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="0.0.0.0", port=8000, logger_level=os.getenv("LOG_LEVEL", "debug")
    )
