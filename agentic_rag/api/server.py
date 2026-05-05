"""
server.py — FastAPI application.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .dependencies import cleanup, get_agent, get_qdrant
from .routes import health_router, query_router, search_router

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up...")
    await get_qdrant()  # verify Qdrant connection
    await get_agent()  # warm up agent + register tools
    log.info("Ready.")
    yield
    log.info("Shutting down...")
    await cleanup()


app = FastAPI(
    title="Agentic RAG API",
    description="University RAG system — query Belarusian university knowledge base",
    version="2.0.0",
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


@app.get("/")
async def root():
    return {
        "name": "Agentic RAG API",
        "version": "2.0.0",
        "endpoints": {
            "health": "GET  /api/v1/health",
            "collections": "GET  /api/v1/collections",
            "stats": "GET  /api/v1/collections/{collection}/stats",
            "search": "POST /api/v1/search",
            "query": "POST /api/v1/query",
            "websocket": "WS   /api/v1/ws/query",
            "docs": "GET  /docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level,
        reload=False,
    )
