"""
Complete FastAPI server for Agentic RAG system.
Uses Qdrant vector database with OpenRouter for embeddings and LLM.
"""

import asyncio
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agentic_rag.agents.base_agent import AgentConfig, AgenticRAG
from agentic_rag.config import settings

# Import custom modules
from agentic_rag.llm.openrouter_client import OpenRouterClient, OpenRouterConfig
from agentic_rag.qdrant.async_manager import AsyncQdrantManager
from agentic_rag.utils.file_parser import chunk_text, process_file_to_documents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Pydantic Models
# ============================================================================


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


# ============================================================================
# Global State
# ============================================================================

_llm_client = None
_qdrant_manager = None
_agent = None


async def get_llm_client():
    """Lazy load LLM client."""
    global _llm_client
    if _llm_client is None:
        config = OpenRouterConfig(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        _llm_client = OpenRouterClient(config)
    return _llm_client


async def get_qdrant_manager():
    """Lazy load Qdrant manager."""
    global _qdrant_manager
    if _qdrant_manager is None:
        # Get API key from settings (which reads from environment)
        api_key = settings.openrouter_api_key
        _qdrant_manager = AsyncQdrantManager(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
            embedding_model=settings.openrouter_embedding_model,
            api_key=api_key,
        )
        await _qdrant_manager.connect()
        await _qdrant_manager.create_collection()
    return _qdrant_manager


async def get_agent(model: Optional[str] = None, temperature: float = 0.7):
    """Get or create agent."""
    global _agent
    if _agent is None:
        llm = await get_llm_client()
        retriever = await get_qdrant_manager()
        config = AgentConfig(
            max_iterations=settings.max_iterations,
            memory_size=settings.memory_size,
            verbose=True,
        )
        _agent = AgenticRAG(llm, retriever, config)
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    logger.info("Starting Agentic RAG API...")
    try:
        llm = await get_llm_client()
        qdrant = await get_qdrant_manager()
        logger.info("Initialization complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down...")
    global _qdrant_manager, _llm_client, _agent

    if _qdrant_manager:
        await _qdrant_manager.close()

    if _llm_client:
        await _llm_client.close()

    _agent = None
    logger.info("Shutdown complete")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Agentic RAG API",
    description="Production-ready RAG system with agents using Qdrant and OpenRouter",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Status Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    try:
        qdrant = await get_qdrant_manager()
        llm = await get_llm_client()

        stats = {}
        if qdrant:
            stats = await qdrant.get_stats()

        return HealthResponse(
            status="healthy",
            qdrant=stats,
            llm_client="openrouter",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        qdrant = await get_qdrant_manager()
        stats = await qdrant.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Retrieval Endpoints
# ============================================================================


@app.post("/search", response_model=SearchResponse)
async def search_knowledge_base(request: SearchRequest):
    """Search the knowledge base."""
    try:
        retriever = await get_qdrant_manager()

        results = await retriever.hybrid_search(
            query=request.query,
            limit=request.limit,
            alpha=request.alpha,
        )

        formatted = [
            SearchResult(
                content=r["content"],
                source=r["source"],
                chunk_id=r["chunk_id"],
                score=r["score"],
            )
            for r in results
        ]

        return SearchResponse(
            query=request.query,
            results=formatted,
            count=len(formatted),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Agentic RAG Endpoints
# ============================================================================


@app.post("/query", response_model=QueryResponse)
async def agentic_query(request: QueryRequest):
    """Run agentic RAG query."""
    try:
        agent = await get_agent(
            model=request.model,
            temperature=request.temperature,
        )
        agent.config.max_iterations = request.max_iterations

        answer = await agent.run(request.query)
        sources = agent.get_sources()

        llm = await get_llm_client()
        cost = getattr(llm, "total_cost", None)

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            iterations=agent.config.max_iterations,
            cost_usd=cost,
        )

    except asyncio.TimeoutError:
        logger.error("Agent query timeout")
        raise HTTPException(status_code=504, detail="Agent processing timeout")

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """WebSocket endpoint for streaming agent responses."""
    await websocket.accept()

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            if data.get("type") != "query":
                await websocket.send_json(
                    {
                        "type": "error",
                        "content": "Invalid message type",
                    }
                )
                continue

            query = data.get("payload", {}).get("query")
            if not query:
                await websocket.send_json(
                    {
                        "type": "error",
                        "content": "Missing query",
                    }
                )
                continue

            agent = await get_agent()

            await websocket.send_json(
                {
                    "type": "thinking",
                    "content": f"Processing query: {query}",
                }
            )

            answer = await agent.run(query)

            await websocket.send_json(
                {
                    "type": "answer",
                    "content": answer,
                }
            )

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "content": str(e),
                }
            )
        except:
            pass


# ============================================================================
# Document Management Endpoints
# ============================================================================


@app.post("/documents/index")
async def index_documents(documents: list[dict]):
    """Index documents into Qdrant."""
    try:
        retriever = await get_qdrant_manager()
        count = await retriever.upsert_documents(documents)

        return {
            "status": "success",
            "documents_indexed": count,
        }

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{source}")
async def delete_documents(source: str):
    """Delete all documents from a source."""
    try:
        retriever = await get_qdrant_manager()
        count = await retriever.delete_by_source(source)

        return {
            "status": "success",
            "documents_deleted": count,
        }

    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# File Upload Endpoints
# ============================================================================


@app.post("/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
):
    """Upload and index a single file."""
    try:
        # Validate file type
        allowed_extensions = {
            ".txt",
            ".text",
            ".md",
            ".markdown",
            ".pdf",
            ".csv",
            ".json",
        }
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {allowed_extensions}",
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Process file into documents
            documents = process_file_to_documents(
                file_path=tmp_path,
                source_name=file.filename,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Index documents
            retriever = await get_qdrant_manager()
            count = await retriever.upsert_documents(documents)

            return FileUploadResponse(
                status="success",
                filename=file.filename,
                documents_indexed=count,
                chunks=len(documents),
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/files/upload-batch", response_model=FilesUploadResponse)
async def upload_files_batch(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
):
    """Upload and index multiple files at once."""
    try:
        allowed_extensions = {
            ".txt",
            ".text",
            ".md",
            ".markdown",
            ".pdf",
            ".csv",
            ".json",
        }

        details = []
        total_documents = 0
        files_processed = 0

        for file in files:
            file_ext = os.path.splitext(file.filename)[1].lower()

            if file_ext not in allowed_extensions:
                details.append(
                    {
                        "filename": file.filename,
                        "status": "skipped",
                        "reason": f"Unsupported file type: {file_ext}",
                    }
                )
                continue

            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_ext
                ) as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_path = tmp_file.name

                try:
                    # Process file into documents
                    documents = process_file_to_documents(
                        file_path=tmp_path,
                        source_name=file.filename,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    # Index documents
                    retriever = await get_qdrant_manager()
                    count = await retriever.upsert_documents(documents)

                    details.append(
                        {
                            "filename": file.filename,
                            "status": "success",
                            "documents_indexed": count,
                            "chunks": len(documents),
                        }
                    )

                    total_documents += count
                    files_processed += 1
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            except Exception as e:
                logger.error(f"Failed to process file {file.filename}: {e}")
                details.append(
                    {
                        "filename": file.filename,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        return FilesUploadResponse(
            status="completed",
            files_processed=files_processed,
            total_documents_indexed=total_documents,
            details=details,
        )

    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete all documents from a specific file."""
    try:
        retriever = await get_qdrant_manager()
        count = await retriever.delete_by_source(filename)

        return {
            "status": "success",
            "filename": filename,
            "documents_deleted": count,
        }

    except Exception as e:
        logger.error(f"File deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Root Endpoint
# ============================================================================


@app.get("/")
async def root():
    """API root."""
    return {
        "name": "Agentic RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "search": "POST /search",
            "query": "POST /query",
            "websocket": "WS /ws/query",
            "docs": "GET /docs",
            "upload_file": "POST /files/upload",
            "upload_batch": "POST /files/upload-batch",
            "delete_file": "DELETE /files/{filename}",
            "index_documents": "POST /documents/index",
            "delete_documents": "DELETE /documents/{source}",
        },
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
