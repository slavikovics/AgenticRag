"""
Complete FastAPI server for Agentic RAG system.
Demonstrates full workflow with async operations.
"""

import os
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

# Import custom modules
from agentic_rag.llm.openrouter_client import OpenRouterClient, OpenRouterConfig
from agentic_rag.weaviate.manager import AsyncWeaviateManager
from agentic_rag.agents.base_agent import AgenticRAG, AgentConfig
from agentic_rag.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Pydantic Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request body for RAG query."""
    query: str
    model: Optional[str] = None  # Uses OPENROUTER_LLM_MODEL from env if not specified
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
    alpha: Optional[float] = 0.5  # Balance between BM25 and semantic search


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


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    weaviate: dict
    llm_client: str = "openrouter"
    version: str = "1.0.0"


# ============================================================================
# Global State (in production, use dependency injection)
# ============================================================================

# Lazy-loaded components
_llm_client = None
_weaviate_manager = None
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


async def get_weaviate_manager():
    """Lazy load Weaviate manager."""
    global _weaviate_manager
    if _weaviate_manager is None:
        _weaviate_manager = AsyncWeaviateManager(
            url=settings.weaviate_url,
            api_key=settings.weaviate_api_key,
            class_name=settings.weaviate_class_name,
            embedding_model=settings.openrouter_embedding_model,
        )
        await _weaviate_manager.connect()
        await _weaviate_manager.create_schema()
    return _weaviate_manager


async def get_agent(model: Optional[str] = None, temperature: float = 0.7):
    """Get or create agent."""
    global _agent
    if _agent is None:
        llm = await get_llm_client()
        retriever = await get_weaviate_manager()
        config = AgentConfig(
            max_iterations=settings.max_iterations,
            memory_size=settings.memory_size,
            verbose=True,
        )
        _agent = AgenticRAG(llm, retriever, config)
    return _agent


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Agentic RAG API",
    description="Production-ready RAG system with agents",
    version="1.0.0",
)

# Add CORS middleware
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
        weaviate = await get_weaviate_manager()
        llm = await get_llm_client()
        
        stats = {}
        if weaviate:
            stats = await weaviate.get_stats()
        
        return HealthResponse(
            status="healthy",
            weaviate=stats,
            llm_client="openrouter",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        weaviate = await get_weaviate_manager()
        stats = await weaviate.get_stats()
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
        retriever = await get_weaviate_manager()
        
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
    """
    Run agentic RAG query.
    
    This endpoint:
    1. Takes user query
    2. Runs agent loop (retrieve → reason → tool use)
    3. Returns final answer with sources
    """
    try:
        agent = await get_agent(
            model=request.model,
            temperature=request.temperature,
        )
        agent.config.max_iterations = request.max_iterations
        
        # Run agent
        answer = await agent.run(request.query)
        
        # Get sources from memory (simplified)
        history = agent.get_conversation_history()
        sources = agent.get_sources()
        
        # Get LLM cost
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
    """
    WebSocket endpoint for streaming agent responses.
    
    Message format:
    ```json
    {
        "type": "query",
        "payload": {
            "query": "What is X?",
            "model": "openai/gpt-4-turbo-preview"  // Optional, uses OPENROUTER_LLM_MODEL from env if not specified
        }
    }
    ```
    
    Response format:
    ```json
    {
        "type": "thinking" | "tool_use" | "answer",
        "content": "..."
    }
    ```
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            msg = await websocket.receive_text()
            data = json.loads(msg)
            
            if data.get("type") != "query":
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid message type",
                })
                continue
            
            query = data.get("payload", {}).get("query")
            if not query:
                await websocket.send_json({
                    "type": "error",
                    "content": "Missing query",
                })
                continue
            
            # Run agent and stream responses
            agent = await get_agent()
            
            # Send thinking
            await websocket.send_json({
                "type": "thinking",
                "content": f"Processing query: {query}",
            })
            
            # Run agent (in real implementation, hook into agent internals for streaming)
            answer = await agent.run(query)
            
            # Send answer
            await websocket.send_json({
                "type": "answer",
                "content": answer,
            })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": str(e),
            })
        except:
            pass


# ============================================================================
# Document Management Endpoints
# ============================================================================

@app.post("/documents/index")
async def index_documents(documents: list[dict]):
    """
    Index documents into Weaviate.
    
    Expected format:
    ```json
    [
        {
            "id": "doc-1",
            "content": "...",
            "source": "file.pdf",
            "chunk_id": 0,
            "metadata": {...}
        }
    ]
    ```
    """
    try:
        retriever = await get_weaviate_manager()
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
        retriever = await get_weaviate_manager()
        count = await retriever.delete_by_source(source)
        
        return {
            "status": "success",
            "documents_deleted": count,
        }
    
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Startup & Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize components on startup."""
    logger.info("Starting Agentic RAG API...")
    try:
        # Initialize managers
        llm = await get_llm_client()
        weaviate = await get_weaviate_manager()
        logger.info("Initialization complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")
    global _weaviate_manager, _llm_client, _agent
    
    if _weaviate_manager:
        await _weaviate_manager.close()
    
    if _llm_client:
        await _llm_client.close()
    
    _agent = None
    logger.info("Shutdown complete")


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
        },
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )