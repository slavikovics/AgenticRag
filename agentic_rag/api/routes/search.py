import logging

from fastapi import APIRouter, HTTPException

from ..dependencies import get_qdrant_manager
from ..models import SearchRequest, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_knowledge_base(request: SearchRequest):
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
