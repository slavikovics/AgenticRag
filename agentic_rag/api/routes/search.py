"""search.py — direct collection search endpoint (bypass agent)."""

import logging

from fastapi import APIRouter, HTTPException

from ..dependencies import get_qdrant
from ..models import SearchRequest, SearchResponse, SearchResult

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Direct semantic search against a specific collection.
    Bypasses the agent loop — useful for debugging retrieval quality.
    """
    try:
        qdrant = await get_qdrant()
        results = await qdrant.search(
            query=request.query,
            collection=request.collection,
            limit=request.limit,
            alpha=request.alpha,
        )
        return SearchResponse(
            query=request.query,
            collection=request.collection,
            results=[
                SearchResult(
                    content=r["content"],
                    source=r["source"],
                    title=r.get("title", ""),
                    score=r["score"],
                )
                for r in results
            ],
            count=len(results),
        )
    except Exception as e:
        log.error("Search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
