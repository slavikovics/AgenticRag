"""health.py — health check and collection stats."""

import logging

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..dependencies import get_agent, get_qdrant
from ..models import CollectionInfo, HealthResponse

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        qdrant = await get_qdrant()
        agent = await get_agent()

        stats = qdrant.get_all_stats()
        collections = [
            CollectionInfo(
                name=s["collection"],
                points_count=s["points_count"],
                status=s["status"],
                vector_size=s["vector_size"],
            )
            for s in stats
        ]

        return HealthResponse(
            status="healthy",
            collections=collections,
            llm_model=settings.llm_model,
            web_search_enabled=bool(settings.tavily_api_key),
        )
    except Exception as e:
        log.error("Health check failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/collections")
async def list_collections():
    """List all available university collections."""
    try:
        qdrant = await get_qdrant()
        return {"collections": qdrant.list_collections()}
    except Exception as e:
        log.error("Failed to list collections: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection}/stats")
async def collection_stats(collection: str):
    """Stats for a specific collection."""
    try:
        qdrant = await get_qdrant()
        return qdrant.get_collection_stats(collection)
    except Exception as e:
        log.error("Failed to get stats for %s: %s", collection, e)
        raise HTTPException(status_code=500, detail=str(e))
