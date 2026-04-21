import logging

from fastapi import APIRouter, HTTPException

from ..dependencies import get_llm_client, get_qdrant_manager
from ..models import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
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


@router.get("/stats")
async def get_stats():
    try:
        qdrant = await get_qdrant_manager()
        stats = await qdrant.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
