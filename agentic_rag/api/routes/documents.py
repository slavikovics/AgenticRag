"""
Document management endpoints for Agentic RAG API.
"""

import logging

from fastapi import APIRouter, HTTPException

from ..dependencies import get_qdrant_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/documents/index")
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


@router.delete("/documents/{source}")
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
