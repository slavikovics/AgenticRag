"""
tools/retrieval.py — retrieval tool handlers.

Single tool with a `collection` parameter — the LLM picks which
university collection (or the general collection) to query.
"""

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

GENERAL_COLLECTION = "general"


def _format_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No relevant documents found."
    formatted = [
        {
            "document": i,
            "source": r.get("source", ""),
            "title": r.get("title", ""),
            "score": r.get("score"),
            "content": (r.get("content") or "")[:1000],
        }
        for i, r in enumerate(results, 1)
    ]
    return json.dumps(formatted, ensure_ascii=False, indent=2)


async def handle_retrieve(
    retriever,
    query: str,
    collection: str = GENERAL_COLLECTION,
    limit: int = 5,
    alpha: float = 0.5,
    score_threshold: float = 0.5,
) -> str:
    try:
        results = await retriever.search(
            query=query,
            collection=collection,
            limit=limit,
            alpha=alpha,
            score_threshold=score_threshold,
        )
        return _format_results(results)
    except Exception as e:
        log.error("Retrieval failed: %s", e)
        return f"Error performing retrieval: {e}"
