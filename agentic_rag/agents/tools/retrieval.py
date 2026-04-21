import json
import logging

logger = logging.getLogger(__name__)


async def handle_retrieve(
    retriever, query: str, limit: int = 5, alpha: float = 0.5
) -> str:
    try:
        results = await retriever.hybrid_search(query=query, limit=limit, alpha=alpha)
        if not results:
            return "No relevant documents found in the knowledge base."
        formatted = []
        for i, result in enumerate(results, 1):
            source = result.get("source", "Unknown")
            score = result.get("score", "N/A")
            content = result.get("content", "")[:1000]
            search_result = {
                "document": i,
                "source": source,
                "score": score,
                "content": content,
            }
            formatted.append(search_result)
        return json.dumps(formatted, indent=2)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return f"Error performing retrieval: {str(e)}"


async def handle_semantic_search(retriever, query: str, limit: int = 5) -> str:
    try:
        results = await retriever.vector_search(query=query, limit=limit)
        if not results:
            return "No semantically similar documents found."
        formatted = []
        for i, result in enumerate(results, 1):
            source = result.get("source", "Unknown")
            distance = result.get("distance", "N/A")
            content = result.get("content", "")[:1000]
            search_result = {
                "document": i,
                "source": source,
                "score": distance,
                "content": content,
            }
            formatted.append(search_result)
        return json.dumps(formatted, indent=2)
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return f"Error performing semantic search: {str(e)}"
