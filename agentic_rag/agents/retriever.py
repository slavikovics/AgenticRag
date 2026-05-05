"""
retriever.py — RetrieverProtocol.

Any class implementing these methods can be plugged into AgenticRAG.
The concrete implementation lives in api/qdrant_client.py.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RetrieverProtocol(Protocol):
    def list_collections(self) -> list[str]:
        """Return all available collection names."""
        ...

    async def search(
        self,
        query: str,
        collection: str,
        limit: int = 5,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Search a named collection by semantic similarity.
        alpha=0 → keyword, alpha=1 → semantic (currently dense-only).
        Returns list of dicts: source, title, score, content.
        """
        ...
