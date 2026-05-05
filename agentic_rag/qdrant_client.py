"""
qdrant_client.py — multi-collection Qdrant search client.

Responsibilities:
- Embed queries using Qwen3-Embedding API
- Search any named collection
- List available collections

No indexing, no upsert — all data is prepared offline by the pipeline.
"""

import logging
from typing import Any, Optional

import aiohttp
from qdrant_client import QdrantClient

log = logging.getLogger(__name__)


# Qwen3-Embedding query instruction — must match what was used during offline indexing.
# Documents are embedded without any prefix; queries get this prefix.
QUERY_INSTRUCTION = "Instruct: Retrieve relevant university information for prospective students\nQuery: "


class Qwen3EmbedClient:
    """
    Async client for Qwen3-Embedding via OpenRouter.
    Reuses a single aiohttp session across requests.

    embed_query() prepends QUERY_INSTRUCTION — must match pipeline_pkg/config.py.
    embed_document() sends raw text with no prefix.
    """

    def __init__(self, api_key: str, api_base: str, model: str):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _embed_raw(self, text: str) -> list[float]:
        """Send text to the embeddings endpoint, return vector."""
        session = await self._get_session()
        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rag-bot",
            "X-Title": "Agentic RAG",
        }
        async with session.post(
            f"{self.api_base}/embeddings",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Embedding API {resp.status}: {body}")
            data = await resp.json()
            return data["data"][0]["embedding"]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a search query — prepends task instruction for better retrieval."""
        return await self._embed_raw(QUERY_INSTRUCTION + query)

    async def embed_document(self, text: str) -> list[float]:
        """Embed a document — no instruction prefix per Qwen3-Embedding spec."""
        return await self._embed_raw(text)

    # Keep embed() as alias for backward compat during transition
    async def embed(self, text: str) -> list[float]:
        return await self.embed_query(text)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class QdrantSearchClient:
    """
    Multi-collection Qdrant search client.
    Implements agents.retriever.RetrieverProtocol via duck typing.
    """

    def __init__(
        self,
        qdrant_url: str,
        embedder: Qwen3EmbedClient,
        grpc_port: int = 6334,
    ):
        self._client = QdrantClient(
            url=qdrant_url,
            grpc_port=grpc_port,
            prefer_grpc=True,
        )
        self._embedder = embedder

    # ── RetrieverProtocol interface ───────────────────────────────────────────

    def list_collections(self) -> list[str]:
        result = self._client.get_collections()
        return sorted(c.name for c in result.collections)

    async def search(
        self,
        query: str,
        collection: str,
        limit: int = 5,
        alpha: float = 0.5,  # accepted for interface compat, dense-only for now
        score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Search a specific collection by semantic similarity.
        Returns empty list if no results exceed score_threshold —
        this triggers the web_search fallback in the agent cleanly.
        """
        try:
            vector = await self._embedder.embed_query(query)
            result = self._client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold,
            )
            return [
                {
                    "source": hit.payload.get("doc_url", ""),
                    "title": hit.payload.get("doc_title", ""),
                    "score": hit.score,
                    "content": hit.payload.get("text", ""),
                }
                for hit in result.points
            ]
        except Exception as e:
            log.error("Search failed (collection=%s): %s", collection, e)
            raise

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_collection_stats(self, collection: str) -> dict[str, Any]:
        info = self._client.get_collection(collection)
        return {
            "collection": collection,
            "points_count": getattr(info, "points_count", 0),
            "status": str(getattr(info, "status", "unknown")),
            "vector_size": info.config.params.vectors.size,
        }

    def get_all_stats(self) -> list[dict[str, Any]]:
        return [self.get_collection_stats(name) for name in self.list_collections()]

    async def close(self):
        await self._embedder.close()
        self._client.close()
