"""
Async Qdrant vector database manager.
Uses OpenRouter for embeddings.
"""

import os
import asyncio
import logging
from typing import Optional, Any, Dict, List
from datetime import datetime
import aiohttp
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

logger = logging.getLogger(__name__)


class AsyncOpenRouterEmbedding:
    """Async generate embeddings using OpenRouter API."""
    
    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-small"):
        if not api_key:
            raise ValueError("OpenRouter API key is required for embeddings")
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def embed_text(self, text: str) -> List[float]:
        await self._ensure_session()
        
        payload = {"model": self.model, "input": [text]}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/agentic-rag",
            "X-Title": "Agentic RAG System",
            "Content-Type": "application/json",
        }
        
        logger.debug(f"Requesting embedding for text length: {len(text)}")
        
        async with self.session.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Embedding API error {response.status}: {error_text}")
                logger.error(f"Headers sent: Authorization={'Bearer ' + self.api_key[:5] + '...' if self.api_key else 'MISSING'}")
                raise RuntimeError(f"Embedding API error {response.status}: {error_text}")
            result = await response.json()
            return result["data"][0]["embedding"]
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        await self._ensure_session()
        
        payload = {"model": self.model, "input": texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/agentic-rag",
            "X-Title": "Agentic RAG System",
            "Content-Type": "application/json",
        }
        
        logger.debug(f"Requesting embeddings for {len(texts)} texts")
        
        async with self.session.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Embedding API error {response.status}: {error_text}")
                logger.error(f"Headers sent: Authorization={'Bearer ' + self.api_key[:5] + '...' if self.api_key else 'MISSING'}")
                raise RuntimeError(f"Embedding API error {response.status}: {error_text}")
            result = await response.json()
            return [item["embedding"] for item in result["data"]]
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None


class AsyncQdrantManager:
    """Async Manager for Qdrant vector database operations."""
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "documents",
        embedding_model: str = "openai/text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self.url = url
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client: Optional[QdrantClient] = None
        self._connected = False
        self._embedder: Optional[AsyncOpenRouterEmbedding] = None
        self._embedding_dim = 1536
        
        # Use provided api_key or get from environment
        openrouter_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            logger.error("OPENROUTER_API_KEY not set. Embeddings will fail. Set it in .env file or pass it directly.")
        else:
            logger.info(f"Initializing OpenRouter embedder with key: {openrouter_api_key[:5]}...")
            self._embedder = AsyncOpenRouterEmbedding(openrouter_api_key, embedding_model)
    
    async def connect(self) -> None:
        try:
            # Check if we should use in-memory mode (for development/testing without Docker)
            qdrant_mode = os.getenv("QDRANT_MODE", "remote").lower()
            use_memory = qdrant_mode == "memory" or os.getenv("QDRANT_USE_MEMORY", "false").lower() == "true"
            
            if use_memory:
                logger.info("Using in-memory Qdrant client")
                self.client = QdrantClient(":memory:")
            else:
                logger.info(f"Connecting to Qdrant at {self.url}")
                self.client = QdrantClient(url=self.url)
            
            self.client.get_collections()
            self._connected = True
            logger.info(f"Connected to Qdrant")
        except Exception as e:
            logger.warning(f"Failed to connect to remote Qdrant: {e}")
            logger.info("Falling back to in-memory Qdrant client")
            self.client = QdrantClient(":memory:")
            self._connected = True
    
    async def ensure_connected(self) -> None:
        if not self._connected or not self.client:
            await self.connect()
    
    async def create_collection(self) -> None:
        await self.ensure_connected()
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection {self.collection_name} already exists")
                return
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_id",
                field_schema=PayloadSchemaType.INTEGER,
            )
            
            logger.info(f"Created collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    async def upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        await self.ensure_connected()
        try:
            if not self._embedder:
                raise RuntimeError("OpenRouter embedder not initialized")
            
            texts = [doc["content"] for doc in documents]
            embeddings = await self._embedder.embed_texts(texts)
            
            points = []
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
                point = PointStruct(
                    id=doc.get("id", abs(hash(doc["content"])) % (2**63)),
                    vector=embedding,
                    payload={
                        "content": doc["content"],
                        "source": doc.get("source", "unknown"),
                        "chunk_id": doc.get("chunk_id", 0),
                        "page": doc.get("page"),
                        "metadata": doc.get("metadata", {}),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "embedding_model": doc.get("embedding_model", self.embedding_model),
                    },
                )
                points.append(point)
            
            total_inserted = 0
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                result = self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
                if result.status == "completed":
                    total_inserted += len(batch)
            
            logger.info(f"Upserted {total_inserted} documents")
            return total_inserted
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            raise
    
    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        alpha: float = 0.5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        await self.ensure_connected()
        try:
            query_vector = None
            if self._embedder:
                query_vector = await self._embedder.embed_text(query)
            
            if not query_vector:
                raise RuntimeError("Could not generate query embedding")
            
            search_filter = None
            if where_filter:
                conditions = []
                for key, value in where_filter.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )
                if conditions:
                    search_filter = Filter(must=conditions)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=0.0,
            )
            
            formatted = []
            for result in results:
                formatted.append({
                    "content": result.payload.get("content", ""),
                    "source": result.payload.get("source"),
                    "chunk_id": result.payload.get("chunk_id"),
                    "page": result.payload.get("page"),
                    "metadata": result.payload.get("metadata"),
                    "score": result.score,
                    "id": result.id,
                })
            
            logger.debug(f"Search returned {len(formatted)} results")
            return formatted
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def vector_search(
        self,
        query: str,
        limit: int = 10,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return await self.hybrid_search(
            query=query,
            limit=limit,
            where_filter=where_filter,
        )
    
    async def delete_by_source(self, source: str) -> int:
        await self.ensure_connected()
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source),
                        )
                    ]
                ),
            )
            logger.info(f"Deleted documents from source: {source}")
            return result.operation_id if hasattr(result, 'operation_id') else 0
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        await self.ensure_connected()
        try:
            info = self.client.get_collection(self.collection_name)
            # Handle both server and in-memory modes
            vectors_count = getattr(info, 'vectors_count', None) or getattr(info, 'points_count', 0)
            points_count = getattr(info, 'points_count', 0)
            status = getattr(info, 'status', 'active')
            
            return {
                "collection_name": self.collection_name,
                "vectors_count": vectors_count if vectors_count is not None else points_count,
                "points_count": points_count,
                "status": str(status) if status else "active",
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise
    
    async def close(self) -> None:
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("Closed Qdrant connection")
        
        if self._embedder:
            await self._embedder.close()
