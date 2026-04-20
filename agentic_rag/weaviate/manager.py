"""
Weaviate vector database manager with hybrid search (BM25 + semantic).
"""

import os
import asyncio
import logging
from typing import Optional, Any, Dict, List
from datetime import datetime
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.collections.classes.types import WeaviateField

logger = logging.getLogger(__name__)


class WeaviateManager:
    """Manager for Weaviate vector database operations."""
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        class_name: str = "Document",
    ):
        """
        Initialize Weaviate manager.
        
        Args:
            url: Weaviate instance URL
            api_key: API key if using Weaviate Cloud
            class_name: Class name for document storage
        """
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        self.client = None
        self._connected = False
    
    def connect(self) -> None:
        """Establish connection to Weaviate."""
        try:
            # Connect to Weaviate with OpenRouter-compatible configuration
            if self.api_key:
                self.client = weaviate.connect_to_custom(
                    http_host=self.url.replace("http://", "").replace("https://", "").split(":")[0],
                    http_port=int(self.url.split(":")[-1]) if ":" in self.url else 8080,
                    http_secure=self.url.startswith("https"),
                    grpc_host=self.url.replace("http://", "").replace("https://", "").split(":")[0],
                    grpc_port=50051,
                    grpc_secure=self.url.startswith("https"),
                    headers={"X-API-Key": self.api_key} if self.api_key else None
                )
            else:
                self.client = weaviate.connect_to_local()
            
            # Verify connection
            if self.client.is_ready():
                self._connected = True
                logger.info(f"Connected to Weaviate at {self.url}")
            else:
                raise RuntimeError("Weaviate not ready")
        
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    
    def ensure_connected(self) -> None:
        """Ensure connection is established."""
        if not self._connected or not self.client:
            self.connect()
    
    def create_schema(self) -> None:
        """Create or update schema for document storage."""
        self.ensure_connected()
        
        try:
            # Check if collection exists
            collections = self.client.collections.list_all()
            if self.class_name in collections:
                logger.info(f"Collection {self.class_name} already exists")
                return
            
            # Create collection with OpenRouter-compatible embedding module
            # Using text2vec-openai module but configured to work with OpenRouter's OpenAI-compatible API
            collection = self.client.collections.create(
                name=self.class_name,
                description="Document chunks for RAG",
                vectorizer_config=Configure.Vectorizer.text2vec_openai(),
                properties=[
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                        description="The text content of the chunk",
                    ),
                    Property(
                        name="source",
                        data_type=DataType.TEXT,
                        description="Source document filename",
                    ),
                    Property(
                        name="chunk_id",
                        data_type=DataType.INT,
                        description="Chunk sequence number",
                    ),
                    Property(
                        name="page",
                        data_type=DataType.INT,
                        description="Page number if applicable",
                    ),
                    Property(
                        name="metadata",
                        data_type=DataType.OBJECT,
                        description="Additional metadata",
                    ),
                    Property(
                        name="timestamp",
                        data_type=DataType.DATE,
                        description="Indexing timestamp",
                    ),
                    Property(
                        name="embedding_model",
                        data_type=DataType.TEXT,
                        description="Which embedding model was used",
                    ),
                ],
            )
            
            logger.info(f"Created collection {self.class_name}")
        
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise
    
    def upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert documents into Weaviate.
        
        Args:
            documents: List of document dicts with 'content', 'source', etc.
            batch_size: Batch size for insertion
        
        Returns:
            Number of documents inserted/updated
        """
        self.ensure_connected()
        
        try:
            collection = self.client.collections.get(self.class_name)
            
            with collection.batch.dynamic() as batch:
                for doc in documents:
                    obj = {
                        "content": doc["content"],
                        "source": doc.get("source", "unknown"),
                        "chunk_id": doc.get("chunk_id", 0),
                        "page": doc.get("page"),
                        "metadata": doc.get("metadata", {}),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "embedding_model": doc.get("embedding_model", "unknown"),
                    }
                    
                    # Use content hash as UUID if provided
                    uuid = doc.get("id")
                    if uuid:
                        batch.add_object(properties=obj, uuid=uuid)
                    else:
                        batch.add_object(properties=obj)
            
            failed = collection.batch.failed_objects
            if failed:
                logger.warning(f"Failed to insert {len(failed)} objects")
            
            inserted = len(documents) - len(failed)
            logger.info(f"Upserted {inserted} documents")
            return inserted
        
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            raise
    
    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        alpha: float = 0.5,  # 0 = BM25 only, 1 = vector only
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining BM25 (keyword) and vector (semantic) search.
        
        Args:
            query: Search query
            limit: Max results
            alpha: Balance between BM25 (0) and vector (1) search
            where_filter: Optional WHERE filter
        
        Returns:
            List of results with scores and metadata
        """
        self.ensure_connected()
        
        try:
            collection = self.client.collections.get(self.class_name)
            
            # Build query
            hybrid_query = collection.query.hybrid(
                query=query,
                alpha=alpha,
                limit=limit,
                return_metadata=["score"],
            )
            
            if where_filter:
                hybrid_query = hybrid_query.with_where(where_filter)
            
            response = hybrid_query.fetch_objects()
            
            # Format results
            formatted = []
            for obj in response.objects:
                formatted.append({
                    "content": obj.properties.get("content", ""),
                    "source": obj.properties.get("source"),
                    "chunk_id": obj.properties.get("chunk_id"),
                    "page": obj.properties.get("page"),
                    "metadata": obj.properties.get("metadata"),
                    "score": obj.metadata.score if obj.metadata else 0,
                    "uuid": str(obj.uuid),
                })
            
            logger.debug(f"Hybrid search returned {len(formatted)} results")
            return formatted
        
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise
    
    def vector_search(
        self,
        query: str,
        limit: int = 10,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Pure vector (semantic) search.
        
        Args:
            query: Query text
            limit: Max results
            where_filter: Optional WHERE filter
        
        Returns:
            List of results
        """
        self.ensure_connected()
        
        try:
            collection = self.client.collections.get(self.class_name)
            
            near_text_query = collection.query.near_text(
                query=query,
                limit=limit,
                return_metadata=["distance"],
            )
            
            if where_filter:
                near_text_query = near_text_query.with_where(where_filter)
            
            response = near_text_query.fetch_objects()
            
            formatted = []
            for obj in response.objects:
                formatted.append({
                    "content": obj.properties.get("content", ""),
                    "source": obj.properties.get("source"),
                    "chunk_id": obj.properties.get("chunk_id"),
                    "page": obj.properties.get("page"),
                    "metadata": obj.properties.get("metadata"),
                    "distance": obj.metadata.distance if obj.metadata else float("inf"),
                    "uuid": str(obj.uuid),
                })
            
            return formatted
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    def delete_by_source(self, source: str) -> int:
        """
        Delete all documents from a specific source.
        
        Args:
            source: Source identifier
        
        Returns:
            Number of documents deleted
        """
        self.ensure_connected()
        
        try:
            collection = self.client.collections.get(self.class_name)
            
            # Delete objects matching the source
            result = collection.data.delete_many(
                where={
                    "path": ["source"],
                    "operator": "Equal",
                    "valueString": source,
                }
            )
            
            count = result.get("matches", 0)
            logger.info(f"Deleted {count} documents from source: {source}")
            return count
        
        except Exception as e:
            logger.error(f"Failed to delete by source: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        self.ensure_connected()
        
        try:
            meta = self.client.get_meta()
            collections = self.client.collections.list_all()
            
            # Get object count
            collection = self.client.collections.get(self.class_name)
            agg_result = collection.aggregate.over_all(total_count=True)
            
            return {
                "version": meta.get("version"),
                "total_documents": agg_result.total_count,
                "collections": len(collections),
                "status": "connected",
            }
        
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "disconnected", "error": str(e)}
    
    def close(self) -> None:
        """Close connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("Closed Weaviate connection")


# Async wrapper for integration with async RAG pipeline
class AsyncWeaviateManager:
    """Async wrapper around WeaviateManager."""
    
    def __init__(self, *args, **kwargs):
        self._manager = WeaviateManager(*args, **kwargs)
        self._lock = asyncio.Lock()
    
    async def connect(self):
        """Async connect."""
        async with self._lock:
            await asyncio.to_thread(self._manager.connect)
    
    async def create_schema(self):
        """Async create schema."""
        async with self._lock:
            await asyncio.to_thread(self._manager.create_schema)
    
    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        alpha: float = 0.5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Async hybrid search."""
        async with self._lock:
            return await asyncio.to_thread(
                self._manager.hybrid_search,
                query,
                limit,
                alpha,
                where_filter,
            )
    
    async def vector_search(
        self,
        query: str,
        limit: int = 10,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Async vector search."""
        async with self._lock:
            return await asyncio.to_thread(
                self._manager.vector_search,
                query,
                limit,
                where_filter,
            )
    
    async def upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Async upsert."""
        async with self._lock:
            return await asyncio.to_thread(
                self._manager.upsert_documents,
                documents,
                batch_size,
            )
    
    async def delete_by_source(self, source: str) -> int:
        """Async delete by source."""
        async with self._lock:
            return await asyncio.to_thread(
                self._manager.delete_by_source,
                source,
            )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Async stats."""
        async with self._lock:
            return await asyncio.to_thread(self._manager.get_stats)
    
    async def close(self):
        """Async close."""
        async with self._lock:
            await asyncio.to_thread(self._manager.close)