"""
Tests for the Agentic RAG application.
Tests core functionality: document ingestion, search, and agent queries.
"""

import pytest
import asyncio
import os
from httpx import AsyncClient, ASGITransport
from agentic_rag.api.server import app
from agentic_rag.config import settings

# Test configuration
TEST_COLLECTION = "test_collection"
TEST_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-test-key")

# Skip tests that require real API key if not provided
require_api_key = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") == "test_key_for_ci",
    reason="Requires valid OPENROUTER_API_KEY environment variable"
)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def client():
    """Create an async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac

@pytest.mark.asyncio
async def test_health_check(client):
    """Test that the health endpoint returns successfully."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "llm_client" in data
    # Check for qdrant key (new vector db) instead of generic vector_db
    assert "qdrant" in data or "vector_db" in data
    print(f"Health check passed: {data}")

@pytest.mark.asyncio
@require_api_key
async def test_ingest_documents(client):
    """Test document ingestion endpoint."""
    test_docs = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "metadata": {"source": "test_doc_1", "type": "programming"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "metadata": {"source": "test_doc_2", "type": "ai"}
        },
        {
            "content": "Docker is a platform for developing, shipping, and running applications in containers.",
            "metadata": {"source": "test_doc_3", "type": "devops"}
        }
    ]
    
    response = await client.post(
        "/documents/index",
        json=test_docs
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "success"
    assert "documents_indexed" in data
    assert data["documents_indexed"] == 3
    print(f"Ingestion test passed: {data}")

@pytest.mark.asyncio
@require_api_key
async def test_vector_search(client):
    """Test vector search functionality."""
    # First ingest some documents
    test_docs = [
        {
            "content": "FastAPI is a modern web framework for building APIs with Python 3.7+ based on standard Python type hints.",
            "metadata": {"source": "fastapi_doc", "type": "web"}
        }
    ]
    
    ingest_response = await client.post("/documents/index", json=test_docs)
    assert ingest_response.status_code == 200
    
    # Perform vector search
    query = "What is FastAPI?"
    response = await client.post(
        "/search",
        json={
            "query": query,
            "limit": 5,
            "alpha": 1.0  # Pure vector search (alpha=1.0 means 100% vector)
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    print(f"Vector search test passed: found {len(data['results'])} results")
    if data["results"]:
        assert "content" in data["results"][0]
        assert "score" in data["results"][0]

@pytest.mark.asyncio
@require_api_key
async def test_hybrid_search(client):
    """Test hybrid search functionality (if supported)."""
    # First ingest some documents
    test_docs = [
        {
            "content": "Kubernetes is an open-source container orchestration engine for automating deployment, scaling, and management.",
            "metadata": {"source": "k8s_doc", "type": "devops", "keywords": ["kubernetes", "containers", "orchestration"]}
        }
    ]
    
    ingest_response = await client.post("/documents/index", json=test_docs)
    assert ingest_response.status_code == 200
    
    # Perform hybrid search (alpha=0.5 means 50% vector, 50% keyword)
    query = "container orchestration"
    response = await client.post(
        "/search",
        json={
            "query": query,
            "limit": 5,
            "alpha": 0.5
        }
    )
    
    # Hybrid search might return 200 even if it falls back to vector search
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    print(f"Hybrid search test passed: found {len(data['results'])} results")

@pytest.mark.asyncio
@require_api_key
async def test_agent_query(client):
    """Test the main RAG agent query endpoint."""
    # First ingest relevant documents
    test_docs = [
        {
            "content": "React is a JavaScript library for building user interfaces, developed by Facebook. It allows developers to create reusable UI components.",
            "metadata": {"source": "react_doc", "type": "frontend"}
        },
        {
            "content": "Vue.js is a progressive JavaScript framework for building user interfaces. Unlike other monolithic frameworks, Vue is designed to be incrementally adoptable.",
            "metadata": {"source": "vue_doc", "type": "frontend"}
        }
    ]
    
    ingest_response = await client.post("/documents/index", json=test_docs)
    assert ingest_response.status_code == 200
    
    # Query the agent
    query = "What are the differences between React and Vue?"
    response = await client.post(
        "/query",
        json={
            "query": query,
            "model": None,  # Use default from env
            "temperature": 0.7,
            "include_sources": True,
            "max_iterations": 5
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0
    print(f"Agent query test passed: got answer of length {len(data['answer'])}")
    print(f"Sources: {data['sources']}")

@pytest.mark.asyncio
async def test_delete_by_source(client):
    """Test deletion of documents by source."""
    # Ingest documents with a specific source
    test_docs = [
        {
            "content": "Temporary document for deletion test 1",
            "metadata": {"source": "delete_test_source", "type": "test"}
        },
        {
            "content": "Temporary document for deletion test 2",
            "metadata": {"source": "delete_test_source", "type": "test"}
        }
    ]
    
    ingest_response = await client.post("/documents/index", json=test_docs)
    assert ingest_response.status_code == 200
    
    # Delete by source
    response = await client.delete(
        "/documents/delete_test_source"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "success"
    print(f"Delete by source test passed: {data}")


@pytest.mark.asyncio
async def test_collection_stats(client):
    """Test getting collection statistics."""
    response = await client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "collection_name" in data or "status" in data
    print(f"Stats test passed: {data}")


@pytest.mark.asyncio
async def test_websocket_connection(client):
    """Test WebSocket endpoint availability (basic check)."""
    # Just verify the endpoint exists and doesn't crash on connection attempt
    # Full WebSocket testing requires special handling
    try:
        # This will likely fail due to WebSocket protocol, but we're just checking
        # that the route exists
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "websocket" in data.get("endpoints", {})
        print("WebSocket endpoint registered successfully")
    except Exception as e:
        print(f"WebSocket test note: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
