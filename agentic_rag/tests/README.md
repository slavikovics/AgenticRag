# Agentic RAG Tests

This directory contains tests for the Agentic RAG application.

## Prerequisites

1. **Qdrant vector database** must be running:
   - Via Docker: `docker compose up -d qdrant`
   - Or use in-memory mode (no Docker needed)

2. **Environment variables** must be set:
   ```bash
   export OPENROUTER_API_KEY="your-api-key"
   export QDRANT_URL="http://localhost:6333"  # or use default
   export QDRANT_COLLECTION_NAME="documents"  # or use default
   ```

3. **Install test dependencies**:
   ```bash
   pip install pytest pytest-asyncio httpx
   ```

## Running Tests

### Run all tests:
```bash
cd agentic_rag
python -m pytest tests/ -v --asyncio-mode=auto
```

### Run specific test:
```bash
python -m pytest tests/test_rag.py::test_health_check -v --asyncio-mode=auto
```

### Run tests with output:
```bash
python -m pytest tests/ -v -s --asyncio-mode=auto
```

## Test Coverage

The tests cover the following functionality:

1. **Health Check** (`test_health_check`)
   - Verifies API is running
   - Checks Qdrant connection
   - Validates LLM client configuration

2. **Document Ingestion** (`test_ingest_documents`)
   - Tests bulk document indexing
   - Validates metadata handling
   - Checks insertion count

3. **Vector Search** (`test_vector_search`)
   - Tests pure vector similarity search
   - Validates result format and scoring

4. **Hybrid Search** (`test_hybrid_search`)
   - Tests combined vector + keyword search
   - Validates alpha parameter weighting

5. **Agent Query** (`test_agent_query`)
   - Tests full RAG pipeline
   - Validates answer generation
   - Checks source attribution

6. **Delete by Source** (`test_delete_by_source`)
   - Tests document deletion
   - Validates cleanup by source identifier

7. **Collection Stats** (`test_collection_stats`)
   - Tests statistics endpoint
   - Validates point and vector counts

8. **WebSocket Endpoint** (`test_websocket_query`)
   - Tests WebSocket route existence
   - Basic connectivity check

## Notes

- Tests use a local Qdrant instance (in-memory or Docker)
- Each test is independent but shares the same collection
- Tests that require OpenRouter API will fail without a valid key
- Some tests may take longer due to embedding generation

## Troubleshooting

### "Connection refused" errors:
Make sure Qdrant is running:
```bash
docker compose up -d qdrant
# or check if it's running
docker ps | grep qdrant
```

### "401 Unauthorized" errors:
Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Event loop errors:
Always use `--asyncio-mode=auto` flag when running pytest.
