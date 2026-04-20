# Agentic RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with agentic capabilities, using OpenRouter for both LLM and embeddings, with Qdrant as the vector database.

## Features

- **OpenRouter Integration**: Uses OpenRouter API for both LLM completions and text embeddings
- **Qdrant Vector Database**: Supports both server mode (Docker) and in-memory mode
- **Agentic RAG**: Intelligent agent that can iterate and refine answers
- **FastAPI Server**: Production-ready REST API with WebSocket support
- **Docker Support**: Easy deployment with docker-compose

## Quick Start

### Option 1: In-Memory Mode (No Docker Required)

1. Copy the environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and set your OpenRouter API key:
```
OPENROUTER_API_KEY=your_api_key_here
QDRANT_MODE=memory
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
python -m uvicorn agentic_rag.api.server:app --host 0.0.0.0 --port 8000
```

### Option 2: With Qdrant Server (Docker)

1. Copy the environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and set your OpenRouter API key:
```
OPENROUTER_API_KEY=your_api_key_here
QDRANT_MODE=remote
```

3. Start with docker-compose:
```bash
docker compose up -d --build
```

### Option 3: In-Memory Mode with Docker (Lightweight)

1. Copy the environment file:
```bash
cp .env.example .env
```

2. Edit `.env`:
```
OPENROUTER_API_KEY=your_api_key_here
QDRANT_MODE=memory
```

3. Start only the API service:
```bash
docker compose up -d --build api
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key (required) | - |
| `OPENROUTER_LLM_MODEL` | LLM model to use | `openai/gpt-4-turbo-preview` |
| `OPENROUTER_EMBEDDING_MODEL` | Embedding model to use | `openai/text-embedding-3-small` |
| `QDRANT_URL` | Qdrant server URL | `http://qdrant:6333` |
| `QDRANT_COLLECTION_NAME` | Collection name | `documents` |
| `QDRANT_MODE` | `remote` or `memory` | `remote` |
| `LOG_LEVEL` | Logging level | `info` |
| `MAX_ITERATIONS` | Max agent iterations | `10` |
| `TEMPERATURE` | LLM temperature | `0.7` |
| `MAX_TOKENS` | Max tokens in response | `2048` |
| `MEMORY_SIZE` | Agent memory size | `20` |

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /search` - Search knowledge base
- `POST /query` - Run agentic RAG query
- `WS /ws/query` - WebSocket for streaming responses
- `POST /documents/index` - Index documents
- `DELETE /documents/{source}` - Delete documents by source

## Example Usage

### Index Documents

```bash
curl -X POST http://localhost:8000/documents/index \
  -H "Content-Type: application/json" \
  -d '[{"content": "Your document text here", "source": "my-doc", "chunk_id": 1}]'
```

### Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "limit": 5}'
```

### Query with Agent

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain retrieval-augmented generation", "include_sources": true}'
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI     │────▶│  OpenRouter │
│             │◀────│  Server      │◀────│  (LLM+Emb)  │
└─────────────┘     └──────────────┘     └─────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │   Qdrant     │
                   │ (or Memory)  │
                   └──────────────┘
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Code Structure

```
agentic_rag/
├── api/
│   └── server.py          # FastAPI application
├── agents/
│   └── base_agent.py      # Agentic RAG implementation
├── llm/
│   └── openrouter_client.py  # OpenRouter LLM client
├── qdrant/
│   └── async_manager.py   # Qdrant vector DB manager
└── config.py              # Configuration settings
```

## License

MIT
