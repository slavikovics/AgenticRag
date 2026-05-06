# Agentic RAG

A tool-augmented RAG system that searches named Qdrant collections and falls back to web search when local knowledge is insufficient.

---

## Architecture

```
agents/          Core agent loop, LLM client, tools, memory
api/             FastAPI server, routes, Qdrant search client
```

```
User query
    │
    ▼
WebSocket / POST /query
    │
    ▼
AgenticRAG.run()
    ├── retrieve_documents(query, collection="my_collection")  ← Qdrant dense search
    └── web_search(query)                                      ← Tavily if nothing found
    │
    ▼
LLM synthesises answer from retrieved chunks
    │
    ▼
Streamed events → client  (or JSON response for POST)
```

## Configuration

Copy `.env.example` to `.env`:

```bash
# OpenRouter — used for both LLM completions and query embeddings
OPENROUTER_API_KEY=sk-or-...

# Tavily web search (optional — web_search tool is disabled if not set)
TAVILY_API_KEY=tvly-...

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_GRPC_PORT=6334

# LLM — must include provider/ prefix for litellm routing
LLM_MODEL=openrouter/qwen/qwen3.5-9b

# Embedding model — OpenRouter slug, no prefix
EMBEDDING_MODEL=qwen/qwen3-embedding-8b
```

All settings can also be overridden in `agentic_rag/config.py`.

---

## API reference

### `GET /api/v1/health`
Server status and stats for all Qdrant collections.

```json
{
  "status": "healthy",
  "collections": [
    {"name": "docs", "points_count": 4821, "status": "green", "vector_size": 4096}
  ],
  "llm_model": "openrouter/qwen/qwen3.5-9b",
  "web_search_enabled": true
}
```

### `GET /api/v1/collections`
List all available collection names.

```json
{"collections": ["general", "docs", "products"]}
```

### `GET /api/v1/collections/{collection}/stats`
Points count and status for a specific collection.

### `POST /api/v1/search`
Direct semantic search — bypasses the agent. Useful for debugging retrieval quality.

```json
{
  "query": "how to reset a password",
  "collection": "docs",
  "limit": 5,
  "score_threshold": 0.5
}
```

Response:
```json
{
  "query": "how to reset a password",
  "collection": "docs",
  "results": [
    {
      "content": "...",
      "source": "https://example.com/docs/auth",
      "title": "Authentication",
      "score": 0.812
    }
  ],
  "count": 3
}
```

### `POST /api/v1/query`
Run the full agentic RAG loop. Returns final answer + sources.

```json
{
  "query": "How do I configure SSO?",
  "collection_hint": "docs",
  "max_iterations": 10
}
```

Response:
```json
{
  "query": "How do I configure SSO?",
  "answer": "To configure SSO, navigate to...",
  "sources": [
    {"tool": "retrieve_documents", "source": "https://example.com/docs/sso", "score": 0.84}
  ]
}
```

`collection_hint` is optional — the agent selects the collection automatically based on the query. Pass it when you know exactly which collection the answer should come from.

### `WS /api/v1/ws/query`
Persistent WebSocket for streaming. **The connection stays open across multiple queries** — send as many messages as needed without reconnecting.

**Client → server:**
```json
{"type": "query", "payload": {"query": "...", "collection_hint": "docs"}}
```

**Server → client** (one message per event):
```json
{"type": "iteration_start", "data": {"query": "...", "max_iterations": 10}}
{"type": "tool_call",        "data": {"tool_name": "retrieve_documents", "arguments": {...}}}
{"type": "tool_result",      "data": {"tool_name": "retrieve_documents", "success": true}}
{"type": "llm_response",     "content": "To configure SSO...", "data": {"iteration": 2}}
{"type": "answer",           "content": "To configure SSO..."}
{"type": "complete",         "data": {"status": "success"}}
```

After `complete`, the socket waits for the next `query` message. On agent error, an `error` event is sent and the socket remains open.

---

## Collections

One Qdrant collection per knowledge domain. Collection names are arbitrary — any slug works.

Collection descriptions are stored in `api/collections.json` and used to build the agent system prompt so the LLM knows which collection to search for which type of query:

```json
{
  "general": {
    "display_name": "General",
    "description": "Aggregated overview and cross-domain statistics"
  },
  "docs": {
    "display_name": "Product Docs",
    "description": "Technical documentation, API reference, guides"
  }
}
```

The API picks up new collections automatically on next startup.

---

## Agents package

The `agents/` package works independently of the API:

```python
from agents import AgenticRAG, AgentConfig, LLMClient
from agents.tools.web_search import make_web_search_tool

llm = LLMClient(
    model="openrouter/qwen/qwen3.5-9b",
    api_key="sk-or-...",
    api_base="https://openrouter.ai/api/v1",
)

agent = AgenticRAG(
    llm=llm,
    retriever=my_retriever,       # any object with list_collections() + search()
    config=AgentConfig(max_iterations=8),
    collection_descriptions={     # optional — improves collection routing
        "docs": {"display_name": "Docs", "description": "Product documentation"}
    },
)

# Optional: add web search fallback
agent.register_tool(make_web_search_tool(api_key="tvly-..."))

# Stream events
async def on_event(event):
    print(event.type, event.content)

answer = await agent.run("How do I configure SSO?", on_event=on_event)
sources = agent.get_sources()
```

### Event types

| Type | When | Key data fields |
|---|---|---|
| `iteration_start` | Once at the start | `query`, `max_iterations` |
| `tool_call` | Before each tool runs | `tool_name`, `arguments` |
| `tool_result` | After each tool runs | `tool_name`, `success`, `result_length` |
| `llm_response` | After each LLM call | `iteration`, `has_tool_calls` |
| `answer` | When final answer is ready | `content` |
| `complete` | Always last | `status` |
| `error` | On timeout or exception | `content` |

### Adding a custom tool

```python
from agents.tools.base import ToolDefinition, ToolType

agent.register_tool(ToolDefinition(
    name="get_pricing",
    description="Fetch current pricing information for a product",
    parameters={
        "type": "object",
        "properties": {
            "product_id": {"type": "string", "description": "Product identifier"}
        },
        "required": ["product_id"],
    },
    type=ToolType.EXTERNAL_API,
    handler=my_async_handler,
))
```

### Implementing a custom retriever

Any object with `list_collections()` and `search()` works:

```python
class MyRetriever:
    def list_collections(self) -> list[str]:
        return ["docs", "general"]

    async def search(
        self,
        query: str,
        collection: str,
        limit: int = 5,
        alpha: float = 0.5,
        score_threshold: float = 0.5,
    ) -> list[dict]:
        # Returns list of: {source, title, score, content}
        ...
```

---

## LLM provider

The agent uses [LiteLLM](https://docs.litellm.ai) which supports 100+ providers.  
Switch the model with no code changes:

```bash
# OpenRouter
LLM_MODEL=openrouter/qwen/qwen3.5-9b

# OpenAI
LLM_MODEL=openai/gpt-4o

# Anthropic
LLM_MODEL=anthropic/claude-sonnet-4-5

# Local Ollama
LLM_MODEL=ollama/llama3
LLM_API_BASE=http://localhost:11434
```

Always use the `provider/model-name` format. Retries with exponential backoff are applied automatically (3 attempts by default, configurable via `AgentConfig`).
