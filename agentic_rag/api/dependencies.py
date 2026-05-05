"""
dependencies.py — FastAPI dependency singletons.

All heavy objects (LLM client, Qdrant client, agent) are created once
at startup and reused across requests.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from agentic_rag.agents import AgentConfig, AgenticRAG, LLMClient
from agentic_rag.agents.tools.web_search import make_web_search_tool

from ..config import settings
from ..qdrant_client import QdrantSearchClient, Qwen3EmbedClient

# Load collection descriptions from JSON — used to build the LLM system prompt
_COLLECTIONS_FILE = Path(__file__).parent / "collections.json"


def _load_collection_descriptions() -> dict:
    if _COLLECTIONS_FILE.exists():
        try:
            return json.loads(_COLLECTIONS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Failed to load collections.json: %s", e)
    return {}


log = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────

_qdrant: Optional[QdrantSearchClient] = None
_llm: Optional[LLMClient] = None
_agent: Optional[AgenticRAG] = None


async def get_qdrant() -> QdrantSearchClient:
    global _qdrant
    if _qdrant is None:
        embedder = Qwen3EmbedClient(
            api_key=settings.openrouter_api_key,
            api_base=settings.openrouter_base_url,
            model=settings.embedding_model,
        )
        _qdrant = QdrantSearchClient(
            qdrant_url=settings.qdrant_url,
            embedder=embedder,
            grpc_port=settings.qdrant_grpc_port,
        )
        log.info("Qdrant client ready. Collections: %s", _qdrant.list_collections())
    return _qdrant


async def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient(
            model=settings.llm_model,
            api_key=settings.openrouter_api_key or None,
            api_base=settings.openrouter_base_url or None,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            retry_attempts=3,
        )
        log.info("LLM client ready: %s", settings.llm_model)
    return _llm


async def get_agent() -> AgenticRAG:
    global _agent
    if _agent is None:
        llm = await get_llm()
        retriever = await get_qdrant()

        config = AgentConfig(
            max_iterations=settings.max_iterations,
            timeout_seconds=settings.agent_timeout,
            memory_size=settings.memory_size,
            verbose=True,
        )

        _agent = AgenticRAG(
            llm=llm,
            retriever=retriever,
            config=config,
            collection_descriptions=_load_collection_descriptions(),
        )

        # Register Tavily web search if key is configured
        if settings.tavily_api_key:
            _agent.register_tool(make_web_search_tool(api_key=settings.tavily_api_key))
            log.info("Tavily web search tool registered")
        else:
            log.info("TAVILY_API_KEY not set — web search disabled")

        log.info("Agent ready")
    return _agent


async def cleanup():
    global _qdrant, _llm, _agent
    if _qdrant:
        await _qdrant.close()
    _qdrant = None
    _llm = None
    _agent = None
    log.info("Resources cleaned up")
