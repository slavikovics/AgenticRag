import logging
from typing import Optional

from agentic_rag.agents.base_agent import AgentConfig, AgenticRAG
from agentic_rag.config import settings
from agentic_rag.llm.openrouter_client import OpenRouterClient, OpenRouterConfig
from agentic_rag.qdrant.async_manager import AsyncQdrantManager

logger = logging.getLogger(__name__)

_llm_client = None
_qdrant_manager = None
_agent = None


async def get_llm_client():
    global _llm_client
    if _llm_client is None:
        config = OpenRouterConfig(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        _llm_client = OpenRouterClient(config)
    return _llm_client


async def get_qdrant_manager():
    global _qdrant_manager
    if _qdrant_manager is None:
        api_key = settings.openrouter_api_key
        _qdrant_manager = AsyncQdrantManager(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
            embedding_model=settings.openrouter_embedding_model,
            api_key=api_key,
        )
        await _qdrant_manager.connect()
        await _qdrant_manager.create_collection()
    return _qdrant_manager


async def get_agent(model: Optional[str] = None, temperature: float = 0.7):
    global _agent
    if _agent is None:
        llm = await get_llm_client()
        retriever = await get_qdrant_manager()
        config = AgentConfig(
            max_iterations=settings.max_iterations,
            memory_size=settings.memory_size,
            verbose=True,
        )
        _agent = AgenticRAG(llm, retriever, config)
    return _agent


async def cleanup_resources():
    global _qdrant_manager, _llm_client, _agent

    if _qdrant_manager:
        await _qdrant_manager.close()

    if _llm_client:
        await _llm_client.close()

    _agent = None
    logger.info("Resources cleaned up")
