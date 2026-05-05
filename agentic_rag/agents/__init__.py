"""
agents — agentic RAG package.

Quick start:
    from agents import AgenticRAG, AgenticRAGSession, AgentConfig, LLMClient

    llm = LLMClient(model="openai/gpt-4o", api_key="sk-...")
    agent = AgenticRAG(llm=llm, retriever=my_retriever)

    async with AgenticRAGSession(agent) as session:
        answer = await session.query("What are the tuition fees at GRSU?")
"""

from .agent import AgenticRAG, AgenticRAGSession
from .config import AgentConfig
from .events import AgentEvent, EventType
from .llm import LLMClient
from .memory import AgentMessage, ConversationMemory
from .retriever import RetrieverProtocol
from .tools.base import ToolDefinition, ToolType

__all__ = [
    "AgenticRAG",
    "AgenticRAGSession",
    "AgentConfig",
    "AgentEvent",
    "EventType",
    "LLMClient",
    "AgentMessage",
    "ConversationMemory",
    "RetrieverProtocol",
    "ToolDefinition",
    "ToolType",
]
