"""
tools/web_search.py — Tavily web search tool.

Usage:
    from agents.tools.web_search import make_web_search_tool

    agent.register_tool(make_web_search_tool(api_key=os.getenv("TAVILY_API_KEY")))

Requirements:
    pip install tavily-python
"""

import json
import logging
from typing import Optional

from tavily import AsyncTavilyClient

from .base import ToolDefinition, ToolType

log = logging.getLogger(__name__)


async def handle_web_search(query: str, api_key: str, max_results: int = 5) -> str:
    try:
        client = AsyncTavilyClient(api_key=api_key)
        response = await client.search(
            query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
        )
        results = response.get("results", [])
        if not results:
            return "No web results found for this query."

        formatted = [
            {
                "result": i,
                "source": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", "")[:1000],
                "score": r.get("score"),
            }
            for i, r in enumerate(results, 1)
        ]
        return json.dumps(formatted, ensure_ascii=False, indent=2)

    except Exception as e:
        log.error("Web search failed: %s", e)
        return f"Error performing web search: {e}"


def make_web_search_tool(api_key: str, max_results: int = 5) -> ToolDefinition:
    """
    Create a Tavily web search tool ready to register with AgenticRAG.

    Example:
        agent.register_tool(make_web_search_tool(api_key=os.getenv("TAVILY_API_KEY")))
    """
    return ToolDefinition(
        name="web_search",
        description=(
            "Search the internet using Tavily when the local knowledge base "
            "does not contain relevant information. Use as a fallback after "
            "retrieve_documents returns no useful results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find information on the web",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Number of results to return (default: {max_results})",
                    "default": max_results,
                },
            },
            "required": ["query"],
        },
        type=ToolType.EXTERNAL_API,
        handler=lambda **kw: handle_web_search(api_key=api_key, **kw),
    )
