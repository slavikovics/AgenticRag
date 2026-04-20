"""
Main Agentic RAG system with ReAct pattern and tool execution.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .config import AgentConfig
from .memory import ConversationMemory
from .tools.definitions import ToolDefinition, ToolType

logger = logging.getLogger(__name__)


class AgenticRAG:
    """
    Agentic RAG system with ReAct pattern.
    Orchestrates retrieval, reasoning, and tool execution.
    """

    def __init__(
        self,
        llm_client,  # OpenRouter async client
        retriever,   # Async retriever (e.g., Weaviate wrapper)
        config: Optional[AgentConfig] = None,
    ):
        self.llm = llm_client
        self.retriever = retriever
        self.config = config or AgentConfig()

        self.tools: Dict[str, ToolDefinition] = {}
        self.memory = ConversationMemory(max_messages=self.config.memory_size)

        # Register default tools
        self._register_default_tools()

    def register_tool(self, tool: ToolDefinition):
        """Register a tool for agent use."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def _register_default_tools(self):
        """Register built-in tools."""

        # Retrieval tool
        retrieval_tool = ToolDefinition(
            name="retrieve_documents",
            description="Search the knowledge base for relevant documents using hybrid search (combines keyword and semantic search)",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant documents",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5,
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Search balance: 0=keyword search only, 1=semantic search only (default: 0.5)",
                        "default": 0.5,
                        "minimum": 0,
                        "maximum": 1,
                    }
                },
                "required": ["query"],
            },
            type=ToolType.RETRIEVAL,
            handler=self._handle_retrieve,
        )
        self.register_tool(retrieval_tool)

        # Semantic search tool
        semantic_tool = ToolDefinition(
            name="semantic_search",
            description="Perform pure semantic (vector) search on the knowledge base",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for semantic matching",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5,
                    }
                },
                "required": ["query"],
            },
            type=ToolType.RETRIEVAL,
            handler=self._handle_semantic_search,
        )
        self.register_tool(semantic_tool)

    async def _handle_retrieve(
        self,
        query: str,
        limit: int = 5,
        alpha: float = 0.5,
    ) -> str:
        """Handle retrieval tool execution."""
        try:
            results = await self.retriever.hybrid_search(
                query=query,
                limit=limit,
                alpha=alpha,
            )

            if not results:
                return "No relevant documents found in the knowledge base."

            # Format results for agent
            formatted = []
            for i, result in enumerate(results, 1):
                source = result.get('source', 'Unknown')
                score = result.get('score', 'N/A')
                content = result.get('content', '')[:1000]  # Limit content length

                formatted.append(
                    f"Document {i}:\n"
                    f"Source: {source}\n"
                    f"Relevance Score: {score}\n"
                    f"Content: {content}...\n"
                )

            return "\n---\n".join(formatted)

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return f"Error performing retrieval: {str(e)}"

    async def _handle_semantic_search(
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """Handle semantic search tool execution."""
        try:
            results = await self.retriever.vector_search(
                query=query,
                limit=limit,
            )

            if not results:
                return "No semantically similar documents found."

            formatted = []
            for i, result in enumerate(results, 1):
                source = result.get('source', 'Unknown')
                distance = result.get('distance', 'N/A')
                content = result.get('content', '')[:1000]

                formatted.append(
                    f"Document {i}:\n"
                    f"Source: {source}\n"
                    f"Semantic Distance: {distance}\n"
                    f"Content: {content}...\n"
                )

            return "\n---\n".join(formatted)

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return f"Error performing semantic search: {str(e)}"

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool by name."""
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"

        tool = self.tools[tool_name]

        try:
            if self.config.verbose:
                logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

            result = await tool.handler(**tool_input)
            return str(result)

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error executing tool: {str(e)}"

    async def execute_tools_concurrently(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Tuple[str, str, str]]:
        """
        Execute multiple tools concurrently.

        Args:
            tool_calls: List of tool call dicts with 'id', 'function.name' and 'function.arguments'

        Returns:
            List of (tool_call_id, tool_name, result) tuples
        """
        tasks = []
        tool_info = []

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id", "")
            function = tool_call.get("function", {})
            name = function.get("name", "")
            args_str = function.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}

            if name in self.tools:
                task = self.execute_tool(name, args)
                tasks.append(task)
                tool_info.append((tool_call_id, name))
            else:
                logger.warning(f"Unknown tool requested: {name}")

        # Run concurrently
        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        return [
            (tool_id, name, str(result) if not isinstance(result, Exception) else f"Error: {result}")
            for (tool_id, name), result in zip(tool_info, results)
        ]

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        tools_desc = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools.values()
        ])

        return f"""You are an AI assistant with access to a knowledge base and tools.

Your task is to answer user questions accurately and helpfully. Follow these guidelines:

1. For questions requiring specific information, use the retrieval tools to search the knowledge base.
2. Always cite sources when using information from retrieved documents.
3. If the knowledge base doesn't contain relevant information, be honest about it.
4. Think step by step about what information you need before using tools.

Available tools:
{tools_desc}

Important:
- Use retrieve_documents for general search (combines keyword and semantic search)
- Use semantic_search when looking for conceptually similar content
- Always read the retrieved content carefully before answering
- Be concise but thorough in your responses
- If a tool returns an error, try a different approach or inform the user"""

    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Call LLM with message history.
        Returns (content, tool_calls).
        """
        # Add tools to request
        tools = [tool.to_openai_format() for tool in self.tools.values()]

        # Call LLM
        content, tool_calls = await self.llm.agentic_complete(
            messages=messages,
            tools=tools if tools else None,
        )

        return content, tool_calls

    async def run(self, user_query: str) -> str:
        """
        Run the agentic RAG loop.

        Args:
            user_query: User question

        Returns:
            Final response
        """
        if self.config.verbose:
            logger.info(f"Starting agent loop for query: {user_query}")

        # Build initial messages
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_query},
        ]

        # Add to memory
        self.memory.add_message("user", user_query)

        iteration = 0
        final_answer = None

        try:
            while iteration < self.config.max_iterations:
                iteration += 1

                if self.config.verbose:
                    logger.info(f"Iteration {iteration}/{self.config.max_iterations}")

                # Call LLM
                response, tool_calls = await self._call_llm(messages)

                # Add assistant response to messages
                assistant_msg = {
                    "role": "assistant",
                    "content": response,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                messages.append(assistant_msg)
                self.memory.add_message(
                    "assistant",
                    response,
                    tool_calls=tool_calls,
                )

                # Check if we should stop
                if not tool_calls:
                    # No more tools to call - this is the final answer
                    final_answer = response
                    break

                # Execute tools
                if self.config.verbose:
                    logger.info(f"Executing {len(tool_calls)} tools...")

                tool_results = await self.execute_tools_concurrently(tool_calls)

                # Add tool results to messages
                for tool_call_id, tool_name, result in tool_results:
                    if self.config.verbose:
                        preview = result[:200] + "..." if len(result) > 200 else result
                        logger.info(f"Tool {tool_name} result: {preview}")

                    # Add tool response to messages
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result,
                    }
                    messages.append(tool_msg)

                    # Add to memory
                    self.memory.add_message(
                        "tool",
                        result,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )

        except asyncio.TimeoutError:
            logger.error("Agent loop timeout")
            final_answer = "I apologize, but the request timed out. Please try a simpler query or try again later."

        except Exception as e:
            logger.error(f"Agent loop error: {e}")
            final_answer = f"I encountered an error while processing your request: {str(e)}"

        if self.config.verbose:
            logger.info(f"Completed in {iteration} iterations")

        return final_answer or "I apologize, but I was unable to generate a response."

    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get formatted conversation history."""
        return self.memory.get_messages()

    def get_sources(self) -> List[Dict[str, Any]]:
        """Get sources used in the conversation."""
        return self.memory.get_sources()


# Async context manager for easy usage
class AgenticRAGSession:
    """Context manager for agentic RAG sessions."""

    def __init__(self, agent: AgenticRAG):
        self.agent = agent

    async def __aenter__(self):
        return self.agent

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.agent.clear_memory()

    async def query(self, question: str) -> str:
        """Run a query in this session."""
        return await self.agent.run(question)
