import asyncio
import json
import logging
from typing import Any, Callable, Optional

from agentic_rag.agents.config import AgentConfig
from agentic_rag.agents.events import AgentEvent, EventType
from agentic_rag.agents.memory import ConversationMemory
from agentic_rag.agents.tools.base import ToolDefinition, ToolType
from agentic_rag.agents.tools.retrieval import handle_retrieve, handle_semantic_search

logger = logging.getLogger(__name__)


class AgenticRAG:
    def __init__(self, llm_client, retriever, config: AgentConfig = None):
        self.llm = llm_client
        self.retriever = retriever
        self.config = config or AgentConfig()
        self.tools: dict[str, ToolDefinition] = {}
        self.memory = ConversationMemory(max_messages=self.config.memory_size)
        self._register_default_tools()

    def register_tool(self, tool: ToolDefinition):
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def _register_default_tools(self):
        retrieval_tool = ToolDefinition(
            name="retrieve_documents",
            description="Search the knowledge base for relevant documents using hybrid search",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 5,
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Search balance 0=keyword 1=semantic",
                        "default": 0.5,
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": ["query"],
            },
            type=ToolType.RETRIEVAL,
            handler=lambda **kw: handle_retrieve(self.retriever, **kw),
        )
        self.register_tool(retrieval_tool)
        semantic_tool = ToolDefinition(
            name="semantic_search",
            description="Perform pure semantic vector search",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            type=ToolType.RETRIEVAL,
            handler=lambda **kw: handle_semantic_search(self.retriever, **kw),
        )
        self.register_tool(semantic_tool)

    async def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
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
        self, tool_calls: list[dict[str, Any]]
    ) -> list[tuple[str, str, str]]:
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
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            (
                tool_id,
                name,
                str(result)
                if not isinstance(result, Exception)
                else f"Error: {result}",
            )
            for (tool_id, name), result in zip(tool_info, results)
        ]

    def _build_system_prompt(self) -> str:
        tools_desc = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in self.tools.values()]
        )
        return f"""You are an AI assistant with access to a knowledge base and tools.
Your task is to answer user questions accurately and helpfully.
1. For questions requiring specific information, use retrieval tools.
2. Always cite sources when using retrieved documents.
3. If the knowledge base lacks relevant info, be honest.
4. Think step by step before using tools.
Available tools:
{tools_desc}
Important:
- Use retrieve_documents for general search
- Use semantic_search for conceptually similar content
- Read retrieved content carefully before answering
- Be concise but thorough
- If a tool errors, try different approach or inform user"""

    async def _call_llm(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, Optional[list[dict[str, Any]]]]:
        tools = [tool.to_openai_format() for tool in self.tools.values()]
        content, tool_calls = await self.llm.agentic_complete(
            messages=messages, tools=tools if tools else None
        )
        return content, tool_calls

    async def _emit(
        self,
        on_event: Optional[Callable[[AgentEvent], Any]],
        event_type: EventType,
        content: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
    ):
        if on_event:
            event = AgentEvent(type=event_type, content=content, data=data)
            if asyncio.iscoroutinefunction(on_event):
                await on_event(event)
            else:
                on_event(event)

    async def run(
        self, user_query: str, on_event: Optional[Callable[[AgentEvent], Any]] = None
    ) -> str:
        logger.info(f"Starting agent loop for query: {user_query}")

        # --- Session-level start event, fired once ---
        await self._emit(
            on_event,
            EventType.ITERATION_START,
            data={
                "query": user_query,
                "max_iterations": self.config.max_iterations,
            },
        )

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_query},
        ]
        self.memory.add_message("user", user_query)

        final_answer: Optional[str] = None

        try:
            for iteration in range(1, self.config.max_iterations + 1):
                if self.config.verbose:
                    logger.info(f"Iteration {iteration}/{self.config.max_iterations}")

                response, tool_calls = await self._call_llm(messages)
                response = response or ""

                await self._emit(
                    on_event,
                    EventType.LLM_RESPONSE,
                    content=response,
                    data={
                        "iteration": iteration,
                        "has_tool_calls": bool(tool_calls),
                        "tool_call_count": len(tool_calls) if tool_calls else 0,
                    },
                )

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)
                self.memory.add_message("assistant", response, tool_calls=tool_calls)

                if not tool_calls:
                    final_answer = response
                    break

                # --- Emit TOOL_CALL before execution ---
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args_str = fn.get("arguments", "{}")
                    try:
                        args = (
                            json.loads(args_str)
                            if isinstance(args_str, str)
                            else args_str
                        )
                    except json.JSONDecodeError:
                        args = {}
                    await self._emit(
                        on_event,
                        EventType.TOOL_CALL,
                        data={
                            "tool_call_id": tc.get("id"),
                            "tool_name": fn.get("name"),
                            "arguments": args,
                        },
                    )

                if self.config.verbose:
                    logger.info(f"Executing {len(tool_calls)} tools...")

                tool_results = await self.execute_tools_concurrently(tool_calls)

                for tool_call_id, tool_name, result in tool_results:
                    await self._emit(
                        on_event,
                        EventType.TOOL_RESULT,
                        data={
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "result_preview": result,
                            "result_length": len(result),
                            "success": not result.startswith("Error:"),
                        },
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result,
                        }
                    )
                    self.memory.add_message(
                        "tool", result, tool_call_id=tool_call_id, name=tool_name
                    )

                # --- Force synthesis on last iteration if still looping ---
                if iteration == self.config.max_iterations:
                    logger.warning(
                        "Max iterations reached — forcing final synthesis call."
                    )
                    synthesis_response, _ = await self._call_llm(messages)
                    final_answer = (
                        synthesis_response
                        or "I reached the iteration limit without a conclusive answer."
                    )
                    await self._emit(
                        on_event,
                        EventType.LLM_RESPONSE,
                        content=final_answer,
                        data={
                            "iteration": iteration + 1,
                            "has_tool_calls": False,
                            "tool_call_count": 0,
                            "forced_synthesis": True,
                        },
                    )

        except asyncio.TimeoutError:
            logger.error("Agent loop timeout")
            final_answer = "I apologize, but the request timed out."
            await self._emit(on_event, EventType.ERROR, content=final_answer)

        except Exception as e:
            logger.error(f"Agent loop error: {e}")
            final_answer = f"I encountered an error: {str(e)}"
            await self._emit(on_event, EventType.ERROR, content=final_answer)

        # --- Always fires regardless of outcome ---
        final_answer = final_answer or "I was unable to generate a response."
        await self._emit(on_event, EventType.ANSWER, content=final_answer)
        await self._emit(
            on_event,
            EventType.COMPLETE,
            data={
                "status": "success"
                if not final_answer.startswith("I encountered")
                else "failed"
            },
        )

        if self.config.verbose:
            logger.info("Agent loop complete.")

        return final_answer

    def clear_memory(self):
        self.memory.clear()

    def get_conversation_history(self) -> list[dict[str, Any]]:
        return self.memory.get_messages()

    def get_sources(self) -> list[dict[str, Any]]:
        return self.memory.get_sources()


class AgenticRAGSession:
    def __init__(self, agent: AgenticRAG):
        self.agent = agent

    async def __aenter__(self):
        return self.agent

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.agent.clear_memory()

    async def query(self, question: str) -> str:
        return await self.agent.run(question)
