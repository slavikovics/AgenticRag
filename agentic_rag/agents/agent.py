"""
agent.py — AgenticRAG loop and session context manager.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Optional

from .config import AgentConfig
from .events import AgentEvent, EventType
from .memory import ConversationMemory
from .tools.base import ToolDefinition, ToolType
from .tools.retrieval import GENERAL_COLLECTION, handle_retrieve

log = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """\
You are an AI assistant helping prospective university students in Belarus \
find information about universities, admissions, dormitories, tuition, and student life.

You have access to a knowledge base split into separate collections — \
one per university, plus a general collection with aggregated statistics.

Available collections:
{collections_desc}

Guidelines:
1. For questions about a specific university — identify it and use that collection.
2. For general questions about the Belarusian education system or comparisons — use "{general}".
3. If unsure which university the user means, search "{general}" first, then follow up.
4. You may call retrieve_documents multiple times with different collections.
5. Always cite the source URL when referencing retrieved content.
6. If nothing relevant is found locally, use web_search as a fallback.

Available tools:
{tools_desc}
"""


class AgenticRAG:
    """
    Agentic RAG loop with tool use, event streaming, and per-run memory.

    Memory is scoped to each run() call — the agent singleton is safe
    to reuse across concurrent requests.
    """

    def __init__(
        self,
        llm,
        retriever,
        config: Optional[AgentConfig] = None,
        collection_descriptions: Optional[dict[str, dict]] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.config = config or AgentConfig()
        # collection_descriptions: { slug: { display_name, description } }
        self._collection_descriptions: dict[str, dict] = collection_descriptions or {}
        self.tools: dict[str, ToolDefinition] = {}
        self._register_default_tools()

    # ── Tool registration ─────────────────────────────────────────────────────

    def register_tool(self, tool: ToolDefinition):
        self.tools[tool.name] = tool
        log.info("Registered tool: %s", tool.name)

    def _register_default_tools(self):
        collections = self.retriever.list_collections() if self.retriever else []
        collection_enum = collections if collections else [GENERAL_COLLECTION]

        self.register_tool(
            ToolDefinition(
                name="retrieve_documents",
                description=(
                    "Search a specific university collection or the general collection. "
                    "Choose the collection based on which university the user is asking about. "
                    f"Use '{GENERAL_COLLECTION}' for cross-university or general questions."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant documents",
                        },
                        "collection": {
                            "type": "string",
                            "description": (
                                f"Collection to search. Use university slug (e.g. 'grsu_by') "
                                f"or '{GENERAL_COLLECTION}' for general info."
                            ),
                            "enum": collection_enum,
                            "default": GENERAL_COLLECTION,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 5)",
                            "default": 5,
                        },
                        "alpha": {
                            "type": "number",
                            "description": "0=keyword only, 1=semantic only (default: 0.5)",
                            "default": 0.5,
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["query", "collection"],
                },
                type=ToolType.RETRIEVAL,
                handler=lambda **kw: handle_retrieve(self.retriever, **kw),
            )
        )

    # ── System prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        tools_desc = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools.values()
        )

        collections = self.retriever.list_collections() if self.retriever else []
        lines = []
        for slug in sorted(collections):
            meta = self._collection_descriptions.get(slug, {})
            display = meta.get("display_name", slug)
            desc = meta.get("description", "")
            if desc:
                lines.append(f'- "{slug}" — {display}: {desc}')
            else:
                lines.append(f'- "{slug}" — {display}')

        collections_desc = "\n".join(lines) or "  (no collections indexed yet)"

        return _SYSTEM_PROMPT_TEMPLATE.format(
            collections_desc=collections_desc,
            general=GENERAL_COLLECTION,
            tools_desc=tools_desc,
        )

    # ── Tool execution ────────────────────────────────────────────────────────

    async def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"
        try:
            if self.config.verbose:
                log.info("Tool: %s  args: %s", tool_name, tool_input)
            result = await self.tools[tool_name].handler(**tool_input)
            return str(result)
        except Exception as e:
            log.error("Tool execution failed (%s): %s", tool_name, e)
            return f"Error executing tool '{tool_name}': {e}"

    async def _execute_tools_concurrently(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[tuple[str, str, str]]:
        tasks, meta = [], []
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}
            if name in self.tools:
                tasks.append(self._execute_tool(name, args))
                meta.append((tc_id, name))
            else:
                log.warning("Unknown tool requested: %s", name)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            (tc_id, name, str(r) if not isinstance(r, Exception) else f"Error: {r}")
            for (tc_id, name), r in zip(meta, results)
        ]

    # ── Event helpers ─────────────────────────────────────────────────────────

    async def _emit(
        self,
        on_event: Optional[Callable],
        event_type: EventType,
        content: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
    ):
        if on_event is None:
            return
        event = AgentEvent(type=event_type, content=content, data=data)
        if asyncio.iscoroutinefunction(on_event):
            await on_event(event)
        else:
            on_event(event)

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(
        self,
        user_query: str,
        on_event: Optional[Callable[[AgentEvent], Any]] = None,
    ) -> str:
        """
        Run one agentic turn. Memory is local to this call — safe for
        concurrent use of the same agent instance.
        """
        log.info("Agent run: %s", user_query)

        # Per-run memory — not shared across concurrent calls
        memory = ConversationMemory(max_messages=self.config.memory_size)

        await self._emit(
            on_event,
            EventType.ITERATION_START,
            data={
                "query": user_query,
                "max_iterations": self.config.max_iterations,
            },
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_query},
        ]
        memory.add_message("user", user_query)

        final_answer: Optional[str] = None

        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                for iteration in range(1, self.config.max_iterations + 1):
                    if self.config.verbose:
                        log.info(
                            "Iteration %d/%d", iteration, self.config.max_iterations
                        )

                    response, tool_calls = await self.llm.agentic_complete(
                        messages=messages,
                        tools=[t.to_openai_format() for t in self.tools.values()],
                    )
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
                    memory.add_message("assistant", response, tool_calls=tool_calls)

                    if not tool_calls:
                        final_answer = response
                        break

                    # Emit TOOL_CALL events before execution
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
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

                    tool_results = await self._execute_tools_concurrently(tool_calls)

                    for tc_id, tool_name, result in tool_results:
                        if (
                            self.tools.get(tool_name)
                            and self.tools[tool_name].type == ToolType.RETRIEVAL
                        ):
                            memory.record_sources(tool_name, result)

                        await self._emit(
                            on_event,
                            EventType.TOOL_RESULT,
                            data={
                                "tool_call_id": tc_id,
                                "tool_name": tool_name,
                                "result_preview": result[:200],
                                "result_length": len(result),
                                "success": not result.startswith("Error:"),
                            },
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": result,
                            }
                        )
                        memory.add_message(
                            "tool", result, tool_call_id=tc_id, name=tool_name
                        )

                    # Force synthesis on last iteration
                    if iteration == self.config.max_iterations:
                        log.warning("Max iterations reached — forcing synthesis")
                        synthesis, _ = await self.llm.agentic_complete(
                            messages=messages
                        )
                        final_answer = (
                            synthesis
                            or "Reached iteration limit without a conclusive answer."
                        )
                        await self._emit(
                            on_event,
                            EventType.LLM_RESPONSE,
                            content=final_answer,
                            data={
                                "iteration": iteration + 1,
                                "has_tool_calls": False,
                                "forced_synthesis": True,
                            },
                        )

        except asyncio.TimeoutError:
            log.error("Agent timed out after %.1fs", self.config.timeout_seconds)
            final_answer = "The request timed out. Please try a simpler query."
            await self._emit(on_event, EventType.ERROR, content=final_answer)

        except Exception as e:
            log.error("Agent error: %s", e, exc_info=True)
            final_answer = f"An error occurred while processing your request: {e}"
            await self._emit(on_event, EventType.ERROR, content=final_answer)

        final_answer = final_answer or "Unable to generate a response."

        await self._emit(on_event, EventType.ANSWER, content=final_answer)
        await self._emit(
            on_event,
            EventType.COMPLETE,
            data={
                "status": "error" if final_answer.startswith("An error") else "success",
            },
        )

        # Return sources from this run's memory
        self._last_sources = memory.get_sources()

        log.info("Agent run complete.")
        return final_answer

    def get_sources(self) -> list[dict[str, Any]]:
        """Return sources collected during the last run() call."""
        return getattr(self, "_last_sources", [])


class AgenticRAGSession:
    """Async context manager for single-session use."""

    def __init__(self, agent: AgenticRAG):
        self.agent = agent

    async def __aenter__(self) -> AgenticRAG:
        return self.agent

    async def __aexit__(self, *_):
        pass  # no shared memory to clear

    async def query(
        self,
        question: str,
        on_event: Optional[Callable[[AgentEvent], Any]] = None,
    ) -> str:
        return await self.agent.run(question, on_event=on_event)
