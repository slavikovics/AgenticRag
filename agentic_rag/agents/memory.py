"""
memory.py — conversation history and source tracking.
"""

import json
from datetime import datetime
from typing import Any, Optional


class AgentMessage:
    def __init__(
        self,
        role: str,
        content: str,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


class ConversationMemory:
    """
    Sliding window of recent messages for LLM context.
    Also tracks retrieved sources separately so the caller
    can inspect what documents were used to build the answer.
    """

    def __init__(self, max_messages: int = 20):
        self._messages: list[AgentMessage] = []
        self._sources: list[dict[str, Any]] = []
        self.max_messages = max_messages

    def add_message(
        self,
        role: str,
        content: str,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self._messages.append(
            AgentMessage(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                name=name,
            )
        )
        # Trim to sliding window
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]

    def record_sources(self, tool_name: str, result: str):
        """
        Called after a retrieval tool returns results.
        Parses the JSON result and stores individual source records.
        """
        try:
            docs = json.loads(result)
            if isinstance(docs, list):
                for doc in docs:
                    self._sources.append(
                        {
                            "tool": tool_name,
                            "document": doc.get("document"),
                            "source": doc.get("source"),
                            "score": doc.get("score"),
                            "content_preview": (doc.get("content") or "")[:200],
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
        except (json.JSONDecodeError, AttributeError):
            pass

    def get_messages(self) -> list[dict[str, Any]]:
        return [msg.to_dict() for msg in self._messages]

    def get_sources(self) -> list[dict[str, Any]]:
        return list(self._sources)

    def clear(self):
        self._messages.clear()
        self._sources.clear()
