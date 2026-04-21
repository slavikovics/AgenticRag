from datetime import datetime
from typing import Any, Dict, List, Optional


class AgentMessage:
    def __init__(
        self,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
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

    def to_dict(self) -> Dict[str, Any]:
        msg = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


class ConversationMemory:
    def __init__(self, max_messages: int = 20):
        self.messages: List[AgentMessage] = []
        self.max_messages = max_messages

    def add_message(
        self,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self.messages.append(
            AgentMessage(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                name=name,
            )
        )

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def get_messages(self) -> List[Dict[str, Any]]:
        return [msg.to_dict() for msg in self.messages]

    def clear(self):
        self.messages.clear()

    def get_sources(self) -> List[Dict[str, Any]]:
        sources = []
        for msg in self.messages:
            if msg.role == "tool" and msg.name == "retrieve_documents":
                try:
                    sources.append(
                        {
                            "tool": msg.name,
                            "content": msg.content[:500],
                        }
                    )
                except Exception:
                    pass
        return sources
