"""
Agent message and conversation memory for Agentic RAG system.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


class AgentMessage:
    """Message in agent conversation."""

    def __init__(
        self,
        role: str,  # "user", "assistant", "system", "tool"
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
        """Convert to dict for API calls."""
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
    """Simple conversation memory buffer."""

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
        """Add message to memory."""
        self.messages.append(AgentMessage(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            name=name,
        ))

        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get messages in API format."""
        return [msg.to_dict() for msg in self.messages]

    def clear(self):
        """Clear memory."""
        self.messages.clear()

    def get_sources(self) -> List[Dict[str, Any]]:
        """Extract sources from conversation history."""
        sources = []
        for msg in self.messages:
            if msg.role == "tool" and msg.name == "retrieve_documents":
                try:
                    # Parse tool result for sources
                    # This is a simplified extraction
                    sources.append({
                        "tool": msg.name,
                        "content": msg.content[:500],
                    })
                except Exception:
                    pass
        return sources
