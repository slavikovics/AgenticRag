"""
events.py — streaming event types emitted during agent loop execution.
"""

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    ITERATION_START = "iteration_start"  # fired once at session start
    LLM_RESPONSE = "llm_response"  # each LLM completion
    TOOL_CALL = "tool_call"  # before each tool execution
    TOOL_RESULT = "tool_result"  # after each tool execution
    ANSWER = "answer"  # final answer ready
    ERROR = "error"  # unrecoverable error
    COMPLETE = "complete"  # always fires last


class AgentEvent(BaseModel):
    type: EventType
    content: Optional[str] = None
    data: Optional[dict[str, Any]] = None
    timestamp: float = Field(default_factory=time.time)

    class Config:
        use_enum_values = True
