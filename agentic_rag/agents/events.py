from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


class EventType(str, Enum):
    ITERATION_START = "iteration_start"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ANSWER = "answer"
    ERROR = "error"
    COMPLETE = "complete"


class AgentEvent(BaseModel):
    type: EventType
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

    class Config:
        use_enum_values = True
