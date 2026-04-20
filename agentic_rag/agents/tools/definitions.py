"""
Tool definitions and types for Agentic RAG system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ToolType(Enum):
    """Types of tools available to agent."""
    RETRIEVAL = "retrieval"
    EXTERNAL_API = "external_api"
    SQL_QUERY = "sql_query"
    CODE_EXECUTION = "code_execution"
    CUSTOM = "custom"


@dataclass
class ToolDefinition:
    """Tool definition for agent use."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema
    type: ToolType
    handler: Callable  # Async function to execute tool

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
