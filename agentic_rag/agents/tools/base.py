from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class ToolType(str, Enum):
    RETRIEVAL = "retrieval"
    EXTERNAL_API = "external_api"
    SQL_QUERY = "sql_query"
    CODE_EXECUTION = "code_execution"
    CUSTOM = "custom"


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    type: ToolType
    handler: Callable

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
