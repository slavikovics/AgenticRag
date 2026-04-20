"""
Configuration for Agentic RAG system.
"""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for agentic RAG."""
    max_iterations: int = 10
    timeout_seconds: float = 60.0
    verbose: bool = True
    use_streaming: bool = False
    memory_size: int = 20
