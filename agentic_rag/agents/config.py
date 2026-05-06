"""
config.py — agent configuration.
"""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    max_iterations: int = 10
    timeout_seconds: float = 120.0  # total budget for the entire run
    tool_timeout_seconds: float = (
        15.0  # per-tool timeout — prevents one hung tool killing the loop
    )
    verbose: bool = True
    memory_size: int = 20

    # LLM generation params
    temperature: float = 0.0  # 0 = deterministic
    max_tokens: int = 2048

    # Retry config (tenacity)
    llm_retry_attempts: int = 3
    llm_retry_min_wait: float = 1.0
    llm_retry_max_wait: float = 10.0
