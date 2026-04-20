"""
Pytest configuration and fixtures for Agentic RAG tests.
"""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Set up environment variables for testing."""
    # Set default test values if not already set
    if not os.getenv("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = "test_key_for_ci"
    
    if not os.getenv("QDRANT_URL"):
        os.environ["QDRANT_URL"] = "http://localhost:6333"
    
    if not os.getenv("QDRANT_COLLECTION_NAME"):
        os.environ["QDRANT_COLLECTION_NAME"] = "test_documents"
    
    if not os.getenv("QDRANT_MODE"):
        os.environ["QDRANT_MODE"] = "remote"
    
    yield
    
    # Cleanup if needed
