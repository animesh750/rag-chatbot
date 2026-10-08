"""Make the test run hermetic: a developer's local .env must never change test behaviour.

This runs before any `rag` import, and rag.config only fills variables that are still unset.
"""

import os

os.environ.update(
    LLM_MODEL="openai/gpt-oss-120b",
    REWRITE_MODEL="openai/gpt-oss-20b",
    REASONING_EFFORT="low",
    DEFAULT_MODE="hybrid",
    DEFAULT_RERANK="true",
    DEFAULT_K="4",
    CHUNK_SIZE="500",
    CHUNK_OVERLAP="50",
    API_KEY="",
    GROQ_API_KEY="test-key-not-used",
)
