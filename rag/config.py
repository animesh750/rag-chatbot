"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Minimal .env loader. Real environment variables always win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


load_dotenv()

# --- Models -----------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
# Small, fast model used only to rewrite follow-up questions into standalone ones.
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "openai/gpt-oss-20b")

# --- Chunking (same defaults as the original app so results stay comparable) --
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- Retrieval ----------------------------------------------------------------
DEFAULT_K = int(os.getenv("DEFAULT_K", "4"))
FETCH_K = int(os.getenv("FETCH_K", "20"))  # candidates pulled before fusion / reranking
RRF_K = int(os.getenv("RRF_K", "60"))  # standard constant from the RRF paper
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "hybrid")  # hybrid | dense | bm25
DEFAULT_RERANK = _bool("DEFAULT_RERANK", True)

# --- Generation ---------------------------------------------------------------
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))  # headroom: reasoning models spend tokens thinking
# gpt-oss models are reasoning models; "low" keeps answers fast and cheap. Empty string disables.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")
HISTORY_MESSAGES = int(os.getenv("HISTORY_MESSAGES", "6"))  # = last 3 turns

# --- API ------------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data/index")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
API_KEY = os.getenv("API_KEY", "")  # if set, API requires the X-API-Key header
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
