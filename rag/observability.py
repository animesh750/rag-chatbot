"""Structured JSON logging + Prometheus metrics + per-stage timing helpers."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram

from . import config

# Request id travels with the logs of one request (set by the API middleware).
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# --- Prometheus metrics (module level, so they register exactly once) ----------
HTTP_REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds",
    "Latency of each RAG pipeline stage",
    ["stage"],
    buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
QUERIES = Counter("rag_queries_total", "RAG queries", ["mode", "rerank"])
LLM_TOKENS = Counter("rag_llm_tokens_total", "Total LLM tokens consumed")
LLM_ERRORS = Counter("rag_llm_errors_total", "Failed LLM calls")
INDEX_CHUNKS = Gauge("rag_indexed_chunks", "Chunks currently indexed")
INDEX_DOCS = Gauge("rag_indexed_documents", "Documents currently indexed")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging() -> None:
    root = logging.getLogger()
    if any(getattr(h, "_rag_json", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler._rag_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(config.LOG_LEVEL.upper())


def get_logger(name: str = "rag") -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"fields": fields})


@contextmanager
def timed(stage: str, timings: dict[str, float]) -> Iterator[None]:
    """Time a block; store milliseconds in `timings` and observe the Prometheus histogram."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[f"{stage}_ms"] = round(elapsed * 1000, 1)
        STAGE_LATENCY.labels(stage=stage).observe(elapsed)
