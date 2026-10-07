"""FastAPI backend for the RAG chatbot.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from rag import __version__, config
from rag.llm import GroqLLM, LLMError
from rag.loader import PDFError, load_pdf_bytes
from rag.observability import (
    HTTP_LATENCY,
    HTTP_REQUESTS,
    INDEX_CHUNKS,
    INDEX_DOCS,
    get_logger,
    log_event,
    request_id_var,
    setup_logging,
)
from rag.pipeline import NOT_FOUND, RAGPipeline

log = get_logger("rag.api")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)
    k: int = Field(default=config.DEFAULT_K, ge=1, le=20)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    rerank: bool = config.DEFAULT_RERANK
    sources: list[str] | None = Field(default=None, description="Restrict search to these document names")
    rewrite: bool = Field(default=True, description="Rewrite follow-up questions into standalone queries")


class Citation(BaseModel):
    n: int
    source: str
    page: int
    score: float
    text: str
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    standalone_question: str
    citations: list[Citation]
    timings_ms: dict[str, float]
    total_tokens: int
    reranked: bool
    request_id: str


class DocumentInfo(BaseModel):
    name: str
    chunks: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _citations(hits) -> list[Citation]:
    return [
        Citation(
            n=i,
            source=h.chunk.source,
            page=h.chunk.page,
            score=round(h.score, 5),
            text=h.chunk.text,
            dense_rank=h.dense_rank,
            bm25_rank=h.bm25_rank,
            rerank_score=None if h.rerank_score is None else round(h.rerank_score, 4),
        )
        for i, h in enumerate(hits, start=1)
    ]


def _safe_name(filename: str | None) -> str:
    name = os.path.basename((filename or "").replace("\\", "/"))
    name = re.sub(r"[^\w.\- ]", "_", name).strip()
    return name or "document.pdf"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def build_default_pipeline() -> RAGPipeline:
    from rag.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
    from rag.retriever import HybridRetriever

    embedder, reranker = SentenceTransformerEmbedder(), CrossEncoderReranker()
    # Load weights now so the first request isn't slow and a missing model fails fast at startup.
    embedder._load()
    reranker._load()
    retriever = HybridRetriever.load(config.DATA_DIR, embedder, reranker)
    return RAGPipeline(retriever, GroqLLM())


def _refresh_index_gauges(pipeline: RAGPipeline) -> None:
    INDEX_CHUNKS.set(len(pipeline.retriever))
    INDEX_DOCS.set(len(pipeline.retriever.sources))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(pipeline: RAGPipeline | None = None, persist: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        if getattr(app.state, "pipeline", None) is None:
            app.state.pipeline = build_default_pipeline()
        _refresh_index_gauges(app.state.pipeline)
        log_event(log, "startup", version=__version__, chunks=len(app.state.pipeline.retriever))
        yield

    app = FastAPI(
        title="RAG Chatbot API",
        version=__version__,
        description="Hybrid (BM25 + dense) retrieval with cross-encoder reranking over your PDFs.",
        lifespan=lifespan,
    )
    app.state.pipeline = pipeline
    app.state.persist = persist

    def get_pipeline(request: Request) -> RAGPipeline:
        return request.app.state.pipeline

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = config.API_KEY
        if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    # ----- middleware: request id, access log, Prometheus ------------------------
    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            elapsed = time.perf_counter() - start
            route = getattr(request.scope.get("route"), "path", "unmatched")  # template, not raw path
            if route != "/metrics":
                HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
                HTTP_LATENCY.labels(request.method, route).observe(elapsed)
                log_event(
                    log,
                    "http_request",
                    method=request.method,
                    route=route,
                    status=status,
                    duration_ms=round(elapsed * 1000, 1),
                )
            request_id_var.reset(token)

    @app.get("/", include_in_schema=False)
    def frontend():
        return FileResponse(Path(__file__).parent / "frontend" / "index.html")

    # ----- health & metrics ----------------------------------------------------------
    @app.get("/health", tags=["ops"])
    def health(p: RAGPipeline = Depends(get_pipeline)):
        return {
            "status": "ok",
            "version": __version__,
            "documents": len(p.retriever.sources),
            "chunks": len(p.retriever),
            "reranker_available": p.retriever.reranker is not None,
        }

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ----- documents --------------------------------------------------------------------
    @app.get(
        "/documents",
        response_model=list[DocumentInfo],
        tags=["documents"],
        dependencies=[Depends(require_api_key)],
    )
    def list_documents(p: RAGPipeline = Depends(get_pipeline)):
        return [DocumentInfo(name=n, chunks=c) for n, c in p.retriever.sources.items()]

    @app.post(
        "/documents",
        response_model=DocumentInfo,
        status_code=201,
        tags=["documents"],
        dependencies=[Depends(require_api_key)],
    )
    async def upload_document(
        request: Request, file: UploadFile = File(...), p: RAGPipeline = Depends(get_pipeline)
    ):
        name = _safe_name(file.filename)
        if not name.lower().endswith(".pdf"):
            raise HTTPException(415, "Only .pdf files are supported")
        limit = config.MAX_UPLOAD_MB * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_MB} MB limit")
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, "File content is not a valid PDF")
        try:
            chunks = load_pdf_bytes(data, name)
        except PDFError as exc:
            raise HTTPException(422, str(exc)) from exc
        n = p.retriever.add_document(name, chunks)  # replaces a previous upload of the same name
        _persist(request, p)
        log_event(log, "document_indexed", name=name, chunks=n, bytes=len(data))
        return DocumentInfo(name=name, chunks=n)

    @app.delete(
        "/documents/{name}", status_code=204, tags=["documents"], dependencies=[Depends(require_api_key)]
    )
    def delete_document(name: str, request: Request, p: RAGPipeline = Depends(get_pipeline)):
        if p.retriever.remove_source(name) == 0:
            raise HTTPException(404, f"No document named {name!r}")
        _persist(request, p)
        return Response(status_code=204)

    def _persist(request: Request, p: RAGPipeline) -> None:
        _refresh_index_gauges(p)
        if request.app.state.persist:
            p.retriever.save(config.DATA_DIR)

    # ----- search (retrieval only: no LLM key needed) ---------------------------------------
    @app.post(
        "/search", response_model=list[Citation], tags=["query"], dependencies=[Depends(require_api_key)]
    )
    def search(body: QueryRequest, p: RAGPipeline = Depends(get_pipeline)):
        hits = p.retriever.search(
            body.question, k=body.k, mode=body.mode, rerank=body.rerank, sources=body.sources
        )
        return _citations(hits)

    # ----- question answering ---------------------------------------------------------------------
    def _prepare(body: QueryRequest, p: RAGPipeline):
        history = [m.model_dump() for m in body.history]
        return history, p.prepare(
            body.question,
            history,
            mode=body.mode,
            rerank=body.rerank,
            k=body.k,
            sources=body.sources,
            rewrite=body.rewrite,
        )

    @app.post("/query", response_model=QueryResponse, tags=["query"], dependencies=[Depends(require_api_key)])
    def query(body: QueryRequest, p: RAGPipeline = Depends(get_pipeline)):
        try:
            res = p.query(
                body.question,
                [m.model_dump() for m in body.history],
                mode=body.mode,
                rerank=body.rerank,
                k=body.k,
                sources=body.sources,
                rewrite=body.rewrite,
            )
        except LLMError as exc:
            raise HTTPException(502, str(exc)) from exc
        return QueryResponse(
            answer=res.answer,
            standalone_question=res.standalone_question,
            citations=_citations(res.hits),
            timings_ms=res.timings,
            total_tokens=res.total_tokens,
            reranked=res.reranked,
            request_id=request_id_var.get(),
        )

    @app.post(
        "/query/stream",
        tags=["query"],
        dependencies=[Depends(require_api_key)],
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    def query_stream(body: QueryRequest, p: RAGPipeline = Depends(get_pipeline)):
        """Server-Sent Events: `meta` (citations), many `token`s, then `done`."""
        rid = request_id_var.get()
        try:
            _, prep = _prepare(body, p)
            stream = None if prep.messages is None else p.stream(prep)
        except LLMError as exc:
            raise HTTPException(502, str(exc)) from exc

        def events() -> Iterator[str]:
            yield _sse(
                "meta",
                {
                    "request_id": rid,
                    "standalone_question": prep.standalone_question,
                    "citations": [c.model_dump() for c in _citations(prep.hits)],
                    "reranked": prep.reranked,
                },
            )
            if stream is None:
                yield _sse("token", {"text": NOT_FOUND})
                yield _sse("done", {"total_tokens": 0, "timings_ms": prep.timings})
                return
            start = time.perf_counter()
            try:
                for piece in stream:
                    yield _sse("token", {"text": piece})
            except Exception as exc:  # the HTTP status is already sent, so report in-band
                log.exception("stream failed")
                yield _sse("error", {"detail": str(exc)})
                return
            prep.timings["generate_ms"] = round((time.perf_counter() - start) * 1000, 1)
            yield _sse(
                "done", {"total_tokens": prep.tokens + stream.total_tokens, "timings_ms": prep.timings}
            )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


# Module-level app for `uvicorn api:app`. Models load in the lifespan hook, not at import.
app = create_app()
