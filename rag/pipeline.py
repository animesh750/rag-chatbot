"""Orchestrates one question: rewrite -> retrieve -> (rerank) -> generate, with timings."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from . import config
from .llm import GroqLLM, LLMError, TokenStream, build_answer_messages
from .observability import QUERIES, get_logger, log_event, timed
from .retriever import Hit, HybridRetriever, Mode
from .smalltalk import classify, reply

log = get_logger("rag.pipeline")

NOT_FOUND = "I couldn't find any relevant information in the uploaded documents."


@dataclass
class QueryResult:
    answer: str
    hits: list[Hit]
    standalone_question: str
    timings: dict[str, float] = field(default_factory=dict)
    total_tokens: int = 0
    reranked: bool = False


@dataclass
class PreparedQuery:
    """Everything known before generation starts (used for streaming)."""

    hits: list[Hit]
    standalone_question: str
    messages: list[dict] | None
    timings: dict[str, float]
    tokens: int
    reranked: bool
    direct_answer: str | None = None  # set for small talk: no search, no LLM call


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: GroqLLM):
        self.retriever = retriever
        self.llm = llm

    def prepare(
        self,
        question: str,
        history: Sequence[dict] = (),
        mode: Mode = config.DEFAULT_MODE,  # type: ignore[assignment]
        rerank: bool = config.DEFAULT_RERANK,
        k: int = config.DEFAULT_K,
        sources: Iterable[str] | None = None,
        rewrite: bool = True,
    ) -> PreparedQuery:
        kind = classify(question)
        if kind:
            answer = reply(kind, len(self.retriever.sources))
            return PreparedQuery([], question, None, {}, 0, False, direct_answer=answer)

        timings: dict[str, float] = {}
        tokens = 0

        standalone = question
        if rewrite and history:
            try:
                with timed("rewrite", timings):
                    standalone, t = self.llm.rewrite_question(question, history)
                tokens += t
            except LLMError as exc:  # rewriting is an optimisation; never fail the query for it
                log.warning("query rewrite failed, using original question: %s", exc)
                standalone = question

        with timed("retrieve", timings):
            hits = self.retriever.retrieve(
                standalone, mode=mode, fetch_k=max(config.FETCH_K, k), sources=sources
            )

        reranked = False
        if rerank and self.retriever.reranker is not None and hits:
            try:
                with timed("rerank", timings):
                    hits = self.retriever.rerank(standalone, hits, top_k=k)
                reranked = True
            except Exception as exc:  # e.g. model failed to load on a small host
                log.warning("reranking failed, falling back to first-stage order: %s", exc)
                hits = hits[:k]
        else:
            hits = hits[:k]

        QUERIES.labels(mode=mode, rerank=str(reranked).lower()).inc()
        messages = build_answer_messages(question, hits, history) if hits else None
        return PreparedQuery(hits, standalone, messages, timings, tokens, reranked)

    def query(self, question: str, history: Sequence[dict] = (), **kwargs) -> QueryResult:
        start = time.perf_counter()
        prep = self.prepare(question, history, **kwargs)
        if prep.direct_answer:
            return QueryResult(prep.direct_answer, [], question, {}, 0, False)
        if prep.messages is None:
            prep.timings["total_ms"] = round((time.perf_counter() - start) * 1000, 1)
            return QueryResult(NOT_FOUND, [], prep.standalone_question, prep.timings, prep.tokens, False)

        with timed("generate", prep.timings):
            answer, gen_tokens = self.llm.chat(prep.messages)
        total = prep.tokens + gen_tokens
        prep.timings["total_ms"] = round((time.perf_counter() - start) * 1000, 1)
        log_event(
            log,
            "query_complete",
            mode=kwargs.get("mode", config.DEFAULT_MODE),
            reranked=prep.reranked,
            n_hits=len(prep.hits),
            tokens=total,
            question_chars=len(question),
            **prep.timings,
        )
        return QueryResult(answer, prep.hits, prep.standalone_question, prep.timings, total, prep.reranked)

    def stream(self, prep: PreparedQuery) -> TokenStream:
        """Start streaming the answer for an already prepared query."""
        if prep.messages is None:
            raise ValueError("Nothing to generate: no passages were retrieved")
        return self.llm.stream(prep.messages)
