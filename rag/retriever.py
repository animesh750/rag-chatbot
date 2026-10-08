"""Hybrid retriever: dense (FAISS) + sparse (BM25) search fused with Reciprocal Rank
Fusion, with an optional cross-encoder reranking stage.

Why hybrid?
  * Dense embeddings capture meaning ("why do chatbots answer differently each time?")
    but can miss exact tokens such as names, numbers and acronyms ("GPT3", "Word2Vec").
  * BM25 is the opposite: great at exact terms, blind to paraphrase.
  * RRF merges the two *rankings* (not raw scores, which live on different scales).
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .embeddings import Embedder, Reranker
from .loader import Chunk

Mode = Literal["hybrid", "dense", "bm25"]
MODES: tuple[str, ...] = ("hybrid", "dense", "bm25")

# ---------------------------------------------------------------------------
# Tokenisation for BM25
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its may might more most of on or our so such than that the their then there these
    they this those to was we were what when where which while who whom why will with would you
    your about also not no""".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _stem(tok: str) -> str:
    """Very small suffix stripper so 'parameters' matches 'parameter', 'trained' matches 'train'."""
    if len(tok) > 5 and tok.endswith("ies"):
        return tok[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 4 and not tok.endswith("ss"):
            return tok[: -len(suffix)]
    return tok


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, keep hyphenated words *and* their parts."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        parts = [raw] + (raw.split("-") if "-" in raw else [])
        for p in parts:
            if p not in _STOPWORDS and len(p) > 1:
                tokens.append(_stem(p))
    return tokens


class LuceneBM25(BM25Okapi):
    """BM25 with the Lucene/Elasticsearch IDF: log(1 + (N - n + 0.5) / (n + 0.5)).

    rank_bm25's classic IDF is exactly 0 when a term appears in half the corpus and negative
    when it appears in more than half, so on small documents keyword search silently returns
    nothing. This variant is always positive.
    """

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
        self.average_idf = sum(self.idf.values()) / max(len(self.idf), 1)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class Hit:
    chunk: Chunk
    score: float  # final ordering score (RRF, raw method score, or reranker score)
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["chunk"] = {**asdict(self.chunk), "id": self.chunk.id}
        return d


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int | None = None,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse several ranked lists of ids. score(d) = sum_i w_i / (k + rank_i(d)), rank from 1."""
    k = config.RRF_K if k is None else k
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
@dataclass
class HybridRetriever:
    embedder: Embedder
    reranker: Reranker | None = None
    chunks: list[Chunk] = field(default_factory=list)
    _emb: np.ndarray | None = field(default=None, repr=False)
    _index: faiss.Index | None = field(default=None, repr=False)
    _bm25: LuceneBM25 | None = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ----- indexing ---------------------------------------------------------
    def add_document(self, source: str, chunks: Sequence[Chunk]) -> int:
        """Index chunks for one document. Re-adding a source replaces the old version."""
        if not chunks:
            return 0
        new_emb = self._normalize(self.embedder.encode([c.text for c in chunks]))
        with self._lock:
            self._remove_unlocked(source)
            self.chunks.extend(chunks)
            self._emb = new_emb if self._emb is None else np.vstack([self._emb, new_emb])
            self._rebuild()
        return len(chunks)

    def remove_source(self, source: str) -> int:
        """Drop a document. Uses stored embeddings, so nothing is re-encoded."""
        with self._lock:
            removed = self._remove_unlocked(source)
            self._rebuild()
        return removed

    def _remove_unlocked(self, source: str) -> int:
        if not self.chunks:
            return 0
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self._emb = self._emb[keep] if keep else None
        return removed

    def clear(self) -> None:
        with self._lock:
            self.chunks, self._emb = [], None
            self._rebuild()

    def _rebuild(self) -> None:
        if not self.chunks or self._emb is None:
            self._index = self._bm25 = None
            return
        index = faiss.IndexFlatIP(self._emb.shape[1])  # inner product == cosine (vectors normalised)
        index.add(self._emb)
        self._index = index
        self._bm25 = LuceneBM25([tokenize(c.text) for c in self.chunks])

    @staticmethod
    def _normalize(emb: np.ndarray) -> np.ndarray:
        emb = np.ascontiguousarray(emb, dtype="float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-12)

    # ----- introspection ------------------------------------------------------
    @property
    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.chunks)

    # ----- first-stage retrieval -------------------------------------------------
    def _dense(self, query: str, n: int, allowed: set[int] | None) -> list[tuple[int, float]]:
        q = self._normalize(self.embedder.encode([query]))
        search_n = len(self.chunks) if allowed is not None else min(n, len(self.chunks))
        scores, ids = self._index.search(q, search_n)
        out = [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]
        if allowed is not None:
            out = [(i, s) for i, s in out if i in allowed]
        return out[:n]

    def _sparse(self, query: str, n: int, allowed: set[int] | None) -> list[tuple[int, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        order = np.argsort(-scores, kind="stable")
        out = [(int(i), float(scores[i])) for i in order if scores[i] > 0]  # 0 = no term overlap
        if allowed is not None:
            out = [(i, s) for i, s in out if i in allowed]
        return out[:n]

    def retrieve(
        self,
        query: str,
        mode: Mode = "hybrid",
        fetch_k: int | None = None,
        sources: Iterable[str] | None = None,
        rrf_weights: tuple[float, float] = (1.0, 1.0),
    ) -> list[Hit]:
        """First-stage candidates (up to fetch_k), best first."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        with self._lock:
            if not self.chunks:
                return []
            n = min(fetch_k or config.FETCH_K, len(self.chunks))
            allowed = None
            if sources is not None:
                wanted = set(sources)
                allowed = {i for i, c in enumerate(self.chunks) if c.source in wanted}
                if not allowed:
                    return []

            dense = self._dense(query, n, allowed) if mode in ("hybrid", "dense") else []
            sparse = self._sparse(query, n, allowed) if mode in ("hybrid", "bm25") else []
            d_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
            s_rank = {i: r for r, (i, _) in enumerate(sparse, 1)}
            d_score = dict(dense)
            s_score = dict(sparse)

            if mode == "dense":
                ordered = [(i, s) for i, s in dense]
            elif mode == "bm25":
                ordered = [(i, s) for i, s in sparse]
            else:
                ordered = reciprocal_rank_fusion(
                    [[i for i, _ in dense], [i for i, _ in sparse]], weights=rrf_weights
                )

            return [
                Hit(
                    chunk=self.chunks[i],
                    score=score,
                    dense_rank=d_rank.get(i),
                    dense_score=d_score.get(i),
                    bm25_rank=s_rank.get(i),
                    bm25_score=s_score.get(i),
                )
                for i, score in ordered[:n]
            ]

    # ----- second-stage reranking ---------------------------------------------------
    def rerank(self, query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
        if self.reranker is None:
            raise RuntimeError("No reranker configured")
        if not hits:
            return []
        scores = self.reranker.score(query, [h.chunk.text for h in hits])
        for h, s in zip(hits, scores, strict=True):
            h.rerank_score = float(s)
            h.score = float(s)
        hits = sorted(hits, key=lambda h: -h.score)
        return hits[:top_k] if top_k else hits

    def search(
        self,
        query: str,
        k: int | None = None,
        mode: Mode = "hybrid",
        rerank: bool = False,
        fetch_k: int | None = None,
        sources: Iterable[str] | None = None,
    ) -> list[Hit]:
        """Convenience wrapper: retrieve -> (optional) rerank -> top-k."""
        k = k or config.DEFAULT_K
        hits = self.retrieve(query, mode=mode, fetch_k=max(fetch_k or config.FETCH_K, k), sources=sources)
        if rerank and self.reranker is not None:
            return self.rerank(query, hits, top_k=k)
        return hits[:k]

    # ----- persistence --------------------------------------------------------------
    def save(self, directory: str | os.PathLike) -> None:
        """Write chunks + embeddings (FAISS and BM25 are cheap to rebuild on load)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            meta = [
                {"text": c.text, "source": c.source, "page": c.page, "chunk_index": c.chunk_index}
                for c in self.chunks
            ]
            tmp_json, tmp_npy = d / "chunks.json.tmp", d / "embeddings.tmp.npy"
            tmp_json.write_text(json.dumps(meta), encoding="utf-8")
            np.save(tmp_npy, self._emb if self._emb is not None else np.zeros((0, 1), "float32"))
            os.replace(tmp_json, d / "chunks.json")
            os.replace(tmp_npy, d / "embeddings.npy")

    @classmethod
    def load(
        cls, directory: str | os.PathLike, embedder: Embedder, reranker: Reranker | None = None
    ) -> HybridRetriever:
        r = cls(embedder=embedder, reranker=reranker)
        d = Path(directory)
        if not (d / "chunks.json").exists() or not (d / "embeddings.npy").exists():
            return r
        meta = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        emb = np.load(d / "embeddings.npy")
        if meta and len(meta) == len(emb):
            r.chunks = [Chunk(**m) for m in meta]
            r._emb = r._normalize(emb)
            r._rebuild()
        return r
