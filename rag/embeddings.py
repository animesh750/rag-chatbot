"""Embedding and reranking model wrappers.

Heavy libraries (torch / sentence-transformers) are imported lazily, so importing
this package, running the unit tests or starting CI does not need them installed.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from . import config


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim)."""


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return one relevance score per text (higher = more relevant)."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBED_MODEL
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 1), dtype="float32")
        emb = self._load().encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype="float32")


class CrossEncoderReranker:
    """Cross-encoder that scores (query, passage) pairs jointly.

    More accurate than comparing embeddings because query and passage attend to each
    other, but slower, so it is only run on the top candidates from first-stage search.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.RERANK_MODEL
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self._load().predict(pairs, show_progress_bar=False)]
