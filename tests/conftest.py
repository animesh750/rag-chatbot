"""Shared fixtures. Tests never download models or call the network:
* HashEmbedder  - deterministic bag-of-words embeddings (stands in for sentence-transformers)
* KeywordReranker - stands in for the cross-encoder
* FakeGroqClient - mimics the shape of Groq's SDK responses, including streaming
"""

from __future__ import annotations

import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest

from rag.llm import GroqLLM
from rag.loader import Chunk
from rag.pipeline import RAGPipeline
from rag.retriever import HybridRetriever, tokenize

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = ROOT / "docs" / "genai-principles.pdf"


class HashEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                out[row, zlib.crc32(tok.encode()) % self.dim] += 1.0
        return out


class KeywordReranker:
    """Scores a passage by how many query words it contains (deterministic)."""

    def __init__(self, boost: str | None = None):
        self.boost = boost

    def score(self, query, texts):
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) + (100 if self.boost and self.boost in t else 0) for t in texts]


class FakeGroqClient:
    def __init__(
        self,
        replies: dict[str, str] | None = None,
        tokens: int = 42,
        fail_times: int = 0,
        fail_status: int = 429,
    ):
        self.replies = replies or {}
        self.tokens = tokens
        self.fail_times = fail_times
        self.fail_status = fail_status
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.fail_times > 0:
            self.fail_times -= 1
            err = RuntimeError("boom")
            err.status_code = self.fail_status  # type: ignore[attr-defined]
            raise err
        text = self.replies.get(kw["model"], "Fake answer [1].")
        if kw.get("stream"):
            words = text.split(" ")
            chunks = [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=w + " "))], x_groq=None
                )
                for w in words
            ]
            chunks.append(
                SimpleNamespace(
                    choices=[], x_groq=SimpleNamespace(usage=SimpleNamespace(total_tokens=self.tokens))
                )
            )
            return iter(chunks)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(total_tokens=self.tokens),
        )


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(60, 60, 540, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def sample_chunks() -> list[Chunk]:
    texts = [
        ("notes.pdf", 1, "GPT3 is a 175 billion parameter model trained on 570 GB of text data."),
        ("notes.pdf", 1, "Transformers process the whole context window at once instead of word by word."),
        ("notes.pdf", 2, "Sampling the next word from a probability distribution makes completions random."),
        ("notes.pdf", 2, "Word2Vec learned vector representations of words from neighbouring words."),
        (
            "other.pdf",
            1,
            "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.",
        ),
    ]
    return [Chunk(t, s, p, i) for i, (s, p, t) in enumerate(texts)]


@pytest.fixture()
def retriever(sample_chunks) -> HybridRetriever:
    r = HybridRetriever(HashEmbedder(), KeywordReranker())
    r.add_document("notes.pdf", [c for c in sample_chunks if c.source == "notes.pdf"])
    r.add_document("other.pdf", [c for c in sample_chunks if c.source == "other.pdf"])
    return r


@pytest.fixture()
def fake_client() -> FakeGroqClient:
    return FakeGroqClient(replies={"llama-3.1-8b-instant": "How many parameters does GPT3 have?"})


@pytest.fixture()
def pipeline(retriever, fake_client) -> RAGPipeline:
    return RAGPipeline(retriever, GroqLLM(client=fake_client, rewrite_model="llama-3.1-8b-instant"))


@pytest.fixture()
def pdf_bytes() -> bytes:
    return make_pdf(
        [
            "GPT3 is a 175 billion parameter model. It was released in 2020 and trained on 570 GB of text.",
            "Word2Vec was introduced by Mikolov in 2013 and learns word vectors from context.",
        ]
    )
