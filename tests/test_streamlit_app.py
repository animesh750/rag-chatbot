"""Runs the real Streamlit script headlessly with fake models (no downloads, no network)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from rag.llm import GroqLLM  # noqa: E402
from tests.conftest import FakeGroqClient, HashEmbedder, KeywordReranker  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def patched(monkeypatch, retriever):
    import rag.embeddings as emb

    monkeypatch.setattr(emb, "SentenceTransformerEmbedder", HashEmbedder)
    monkeypatch.setattr(emb, "CrossEncoderReranker", KeywordReranker)
    client = FakeGroqClient(replies={"openai/gpt-oss-20b": "How many parameters does GPT3 have?"})
    monkeypatch.setattr(GroqLLM, "_get_client", lambda self: client)
    return client


def make_app(retriever):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["retriever"] = retriever
    at.session_state["uploaded_docs"] = dict(retriever.sources)
    return at


def test_empty_state_shows_upload_prompt(patched):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Upload one or more PDFs" in i.value for i in at.info)


def test_full_chat_turn_with_streaming_and_sources(patched, retriever):
    at = make_app(retriever).run()
    assert not at.exception
    at.chat_input[0].set_value("How many parameters does GPT3 have?").run()
    assert not at.exception
    roles = [m.name for m in at.chat_message]
    assert roles.count("user") == 1 and roles.count("assistant") >= 1
    msgs = at.session_state["messages"]
    assert msgs[-1]["content"].strip() == "Fake answer [1]."
    assert msgs[-1]["sources"] and msgs[-1]["meta"]["mode"] == "hybrid"
    assert at.session_state["total_tokens"] == 42


def test_follow_up_question_is_rewritten(patched, retriever):
    at = make_app(retriever).run()
    at.chat_input[0].set_value("Tell me about GPT3").run()
    at.chat_input[0].set_value("how many parameters does it have?").run()
    assert not at.exception
    meta = at.session_state["messages"][-1]["meta"]
    assert meta["standalone"] == "How many parameters does GPT3 have?"


def test_deleting_a_document_removes_it_from_the_index(patched, retriever):
    at = make_app(retriever).run()
    next(b for b in at.button if b.key == "del_other.pdf").click().run()
    assert not at.exception
    assert "other.pdf" not in at.session_state["uploaded_docs"]
    assert "other.pdf" not in retriever.sources
    assert at.session_state["uploader_key"] == 1  # widget reset so the file isn't silently re-indexed


def test_greeting_in_the_ui_is_not_reported_as_not_found(patched, retriever):
    at = make_app(retriever).run()
    at.chat_input[0].set_value("hellow").run()
    assert not at.exception
    answer = at.session_state["messages"][-1]["content"]
    assert "ready to answer" in answer and "couldn't find" not in answer
    assert patched.calls == []
