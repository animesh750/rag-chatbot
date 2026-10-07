import json

import pytest
from fastapi.testclient import TestClient

from api import create_app
from rag import config


@pytest.fixture()
def client(pipeline):
    app = create_app(pipeline=pipeline, persist=False)
    with TestClient(app) as c:
        yield c


def upload(client, pdf_bytes, name="gpt.pdf"):
    return client.post("/documents", files={"file": (name, pdf_bytes, "application/pdf")})


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        events.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return events


def test_health_reports_index_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 2 and body["reranker_available"] is True
    assert "X-Request-ID" in client.get("/health").headers


def test_upload_list_search_query_delete_flow(client, pdf_bytes):
    r = upload(client, pdf_bytes)
    assert r.status_code == 201 and r.json()["name"] == "gpt.pdf" and r.json()["chunks"] >= 2
    assert "gpt.pdf" in [d["name"] for d in client.get("/documents").json()]

    hits = client.post(
        "/search", json={"question": "Mikolov Word2Vec", "sources": ["gpt.pdf"], "k": 2}
    ).json()
    assert hits and hits[0]["source"] == "gpt.pdf" and hits[0]["page"] == 2

    q = client.post("/query", json={"question": "How many parameters does GPT3 have?", "k": 3})
    body = q.json()
    assert q.status_code == 200 and body["answer"] == "Fake answer [1]."
    assert body["citations"][0]["n"] == 1 and body["total_tokens"] == 42 and body["request_id"]
    assert body["timings_ms"]["total_ms"] >= 0

    assert client.delete("/documents/gpt.pdf").status_code == 204
    assert client.delete("/documents/gpt.pdf").status_code == 404


def test_upload_validation(client, pdf_bytes, monkeypatch):
    assert upload(client, b"hello", "notes.txt").status_code == 415
    assert upload(client, b"not really a pdf", "fake.pdf").status_code == 415
    assert upload(client, b"%PDF-1.4 garbage", "broken.pdf").status_code == 422
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    assert upload(client, pdf_bytes).status_code == 413


def test_filename_is_sanitised(client, pdf_bytes):
    r = upload(client, pdf_bytes, "../../etc/pass wd?.pdf")
    assert r.status_code == 201 and "/" not in r.json()["name"] and ".." not in r.json()["name"]


def test_query_validation_errors(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "x", "mode": "magic"}).status_code == 422
    assert client.post("/query", json={"question": "x", "k": 0}).status_code == 422


def test_stream_endpoint_emits_meta_tokens_done(client):
    r = client.post("/query/stream", json={"question": "GPT3 parameters", "rerank": False, "k": 2})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "done" and kinds.count("token") > 1
    assert events[0][1]["citations"][0]["source"] == "notes.pdf"
    assert "".join(d["text"] for k, d in events if k == "token").strip() == "Fake answer [1]."
    assert events[-1][1]["total_tokens"] == 42


def test_stream_with_no_matches_returns_not_found_event(pipeline):
    pipeline.retriever.clear()
    with TestClient(create_app(pipeline=pipeline, persist=False)) as c:
        events = parse_sse(c.post("/query/stream", json={"question": "anything"}).text)
    assert events[1][1]["text"].startswith("I couldn't find")


def test_llm_failure_becomes_502(retriever):
    from rag.llm import GroqLLM
    from rag.pipeline import RAGPipeline
    from tests.conftest import FakeGroqClient

    pipe = RAGPipeline(retriever, GroqLLM(client=FakeGroqClient(fail_times=9, fail_status=400)))
    with TestClient(create_app(pipeline=pipe, persist=False)) as c:
        assert c.post("/query", json={"question": "GPT3 parameters"}).status_code == 502


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "s3cret")
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays open for load balancers


def test_metrics_endpoint_exposes_prometheus_series(client):
    client.post("/query", json={"question": "GPT3 parameters"})
    text = client.get("/metrics").text
    assert 'rag_http_requests_total{method="POST",route="/query",status="200"}' in text
    assert "rag_stage_latency_seconds_bucket" in text and "rag_llm_tokens_total" in text
    assert "rag_indexed_chunks" in text


def test_uploaded_documents_survive_a_server_restart(tmp_path, monkeypatch, pdf_bytes, fake_client):
    import api as api_module
    from rag.llm import GroqLLM
    from rag.pipeline import RAGPipeline
    from rag.retriever import HybridRetriever
    from tests.conftest import HashEmbedder, KeywordReranker

    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))

    def build():
        retriever = HybridRetriever.load(config.DATA_DIR, HashEmbedder(), KeywordReranker())
        return RAGPipeline(retriever, GroqLLM(client=fake_client))

    monkeypatch.setattr(api_module, "build_default_pipeline", build)

    with TestClient(create_app(persist=True)) as first:
        assert first.get("/health").json()["documents"] == 0
        assert upload(first, pdf_bytes, "persist.pdf").status_code == 201

    with TestClient(create_app(persist=True)) as second:  # "restart": brand-new app + retriever
        assert second.get("/health").json()["documents"] == 1
        hits = second.post("/search", json={"question": "Mikolov Word2Vec", "mode": "bm25"}).json()
        assert hits and hits[0]["source"] == "persist.pdf"
