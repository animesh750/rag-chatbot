import json

import pytest

from evaluation.run_eval import (
    DEFAULT_DATASET,
    evaluate_retrieval,
    first_relevant_rank,
    is_relevant,
    judge_answer,
    load_dataset,
    parse_config,
    to_markdown,
)
from rag.loader import Chunk, load_pdf_path
from rag.retriever import Hit, HybridRetriever
from tests.conftest import SAMPLE_PDF, FakeGroqClient, HashEmbedder, KeywordReranker


def test_dataset_is_well_formed():
    items = load_dataset()
    assert len(items) >= 30 and len({i["id"] for i in items}) == len(items)
    assert {i["type"] for i in items} == {"lexical", "paraphrase"}
    assert all(i["gold"] and i["reference"] and i["question"].endswith("?") for i in items)


def test_every_gold_phrase_exists_in_the_sample_pdf_chunks():
    """Guards the benchmark: if chunking changes and a gold phrase gets split, we want to know."""
    chunks = load_pdf_path(str(SAMPLE_PDF))
    missing = [
        i["id"]
        for i in load_dataset(DEFAULT_DATASET)
        if not any(is_relevant(c.text, i["gold"]) for c in chunks)
    ]
    assert missing == []


def test_relevance_matching_ignores_case_and_whitespace():
    assert is_relevant("Foo   BAR\nbaz", ["foo bar baz"])
    assert not is_relevant("nothing here", ["foo"])


def test_first_relevant_rank():
    hits = [Hit(Chunk(t, "s", 1, i), 0.0) for i, t in enumerate(["a", "b gold b", "gold"])]
    assert first_relevant_rank(hits, ["gold"]) == 2
    assert first_relevant_rank(hits, ["absent"]) is None


def test_parse_config():
    assert parse_config("hybrid+rerank") == ("hybrid", True)
    assert parse_config("bm25") == ("bm25", False)
    for bad in ("magic", "hybrid+foo", "dense+rerank+x"):
        with pytest.raises(ValueError):
            parse_config(bad)


def test_metrics_are_computed_correctly(sample_chunks):
    r = HybridRetriever(HashEmbedder(), KeywordReranker())
    r.add_document("notes.pdf", sample_chunks[:4])
    items = [
        {"id": "a", "question": "GPT3 parameters", "type": "lexical", "gold": ["175 billion"]},
        {"id": "b", "question": "Word2Vec", "type": "lexical", "gold": ["neighbouring words"]},
        {
            "id": "c",
            "question": "zzzz unknown",
            "type": "paraphrase",
            "gold": ["175 billion"],
        },  # unanswerable
    ]
    res = evaluate_retrieval(r, items, "bm25", k_max=3)
    assert res["overall"]["hit@1"] == pytest.approx(2 / 3)
    assert res["overall"]["mrr@3"] == pytest.approx(2 / 3)
    assert res["by_type"]["lexical"]["hit@1"] == 1.0 and res["by_type"]["paraphrase"]["hit@1"] == 0.0
    assert res["failures"] == ["c"]


def test_judge_parses_json_even_with_chatter():
    llm = __import__("rag.llm", fromlist=["GroqLLM"]).GroqLLM(
        client=FakeGroqClient(replies={"llama-3.3-70b-versatile": 'Sure! {"faithful": 1, "correct": 0}'})
    )
    assert judge_answer(llm, "q", "ref", "ans", "ctx") == {"faithful": 1, "correct": 0}
    junk = __import__("rag.llm", fromlist=["GroqLLM"]).GroqLLM(
        client=FakeGroqClient(replies={"llama-3.3-70b-versatile": "no json at all"})
    )
    assert judge_answer(junk, "q", "ref", "ans", "ctx") == {"faithful": 0, "correct": 0}


def test_markdown_report_has_one_row_per_config():
    cfg = {"hit@1": 0.5, "hit@3": 0.75, "hit@5": 1.0, "mrr@5": 0.6, "n": 4}
    results = [
        {
            "config": c,
            "overall": cfg,
            "by_type": {"lexical": cfg, "paraphrase": cfg},
            "latency_ms_median": 1.0,
            "failures": [],
        }
        for c in ("bm25", "hybrid")
    ]
    meta = {
        "n_questions": 4,
        "document": "d.pdf",
        "n_chunks": 9,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "embed_model": "m",
        "rerank_model": "r",
        "k": 5,
        "date": "2026-01-01",
    }
    md = to_markdown(results, meta)
    assert md.count("\n| bm25 |") == 1 and md.count("\n| hybrid |") == 1 and "75%" in md
    json.dumps(results)  # must be serialisable for results.json


def test_cli_quality_gate_passes_and_fails(tmp_path, capsys):
    from evaluation.run_eval import main

    base = ["--modes", "bm25", "--out", str(tmp_path / "res")]
    assert main(base + ["--min-hit3", "0.10"]) == 0
    assert (tmp_path / "res.md").exists() and (tmp_path / "res.json").exists()
    assert main(base + ["--min-hit3", "1.01"]) == 1
    assert "FAILED quality gate" in capsys.readouterr().err
