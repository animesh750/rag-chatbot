"""Measure retrieval quality (and optionally answer quality) for each search configuration.

Retrieval metrics need no API key:
    python -m evaluation.run_eval --modes bm25,dense,hybrid,hybrid+rerank

Answer quality (LLM-as-judge for faithfulness and correctness) needs GROQ_API_KEY:
    python -m evaluation.run_eval --modes hybrid+rerank --judge

A chunk counts as relevant if it contains any of the question's `gold` evidence phrases
(whitespace/case-insensitive). Matching on text instead of chunk ids keeps the evaluation
valid even if you change chunk size or overlap.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np

from rag import config
from rag.loader import load_pdf_path
from rag.retriever import HybridRetriever

DEFAULT_DATASET = Path(__file__).with_name("dataset.jsonl")
KS = (1, 3, 5)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def is_relevant(chunk_text: str, gold: Sequence[str]) -> bool:
    t = _norm(chunk_text)
    return any(_norm(g) in t for g in gold)


def first_relevant_rank(hits, gold: Sequence[str]) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if is_relevant(h.chunk.text, gold):
            return rank
    return None


def parse_config(name: str) -> tuple[str, bool]:
    """'hybrid+rerank' -> ('hybrid', True)"""
    mode, _, suffix = name.partition("+")
    if mode not in ("bm25", "dense", "hybrid") or suffix not in ("", "rerank"):
        raise ValueError(f"Unknown configuration {name!r}")
    return mode, suffix == "rerank"


def evaluate_retrieval(
    retriever: HybridRetriever, items: list[dict], config_name: str, k_max: int = 5
) -> dict:
    mode, rerank = parse_config(config_name)
    ranks: list[int | None] = []
    types: list[str] = []
    latencies: list[float] = []
    for it in items:
        t0 = time.perf_counter()
        hits = retriever.search(it["question"], k=k_max, mode=mode, rerank=rerank)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranks.append(first_relevant_rank(hits, it["gold"]))
        types.append(it["type"])

    def summarize(idx: list[int]) -> dict:
        rs = [ranks[i] for i in idx]
        out = {f"hit@{k}": sum(1 for r in rs if r is not None and r <= k) / len(rs) for k in KS if k <= k_max}
        out[f"mrr@{k_max}"] = sum(1 / r for r in rs if r is not None) / len(rs)
        out["n"] = len(rs)
        return out

    result = {"config": config_name, "overall": summarize(list(range(len(items)))), "by_type": {}}
    for t in sorted(set(types)):
        result["by_type"][t] = summarize([i for i, x in enumerate(types) if x == t])
    result["latency_ms_median"] = round(statistics.median(latencies), 1)
    result["failures"] = [items[i]["id"] for i, r in enumerate(ranks) if r is None]
    return result


# ---------------------------------------------------------------------------
# Optional answer-quality evaluation (LLM as judge)
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are grading a question-answering system. Reply with ONLY a JSON object:
{"faithful": 0 or 1, "correct": 0 or 1}
- faithful = 1 only if every claim in ANSWER is supported by CONTEXT (no outside facts).
- correct = 1 only if ANSWER conveys the same key facts as REFERENCE."""


def judge_answer(llm, question: str, reference: str, answer: str, context: str) -> dict:
    prompt = f"QUESTION: {question}\n\nREFERENCE: {reference}\n\nCONTEXT:\n{context}\n\nANSWER: {answer}"
    text, _ = llm.chat(
        [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    return {"faithful": int(bool(data.get("faithful", 0))), "correct": int(bool(data.get("correct", 0)))}


def evaluate_answers(pipeline, items: list[dict], config_name: str, k: int, sleep: float = 2.0) -> dict:
    mode, rerank = parse_config(config_name)
    faithful, correct = [], []
    for it in items:
        res = pipeline.query(it["question"], mode=mode, rerank=rerank, k=k, rewrite=False)
        context = "\n\n".join(h.chunk.text for h in res.hits)
        verdict = judge_answer(pipeline.llm, it["question"], it["reference"], res.answer, context)
        faithful.append(verdict["faithful"])
        correct.append(verdict["correct"])
        time.sleep(sleep)  # stay inside free-tier rate limits
    n = len(items)
    return {"config": config_name, "faithfulness": sum(faithful) / n, "correctness": sum(correct) / n, "n": n}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def to_markdown(results: list[dict], meta: dict, answer_results: list[dict] | None = None) -> str:
    k_max = meta["k"]
    types = sorted({t for r in results for t in r["by_type"]})
    lines = [
        f"_{meta['n_questions']} questions over `{meta['document']}` · {meta['n_chunks']} chunks · "
        f"chunk size {meta['chunk_size']}/{meta['chunk_overlap']} · embeddings `{meta['embed_model']}` · "
        f"reranker `{meta['rerank_model']}` · run on {meta['date']}_",
        "",
        f"| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@{k_max} | "
        + " | ".join(f"Hit@3 ({t})" for t in types)
        + " | Median latency |",
        "|---|---|---|---|---|" + "---|" * len(types) + "---|",
    ]
    for r in results:
        o = r["overall"]
        row = [
            r["config"],
            f"{o['hit@1']:.0%}",
            f"{o['hit@3']:.0%}",
            f"{o['hit@5']:.0%}",
            f"{o[f'mrr@{k_max}']:.3f}",
            *[f"{r['by_type'][t]['hit@3']:.0%}" for t in types],
            f"{r['latency_ms_median']} ms",
        ]
        lines.append("| " + " | ".join(row) + " |")
    if answer_results:
        lines += ["", "| Configuration | Faithfulness | Correctness |", "|---|---|---|"]
        for a in answer_results:
            lines.append(f"| {a['config']} | {a['faithfulness']:.0%} | {a['correctness']:.0%} |")
    return "\n".join(lines)


class _NullEmbedder:
    """Used when only BM25 is evaluated, so no model download is needed."""

    def encode(self, texts):
        return np.ones((len(texts), 1), dtype="float32")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default="docs/genai-principles.pdf")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--modes", default="bm25,dense,hybrid,hybrid+rerank")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--judge", action="store_true", help="also score generated answers with an LLM judge")
    ap.add_argument("--judge-config", default="hybrid+rerank")
    ap.add_argument("--out", default="evaluation/results", help="path prefix for .md and .json output")
    ap.add_argument(
        "--min-hit3",
        type=float,
        default=None,
        help="exit with status 1 if any configuration's overall Hit@3 is below this (CI gate)",
    )
    args = ap.parse_args(argv)

    configs = [c.strip() for c in args.modes.split(",") if c.strip()]
    parsed = [parse_config(c) for c in configs]
    need_dense = any(m in ("dense", "hybrid") for m, _ in parsed)
    need_rerank = any(r for _, r in parsed)

    items = load_dataset(args.dataset)
    chunks = load_pdf_path(args.pdf)
    print(f"{len(items)} questions, {len(chunks)} chunks from {Path(args.pdf).name}", file=sys.stderr)

    if need_dense:
        from rag.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    else:
        embedder = _NullEmbedder()
    reranker = None
    if need_rerank:
        from rag.embeddings import CrossEncoderReranker

        reranker = CrossEncoderReranker()

    retriever = HybridRetriever(embedder, reranker)
    retriever.add_document(Path(args.pdf).name, chunks)

    results = []
    for name in configs:
        print(f"evaluating {name} ...", file=sys.stderr)
        results.append(evaluate_retrieval(retriever, items, name, k_max=args.k))

    answer_results = None
    if args.judge:
        from rag.llm import GroqLLM
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(retriever, GroqLLM())
        print(f"judging answers for {args.judge_config} ...", file=sys.stderr)
        answer_results = [evaluate_answers(pipeline, items, args.judge_config, k=min(args.k, 4))]

    meta = {
        "n_questions": len(items),
        "document": Path(args.pdf).name,
        "n_chunks": len(chunks),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embed_model": config.EMBED_MODEL if need_dense else "n/a",
        "rerank_model": config.RERANK_MODEL if need_rerank else "n/a",
        "k": args.k,
        "date": date.today().isoformat(),
    }
    md = to_markdown(results, meta, answer_results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "retrieval": results, "answers": answer_results}, indent=2),
        encoding="utf-8",
    )
    print(md)

    if args.min_hit3 is not None:
        below = [r["config"] for r in results if r["overall"]["hit@3"] < args.min_hit3]
        if below:
            print(f"FAILED quality gate (Hit@3 < {args.min_hit3:.0%}): {', '.join(below)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
