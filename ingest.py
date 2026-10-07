"""Index PDFs into a persistent hybrid index that the FastAPI service loads on startup.

    python ingest.py docs/genai-principles.pdf another.pdf
    python ingest.py docs/*.pdf --index-dir data/index

(This replaces the old ChromaDB-based script, which the app no longer reads.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag import config
from rag.loader import load_pdf_path
from rag.retriever import HybridRetriever


def ingest(paths: list[str], index_dir: str, embedder) -> HybridRetriever:
    retriever = HybridRetriever.load(index_dir, embedder)
    for path in paths:
        chunks = load_pdf_path(path)
        n = retriever.add_document(Path(path).name, chunks)
        print(f"✓ {Path(path).name}: {n} chunks")
    retriever.save(index_dir)
    print(f"✅ Index at '{index_dir}': {len(retriever.sources)} document(s), {len(retriever)} chunks")
    return retriever


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="*", default=["docs/genai-principles.pdf"])
    ap.add_argument("--index-dir", default=config.DATA_DIR)
    args = ap.parse_args()

    from rag.embeddings import SentenceTransformerEmbedder

    ingest(args.pdfs, args.index_dir, SentenceTransformerEmbedder())


if __name__ == "__main__":
    main()
