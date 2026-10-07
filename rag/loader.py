"""PDF loading and page-aware chunking.

Chunking page by page (instead of on the concatenated document) means every chunk
knows which page it came from, so answers can cite "file.pdf, p.7".
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str  # file name
    page: int  # 1-based page number
    chunk_index: int  # position within the whole document

    @property
    def id(self) -> str:
        return f"{self.source}:p{self.page}:c{self.chunk_index}"


class PDFError(ValueError):
    """Raised when a PDF cannot be read or has no extractable text."""


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for pages that contain text."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # pymupdf raises several different error types
        raise PDFError(f"Could not open PDF: {exc}") from exc
    try:
        pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    finally:
        doc.close()
    return [(n, t) for n, t in pages if t.strip()]


def chunk_pages(
    pages: list[tuple[int, str]],
    source: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks: list[Chunk] = []
    for page_no, text in pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(piece, source, page_no, len(chunks)))
    return chunks


def load_pdf_bytes(pdf_bytes: bytes, source: str, **kwargs) -> list[Chunk]:
    pages = extract_pages(pdf_bytes)
    if not pages:
        raise PDFError(
            f"No extractable text in '{source}'. It may be a scanned PDF (OCR is not supported yet)."
        )
    return chunk_pages(pages, source, **kwargs)


def load_pdf_path(path: str, **kwargs) -> list[Chunk]:
    from pathlib import Path

    p = Path(path)
    return load_pdf_bytes(p.read_bytes(), p.name, **kwargs)
