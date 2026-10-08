import pytest

from rag.loader import PDFError, chunk_pages, load_pdf_bytes, load_pdf_path
from tests.conftest import SAMPLE_PDF, make_pdf


def test_chunks_carry_page_numbers_and_unique_ids():
    chunks = load_pdf_path(str(SAMPLE_PDF))
    assert {c.page for c in chunks} == set(range(1, 13))
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.source == "genai-principles.pdf" for c in chunks)


def test_chunks_respect_size_limit():
    chunks = load_pdf_path(str(SAMPLE_PDF))
    assert max(len(c.text) for c in chunks) <= 500


def test_chunk_pages_never_mixes_pages():
    chunks = chunk_pages([(1, "alpha " * 200), (2, "beta " * 200)], "x.pdf", chunk_size=100, chunk_overlap=10)
    assert all(("alpha" in c.text) == (c.page == 1) for c in chunks)


def test_garbage_bytes_raise_pdf_error():
    with pytest.raises(PDFError):
        load_pdf_bytes(b"this is not a pdf", "bad.pdf")


def test_pdf_without_text_raises_pdf_error():
    with pytest.raises(PDFError, match="No extractable text"):
        load_pdf_bytes(make_pdf([""]), "blank.pdf")
