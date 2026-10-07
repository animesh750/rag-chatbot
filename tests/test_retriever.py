import pytest

from rag.loader import Chunk
from rag.retriever import HybridRetriever, reciprocal_rank_fusion, tokenize
from tests.conftest import HashEmbedder, KeywordReranker


def test_tokenize_keeps_hyphenated_words_and_parts_and_drops_stopwords():
    toks = tokenize("The in-context learning of Word2Vec parameters")
    assert "in-context" in toks and "context" in toks
    assert "word2vec" in toks
    assert "the" not in toks and "of" not in toks
    assert "parameter" in toks  # plural stripped so it matches 'parameters' in documents


def test_rrf_rewards_agreement_between_rankers():
    fused = dict(reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60))
    assert fused[1] > fused[3] > fused[2] > fused[4]
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_weights_shift_the_winner():
    plain = reciprocal_rank_fusion([[1, 2], [2, 1]], k=60)
    weighted = reciprocal_rank_fusion([[1, 2], [2, 1]], k=60, weights=[1.0, 3.0])
    assert plain[0][0] == 1  # tie broken by id
    assert weighted[0][0] == 2


def test_bm25_finds_exact_identifier(retriever):
    hits = retriever.search("GPT3 parameters", k=2, mode="bm25")
    assert "175 billion" in hits[0].chunk.text
    assert hits[0].bm25_rank == 1 and hits[0].dense_rank is None


def test_dense_mode_only_populates_dense_fields(retriever):
    hits = retriever.search("random sampling of the next word", k=2, mode="dense")
    assert hits and all(h.dense_rank is not None and h.bm25_rank is None for h in hits)
    assert "Sampling" in hits[0].chunk.text


def test_hybrid_combines_both_signals(retriever):
    hits = retriever.search("Word2Vec word vectors", k=3, mode="hybrid")
    top = hits[0]
    assert "Word2Vec" in top.chunk.text
    assert top.dense_rank is not None and top.bm25_rank is not None


def test_bm25_returns_nothing_when_no_term_overlaps(retriever):
    assert retriever.search("zzzzqqqq", k=3, mode="bm25") == []


def test_source_filter_restricts_results(retriever):
    hits = retriever.search("model", k=5, mode="hybrid", sources=["other.pdf"])
    assert hits and all(h.chunk.source == "other.pdf" for h in hits)
    assert retriever.search("model", mode="hybrid", sources=["missing.pdf"]) == []


def test_rerank_can_reorder_first_stage_results(sample_chunks):
    r = HybridRetriever(HashEmbedder(), KeywordReranker(boost="Photosynthesis"))
    r.add_document("notes.pdf", sample_chunks[:4])
    r.add_document("other.pdf", sample_chunks[4:])
    plain = r.search("what is a model", k=3, mode="hybrid", rerank=False)
    reranked = r.search("what is a model", k=3, mode="hybrid", rerank=True)
    assert "Photosynthesis" not in plain[0].chunk.text
    assert "Photosynthesis" in reranked[0].chunk.text
    assert reranked[0].rerank_score is not None


def test_rerank_without_reranker_is_skipped_in_search(sample_chunks):
    r = HybridRetriever(HashEmbedder(), reranker=None)
    r.add_document("notes.pdf", sample_chunks[:4])
    hits = r.search("GPT3", k=2, rerank=True)
    assert hits and hits[0].rerank_score is None


def test_invalid_mode_raises(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="semantic")  # type: ignore[arg-type]


def test_empty_retriever_returns_empty_list():
    assert HybridRetriever(HashEmbedder()).search("anything") == []


def test_remove_source_does_not_reencode(sample_chunks):
    emb = HashEmbedder()
    r = HybridRetriever(emb)
    r.add_document("notes.pdf", sample_chunks[:4])
    r.add_document("other.pdf", sample_chunks[4:])
    calls_before = emb.calls
    assert r.remove_source("other.pdf") == 1
    assert emb.calls == calls_before
    assert r.sources == {"notes.pdf": 4}
    assert r.remove_source("other.pdf") == 0


def test_removing_last_document_empties_index(retriever):
    retriever.remove_source("notes.pdf")
    retriever.remove_source("other.pdf")
    assert len(retriever) == 0 and retriever.search("GPT3") == []


def test_re_adding_same_source_replaces_instead_of_duplicating(retriever, sample_chunks):
    before = len(retriever)
    retriever.add_document("other.pdf", [Chunk("Brand new text about ribosomes.", "other.pdf", 1, 0)])
    assert len(retriever) == before  # 1 old chunk swapped for 1 new
    assert retriever.search("ribosomes", mode="bm25")[0].chunk.source == "other.pdf"
    assert retriever.search("photosynthesis", mode="bm25") == []


def test_save_and_load_roundtrip(retriever, tmp_path):
    retriever.save(tmp_path)
    loaded = HybridRetriever.load(tmp_path, HashEmbedder())
    assert len(loaded) == len(retriever) and loaded.sources == retriever.sources
    a = retriever.search("GPT3 parameters", k=3, mode="hybrid")
    b = loaded.search("GPT3 parameters", k=3, mode="hybrid")
    assert [h.chunk.id for h in a] == [h.chunk.id for h in b]


def test_load_from_empty_directory_gives_empty_retriever(tmp_path):
    assert len(HybridRetriever.load(tmp_path / "nope", HashEmbedder())) == 0


@pytest.mark.parametrize("n_chunks", [1, 2, 3])
def test_bm25_works_on_tiny_corpora(n_chunks):
    """Classic BM25 IDF is 0 when a term is in half the corpus, which used to silence keyword search."""
    texts = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"][:n_chunks]
    r = HybridRetriever(HashEmbedder())
    r.add_document("d.pdf", [Chunk(t, "d.pdf", 1, i) for i, t in enumerate(texts)])
    hits = r.search("alpha", k=3, mode="bm25")
    assert hits and hits[0].chunk.text == "alpha beta gamma"


def test_bm25_scores_are_positive_even_for_terms_in_every_chunk():
    r = HybridRetriever(HashEmbedder())
    r.add_document("d.pdf", [Chunk(f"common word {w}", "d.pdf", 1, i) for i, w in enumerate("abc")])
    hits = r.search("common", k=3, mode="bm25")
    assert len(hits) == 3 and all(h.score > 0 for h in hits)
