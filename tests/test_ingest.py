from ingest import ingest
from rag.retriever import HybridRetriever
from tests.conftest import SAMPLE_PDF, HashEmbedder


def test_ingest_builds_a_persistent_index_that_can_be_reloaded(tmp_path):
    ingest([str(SAMPLE_PDF)], str(tmp_path), HashEmbedder())
    reloaded = HybridRetriever.load(tmp_path, HashEmbedder())
    assert reloaded.sources == {"genai-principles.pdf": 55}
    assert "175 billion" in reloaded.search("GPT3 parameters", k=1, mode="bm25")[0].chunk.text


def test_ingest_is_idempotent_for_the_same_file(tmp_path):
    ingest([str(SAMPLE_PDF)], str(tmp_path), HashEmbedder())
    ingest([str(SAMPLE_PDF)], str(tmp_path), HashEmbedder())
    assert len(HybridRetriever.load(tmp_path, HashEmbedder())) == 55
