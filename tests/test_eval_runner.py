from collections.abc import Sequence
from pathlib import Path

import pytest

from app.ingestion.sources import SourceSpec
from app.retrieval.keyword_repository import KeywordMatch
from app.retrieval.rerank import RerankService
from app.retrieval.vector_repository import RetrievedChunk
from evals.dataset import EvalQuestion
from evals.run_retrieval_evals import RetrievalEvaluator
from evals.seed_corpus import CachedCorpusClient


class FakeEmbedder:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def embed_query(self, question: str) -> list[float]:
        self.questions.append(question)
        return [1.0] + [0.0] * 383


def vector_match(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        source="rails",
        version="8.1.4",
        title="Guide",
        section_path=("Guide",),
        source_url=f"https://example.test/{chunk_id}",
        text=f"Passage {chunk_id}",
        token_count=3,
        cosine_distance=0.1,
    )


def keyword_match(chunk_id: str) -> KeywordMatch:
    chunk = vector_match(chunk_id)
    return KeywordMatch(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source=chunk.source,
        version=chunk.version,
        title=chunk.title,
        section_path=chunk.section_path,
        source_url=chunk.source_url,
        text=chunk.text,
        token_count=chunk.token_count,
        fts_rank=0.3,
    )


class FakeVectorSearcher:
    def search(self, vector: list[float], limit: int) -> list[RetrievedChunk]:
        assert limit == 30
        return [vector_match("b"), vector_match("a")]


class FakeKeywordSearcher:
    def search(self, question: str, limit: int) -> list[KeywordMatch]:
        assert (question, limit) == ("How to use indexes?", 30)
        return [keyword_match("a")]


class FakeScorer:
    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        assert question == "How to use indexes?"
        assert len(passages) == 2
        return [0.1, 0.9]


def test_evaluator_reuses_one_embedding_for_all_four_modes() -> None:
    embedder = FakeEmbedder()
    evaluator = RetrievalEvaluator(
        embedder,
        FakeVectorSearcher(),
        FakeKeywordSearcher(),
        RerankService(FakeScorer()),
        vector_limit=30,
        keyword_limit=30,
        rerank_top_n=20,
        rrf_k=60,
        gate_threshold=0.5,
    )
    item = EvalQuestion("Q001", "How to use indexes?", ("a",), ("index", "use"), True, "how_to")

    result = evaluator.call(item)

    assert embedder.questions == [item.question]
    assert result.rankings["vector"] == ("b", "a")
    assert result.rankings["keyword"] == ("a",)
    assert result.rankings["hybrid"] == ("a", "b")
    assert result.rankings["hybrid_rerank"] == ("b", "a")
    assert result.best_rerank_score == 0.9
    assert result.gated is False


def test_cached_corpus_client_reads_only_catalog_urls(tmp_path: Path) -> None:
    spec = SourceSpec(
        document_id="rails:test",
        source="rails",
        version="8.1.4",
        format="markdown",
        fetch_url="https://example.test/test.md",
        source_url="https://example.test/test.html",
        relative_path="rails/test.md",
    )
    (tmp_path / "rails").mkdir()
    (tmp_path / "rails" / "test.md").write_bytes(b"test body")
    client = CachedCorpusClient([spec], tmp_path)

    assert client.fetch(spec.fetch_url) == b"test body"
    with pytest.raises(ValueError, match="Unknown"):
        client.fetch("https://example.test/other.md")
