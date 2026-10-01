from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.ingestion import embed as embed_module
from app.ingestion.embed import FastEmbedClient
from app.main import create_app
from app.retrieval.keyword_repository import KeywordMatch
from app.retrieval.rerank import RerankService
from app.retrieval.service import QueryService, get_query_service
from app.retrieval.vector_repository import RetrievedChunk


class FakeEmbedder:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector if vector is not None else [1.0] + [0.0] * 383
        self.questions: list[str] = []

    def embed_query(self, question: str) -> list[float]:
        self.questions.append(question)
        return self.vector


class FakeVectorSearcher:
    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.results = results if results is not None else []
        self.calls: list[tuple[list[float], int]] = []

    def search(self, vector: list[float], limit: int) -> list[RetrievedChunk]:
        self.calls.append((vector, limit))
        return self.results


class FakeKeywordSearcher:
    def __init__(self, results: list[KeywordMatch] | None = None) -> None:
        self.results = results if results is not None else []
        self.calls: list[tuple[str, int]] = []

    def search(self, question: str, limit: int) -> list[KeywordMatch]:
        self.calls.append((question, limit))
        return self.results


class FakeScorer:
    def __init__(self, score: float = 0.75) -> None:
        self.value = score
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((question, list(passages)))
        return [self.value] * len(passages)


def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="rails:query:abc",
        document_id="rails:query",
        source="rails",
        version="8.1.4",
        title="Active Record Query Interface",
        section_path=("Active Record Query Interface", "Finding Records"),
        source_url="https://guides.rubyonrails.org/v8.1/active_record_querying.html#finding-records",
        text="Use `where` to filter records.",
        token_count=9,
        cosine_distance=0.12,
    )


def sample_keyword_match() -> KeywordMatch:
    chunk = sample_chunk()
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
        fts_rank=0.25,
    )


def test_query_service_embeds_normalized_question_and_returns_sources() -> None:
    embedder = FakeEmbedder()
    vector_searcher = FakeVectorSearcher([sample_chunk()])
    keyword_searcher = FakeKeywordSearcher([sample_keyword_match()])
    scorer = FakeScorer()

    result = QueryService(embedder, vector_searcher, keyword_searcher, RerankService(scorer)).call(
        "  How   do I find records?  ", top_k=3
    )

    assert result.question == "How do I find records?"
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk.chunk_id == sample_chunk().chunk_id
    assert result.chunks[0].chunk.vector_rank == 1
    assert result.chunks[0].chunk.keyword_rank == 1
    assert result.chunks[0].score == 0.75
    assert result.gate.gated is False
    assert embedder.questions == ["How do I find records?"]
    assert vector_searcher.calls == [(embedder.vector, 30)]
    assert keyword_searcher.calls == [("How do I find records?", 30)]
    assert scorer.calls[0][0] == "How do I find records?"


def test_query_service_returns_empty_list_for_empty_corpus() -> None:
    scorer = FakeScorer()
    result = QueryService(
        FakeEmbedder(), FakeVectorSearcher(), FakeKeywordSearcher(), RerankService(scorer)
    ).call("What is an index?")

    assert result.chunks == ()
    assert result.gate.gated is True
    assert result.gate.reason == "no_sources"
    assert scorer.calls == []


@pytest.mark.parametrize(
    ("question", "top_k"),
    [("   ", 5), ("x" * 501, 5), ("valid", 0), ("valid", 21)],
)
def test_query_service_rejects_invalid_requests_before_embedding(question: str, top_k: int) -> None:
    embedder = FakeEmbedder()
    vector_searcher = FakeVectorSearcher()
    keyword_searcher = FakeKeywordSearcher()
    with pytest.raises(ValueError):
        QueryService(embedder, vector_searcher, keyword_searcher, RerankService(FakeScorer())).call(
            question, top_k
        )
    assert embedder.questions == []
    assert vector_searcher.calls == []
    assert keyword_searcher.calls == []


def test_query_service_rejects_invalid_embedding() -> None:
    vector_searcher = FakeVectorSearcher()
    keyword_searcher = FakeKeywordSearcher()
    with pytest.raises(ValueError, match="384"):
        QueryService(
            FakeEmbedder([0.0] * 384),
            vector_searcher,
            keyword_searcher,
            RerankService(FakeScorer()),
        ).call("valid")
    assert vector_searcher.calls == []
    assert keyword_searcher.calls == []


def test_fastembed_adapter_uses_query_embedding_method(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def __init__(self, **kwargs: object) -> None:
            self.questions: list[str] = []

        def query_embed(self, question: str) -> Iterator[list[float]]:
            self.questions.append(question)
            yield [1.0] + [0.0] * 383

    model = FakeModel()
    monkeypatch.setattr(embed_module, "TextEmbedding", lambda **kwargs: model)
    adapter = FastEmbedClient(Path("/tmp/unused-model-cache"))

    assert adapter.embed_query("How do indexes work?") == [1.0] + [0.0] * 383
    assert model.questions == ["How do indexes work?"]


@pytest.fixture
def api_client() -> Iterator[
    tuple[TestClient, FakeEmbedder, FakeVectorSearcher, FakeKeywordSearcher]
]:
    app = create_app()
    embedder = FakeEmbedder()
    vector_searcher = FakeVectorSearcher([sample_chunk()])
    keyword_searcher = FakeKeywordSearcher([sample_keyword_match()])
    app.dependency_overrides[get_query_service] = lambda: QueryService(
        embedder, vector_searcher, keyword_searcher, RerankService(FakeScorer())
    )
    with TestClient(app) as client:
        yield client, embedder, vector_searcher, keyword_searcher
    app.dependency_overrides.clear()


def test_query_api_returns_ranked_chunk_metadata(
    api_client: tuple[TestClient, FakeEmbedder, FakeVectorSearcher, FakeKeywordSearcher],
) -> None:
    client, embedder, vector_searcher, keyword_searcher = api_client

    response = client.post("/query", json={"question": "  finding records  ", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "finding records"
    assert body["retrieval"] == "hybrid"
    assert body["gated"] is False
    assert body["gate_threshold"] is None
    assert body["refusal"] is None
    assert len(body["results"]) == 1
    assert body["results"][0]["section_path"] == [
        "Active Record Query Interface",
        "Finding Records",
    ]
    assert body["results"][0]["source_url"].endswith("#finding-records")
    assert body["results"][0]["cosine_distance"] == 0.12
    assert body["results"][0]["fts_rank"] == 0.25
    assert body["results"][0]["vector_rank"] == 1
    assert body["results"][0]["keyword_rank"] == 1
    assert body["results"][0]["rrf_score"] == pytest.approx(2 / 61)
    assert body["results"][0]["rerank_score"] == 0.75
    assert embedder.questions == ["finding records"]
    assert vector_searcher.calls[0][1] == 30
    assert keyword_searcher.calls == [("finding records", 30)]


def test_query_api_returns_refusal_for_configured_low_score() -> None:
    app = create_app()
    app.dependency_overrides[get_query_service] = lambda: QueryService(
        FakeEmbedder(),
        FakeVectorSearcher([sample_chunk()]),
        FakeKeywordSearcher(),
        RerankService(FakeScorer(0.2)),
        gate_threshold=0.3,
    )
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "index"})
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["gated"] is True
    assert response.json()["gate_reason"] == "below_threshold"
    assert response.json()["gate_threshold"] == 0.3
    assert response.json()["refusal"] == "The available documentation does not cover this question."
    assert len(response.json()["results"]) == 1


@pytest.mark.parametrize(
    "payload",
    [{"question": "   "}, {"question": "x" * 501}, {"question": "valid", "top_k": 21}],
)
def test_query_api_rejects_invalid_input(
    api_client: tuple[TestClient, FakeEmbedder, FakeVectorSearcher, FakeKeywordSearcher],
    payload: dict[str, object],
) -> None:
    client, embedder, _, _ = api_client

    response = client.post("/query", json=payload)

    assert response.status_code == 422
    assert embedder.questions == []
