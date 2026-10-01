from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.ingestion import embed as embed_module
from app.ingestion.embed import FastEmbedClient
from app.main import create_app
from app.retrieval.service import QueryService, get_query_service
from app.retrieval.vector_repository import RetrievedChunk


class FakeEmbedder:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector if vector is not None else [1.0] + [0.0] * 383
        self.questions: list[str] = []

    def embed_query(self, question: str) -> list[float]:
        self.questions.append(question)
        return self.vector


class FakeSearcher:
    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.results = results if results is not None else []
        self.calls: list[tuple[list[float], int]] = []

    def search(self, vector: list[float], limit: int) -> list[RetrievedChunk]:
        self.calls.append((vector, limit))
        return self.results


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


def test_query_service_embeds_normalized_question_and_returns_sources() -> None:
    embedder = FakeEmbedder()
    searcher = FakeSearcher([sample_chunk()])

    result = QueryService(embedder, searcher).call("  How   do I find records?  ", top_k=3)

    assert result.question == "How do I find records?"
    assert result.chunks == (sample_chunk(),)
    assert embedder.questions == ["How do I find records?"]
    assert searcher.calls == [(embedder.vector, 3)]


def test_query_service_returns_empty_list_for_empty_corpus() -> None:
    result = QueryService(FakeEmbedder(), FakeSearcher()).call("What is an index?")

    assert result.chunks == ()


@pytest.mark.parametrize(
    ("question", "top_k"),
    [("   ", 5), ("x" * 501, 5), ("valid", 0), ("valid", 21)],
)
def test_query_service_rejects_invalid_requests_before_embedding(question: str, top_k: int) -> None:
    embedder = FakeEmbedder()
    with pytest.raises(ValueError):
        QueryService(embedder, FakeSearcher()).call(question, top_k)
    assert embedder.questions == []


def test_query_service_rejects_invalid_embedding() -> None:
    searcher = FakeSearcher()
    with pytest.raises(ValueError, match="384"):
        QueryService(FakeEmbedder([0.0] * 384), searcher).call("valid")
    assert searcher.calls == []


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
def api_client() -> Iterator[tuple[TestClient, FakeEmbedder, FakeSearcher]]:
    app = create_app()
    embedder = FakeEmbedder()
    searcher = FakeSearcher([sample_chunk()])
    app.dependency_overrides[get_query_service] = lambda: QueryService(embedder, searcher)
    with TestClient(app) as client:
        yield client, embedder, searcher
    app.dependency_overrides.clear()


def test_query_api_returns_ranked_chunk_metadata(
    api_client: tuple[TestClient, FakeEmbedder, FakeSearcher],
) -> None:
    client, embedder, searcher = api_client

    response = client.post("/query", json={"question": "  finding records  ", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "finding records"
    assert body["retrieval"] == "vector"
    assert len(body["results"]) == 1
    assert body["results"][0]["section_path"] == [
        "Active Record Query Interface",
        "Finding Records",
    ]
    assert body["results"][0]["source_url"].endswith("#finding-records")
    assert body["results"][0]["cosine_distance"] == 0.12
    assert embedder.questions == ["finding records"]
    assert searcher.calls[0][1] == 1


@pytest.mark.parametrize(
    "payload",
    [{"question": "   "}, {"question": "x" * 501}, {"question": "valid", "top_k": 21}],
)
def test_query_api_rejects_invalid_input(
    api_client: tuple[TestClient, FakeEmbedder, FakeSearcher], payload: dict[str, object]
) -> None:
    client, embedder, _ = api_client

    response = client.post("/query", json=payload)

    assert response.status_code == 422
    assert embedder.questions == []
