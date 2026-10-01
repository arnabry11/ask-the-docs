import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.connection import get_engine
from app.db.models import LlmCacheRecord
from app.generation.openrouter_client import OpenRouterError, TextDelta
from app.generation.repository import CachedAnswer, GenerationRepository
from app.generation.service import AnswerService
from app.retrieval.fusion import FusedChunk
from app.retrieval.gate import decide_gate
from app.retrieval.rerank import RerankedChunk
from app.retrieval.service import QueryResult


def retrieved_chunk(score: float = 3.0) -> RerankedChunk:
    return RerankedChunk(
        FusedChunk(
            chunk_id="rails:guide:chunk",
            document_id="rails:guide",
            source="rails",
            version="8.1.4",
            title="Rails Guide",
            section_path=("Rails Guide", "Indexes"),
            source_url="https://example.test/guide#indexes",
            text="An index speeds lookups.",
            token_count=6,
            rrf_score=0.02,
            vector_rank=1,
            keyword_rank=None,
            cosine_distance=0.1,
            fts_rank=None,
        ),
        score,
    )


class FakeRetriever:
    def __init__(self, score: float = 3.0) -> None:
        self.result = retrieved_chunk(score)

    def call(self, question: str, top_k: int) -> QueryResult:
        chunks = (self.result,)
        return QueryResult(" ".join(question.split()), chunks, decide_gate(chunks, 1.5))


class FakeStore:
    def __init__(self) -> None:
        self.cached: CachedAnswer | None = None
        self.saves = 0

    def get_cached(self, key: str) -> CachedAnswer | None:
        return self.cached

    def save(self, answer: CachedAnswer) -> CachedAnswer:
        self.saves += 1
        self.cached = answer
        return answer


class FakeProvider:
    def __init__(self, events: list[TextDelta] | None = None) -> None:
        self.events = events if events is not None else [TextDelta("An index speeds lookups [1].")]
        self.calls = 0
        self.closed = False
        self.error: OpenRouterError | None = None
        self.error_after: OpenRouterError | None = None

    def stream(self, messages: object, **kwargs: object) -> Iterator[TextDelta]:
        self.calls += 1
        try:
            if self.error is not None:
                raise self.error
            yield from self.events
            if self.error_after is not None:
                raise self.error_after
        finally:
            self.closed = True


def service(
    retriever: FakeRetriever | None = None,
    store: FakeStore | None = None,
    provider: FakeProvider | None = None,
    *,
    model: str = "test/model",
) -> AnswerService:
    return AnswerService(
        retriever or FakeRetriever(),  # type: ignore[arg-type]
        store or FakeStore(),  # type: ignore[arg-type]
        provider if provider is not None else FakeProvider(),  # type: ignore[arg-type]
        model=model,
        context_chunks=4,
        context_tokens_per_chunk=300,
        max_output_tokens=256,
    )


def test_successful_answer_is_cached_and_replayed_without_a_second_provider_call() -> None:
    store = FakeStore()
    provider = FakeProvider()
    workflow = service(store=store, provider=provider)

    first = list(workflow.stream("  How  do indexes work? "))
    second = list(workflow.stream("How do indexes work?"))

    assert [event.name for event in first] == [
        "progress",
        "sources",
        "progress",
        "progress",
        "token",
        "progress",
        "done",
    ]
    assert [event.data["stage"] for event in first if event.name == "progress"] == [
        "searching",
        "reviewing",
        "writing",
        "checking_citations",
    ]
    assert first[1].data["sources"][0]["source_url"].endswith("#indexes")
    assert first[4].data == {"text": "An index speeds lookups [1].", "provisional": True}
    assert first[-1].data["status"] == "generated"
    assert first[-1].data["citations"][0]["chunk_id"] == "rails:guide:chunk"
    assert second[-1].data["status"] == "cached"
    assert second[-1].data["cached"] is True
    assert provider.calls == 1
    assert store.saves == 1


def test_gate_and_missing_configuration_never_call_the_provider() -> None:
    provider = FakeProvider()
    gated = list(service(FakeRetriever(score=0.5), provider=provider).stream("Question?"))
    assert gated[-1].data["status"] == "gated"
    assert gated[-1].data["refusal"] is True

    unconfigured = list(service(provider=provider, model="").stream("Question?"))
    assert unconfigured[-1].data["status"] == "not_configured"
    assert provider.calls == 0


def test_cached_answer_remains_available_without_provider_key() -> None:
    store = FakeStore()
    store.cached = CachedAnswer(
        "a" * 64, "An index speeds lookups [1].", [{"marker": 1}], "test/model", "v1"
    )
    workflow = AnswerService(
        FakeRetriever(),  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        None,
        model="test/model",
        context_chunks=4,
        context_tokens_per_chunk=300,
        max_output_tokens=256,
    )

    events = list(workflow.stream("How do indexes work?"))

    assert events[-1].data["status"] == "cached"
    assert events[-1].data["answer"] == store.cached.answer


@pytest.mark.parametrize(
    ("events", "status", "error_code"),
    [
        ([TextDelta("Unsupported [2].")], "invalid_citations", "unknown_citation"),
        ([], "invalid_citations", "empty_answer"),
    ],
)
def test_invalid_output_is_not_cached(
    events: list[TextDelta], status: str, error_code: str
) -> None:
    store = FakeStore()
    result = list(service(store=store, provider=FakeProvider(events)).stream("Question?"))

    assert result[-1].data["status"] == status
    assert result[-1].data["answer"] is None
    assert result[-1].data["error_code"] == error_code
    assert store.saves == 0


def test_provider_error_does_not_cache() -> None:
    store = FakeStore()
    provider = FakeProvider()
    provider.error = OpenRouterError("http_503", 503)

    result = list(service(store=store, provider=provider).stream("Question?"))

    assert result[-1].data["status"] == "provider_error"
    assert result[-1].data["error_code"] == "http_503"
    assert store.saves == 0


def test_midstream_error_keeps_tokens_provisional_and_does_not_cache() -> None:
    store = FakeStore()
    provider = FakeProvider([TextDelta("Unverified answer [1].")])
    provider.error_after = OpenRouterError("stream_error")

    events = list(service(store=store, provider=provider).stream("Question?"))

    assert [event.name for event in events] == [
        "progress",
        "sources",
        "progress",
        "progress",
        "token",
        "done",
    ]
    assert events[-1].data["status"] == "provider_error"
    assert events[-1].data["answer"] is None
    assert store.saves == 0


def test_client_disconnect_closes_provider_stream() -> None:
    provider = FakeProvider()
    stream = service(provider=provider).stream("Question?")

    assert [next(stream).name for _ in range(5)] == [
        "progress",
        "sources",
        "progress",
        "progress",
        "token",
    ]
    stream.close()

    assert provider.closed is True


def test_live_generation_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = Settings(_env_file=None)

    assert settings.openrouter_api_key == ""
    assert settings.llm_model == ""


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1", reason="requires PostgreSQL with generation tables"
)
def test_answer_workflow_caches_real_database_result_without_a_second_call() -> None:
    engine = get_engine()
    model = f"test/{uuid4().hex}"
    provider = FakeProvider()
    workflow = AnswerService(
        FakeRetriever(),  # type: ignore[arg-type]
        GenerationRepository(engine),
        provider,  # type: ignore[arg-type]
        model=model,
        context_chunks=4,
        context_tokens_per_chunk=300,
        max_output_tokens=256,
    )
    try:
        first = list(workflow.stream("How do indexes work?"))
        second = list(workflow.stream("How do indexes work?"))

        assert first[-1].data["status"] == "generated"
        assert second[-1].data["status"] == "cached"
        assert provider.calls == 1
        with Session(engine) as session:
            rows = session.scalars(
                select(LlmCacheRecord).where(LlmCacheRecord.model == model)
            ).all()
        assert len(rows) == 1
        assert rows[0].answer == "An index speeds lookups [1]."
    finally:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM llm_cache WHERE model = :model"), {"model": model})
