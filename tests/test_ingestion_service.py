import os
from uuid import uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.connection import get_engine
from app.db.models import ChunkRecord, IngestionAttemptRecord
from app.ingestion.chunk import Chunk
from app.ingestion.parse import ParsedDocument
from app.ingestion.repository import AttemptStatus, IngestionRepository
from app.ingestion.service import IngestDocument
from app.ingestion.sources import SourceSpec


class FakeClient:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.calls = 0

    def fetch(self, url: str) -> bytes:
        self.calls += 1
        return self.content


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.0] * 383 + [1.0] for _ in texts]


class FakeStore:
    def __init__(self) -> None:
        self.fingerprint: str | None = None
        self.saved: tuple[ParsedDocument, list[Chunk]] | None = None
        self.attempts: list[tuple[str, AttemptStatus, str | None]] = []

    def current_fingerprint(self, document_id: str) -> str | None:
        return self.fingerprint

    def replace_document(
        self,
        document: ParsedDocument,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        fingerprint: str,
    ) -> None:
        assert len(chunks) == len(embeddings)
        self.fingerprint = fingerprint
        self.saved = document, chunks

    def record_attempt(
        self, document_id: str, status: AttemptStatus, detail: str | None = None
    ) -> None:
        self.attempts.append((document_id, status, detail))


def spec(document_id: str = "rails:example") -> SourceSpec:
    return SourceSpec(
        document_id=document_id,
        source="rails",
        version="8.1.4",
        format="markdown",
        fetch_url="https://example.test/example.md",
        source_url="https://guides.rubyonrails.org/v8.1/example.html",
        relative_path="rails/example.md",
    )


def test_ingestion_skips_unchanged_content_and_replaces_changed_content() -> None:
    client = FakeClient(b"Example\n=======\n\n## Querying\n\nUse an index.\n")
    store = FakeStore()
    embedder = FakeEmbedder()
    service = IngestDocument(client, store, embedder)

    first = service.call(spec())
    second = service.call(spec())
    client.content = b"Example\n=======\n\n## Querying\n\nUse a different index.\n"
    third = service.call(spec())

    assert (first.status, first.chunk_count) == ("completed", 1)
    assert (second.status, second.chunk_count) == ("skipped", None)
    assert (third.status, third.chunk_count) == ("completed", 1)
    assert len(embedder.calls) == 2
    assert embedder.calls[0] == ["passage: Example > Querying\nUse an index."]
    assert [status for _, status, _ in store.attempts] == ["completed", "skipped", "completed"]
    assert store.saved is not None
    assert store.saved[1][0].text == "Use a different index."


def test_ingestion_records_failure_for_retry() -> None:
    client = FakeClient(b"invalid document without headings")
    store = FakeStore()
    embedder = FakeEmbedder()

    with pytest.raises(ValueError, match="no headings"):
        IngestDocument(client, store, embedder).call(spec())

    assert store.saved is None
    assert store.attempts[0][1] == "failed"
    assert "no headings" in (store.attempts[0][2] or "")


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1", reason="requires PostgreSQL with pgvector"
)
def test_ingestion_is_idempotent_with_real_repository() -> None:
    document_id = f"rails:test-{uuid4().hex}"
    source = spec(document_id)
    client = FakeClient(b"Example\n=======\n\n## Querying\n\nUse an index.\n")
    embedder = FakeEmbedder()
    engine = get_engine()
    service = IngestDocument(client, IngestionRepository(engine), embedder)

    try:
        assert service.call(source).status == "completed"
        with Session(engine) as session:
            old_chunk_id = session.scalar(
                select(ChunkRecord.id).where(ChunkRecord.document_id == document_id)
            )
        assert service.call(source).status == "skipped"
        assert len(embedder.calls) == 1

        client.content = b"Example\n=======\n\n## Querying\n\nUse a different index.\n"
        assert service.call(source).status == "completed"
        assert len(embedder.calls) == 2

        with Session(engine) as session:
            chunks = session.scalars(
                select(ChunkRecord).where(ChunkRecord.document_id == document_id)
            ).all()
            attempts = session.scalars(
                select(IngestionAttemptRecord)
                .where(IngestionAttemptRecord.document_id == document_id)
                .order_by(IngestionAttemptRecord.id)
            ).all()
        assert len(chunks) == 1
        assert chunks[0].id != old_chunk_id
        assert chunks[0].text == "Use a different index."
        assert [attempt.status for attempt in attempts] == ["completed", "skipped", "completed"]
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM ingestion_attempts WHERE document_id = :id"), {"id": document_id}
            )
            connection.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})
