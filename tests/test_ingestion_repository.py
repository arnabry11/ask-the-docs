import os
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.connection import get_engine
from app.db.models import ChunkRecord, DocumentRecord, IngestionAttemptRecord
from app.ingestion.chunk import chunk_document
from app.ingestion.parse import ParsedDocument, ParsedSection
from app.ingestion.repository import IngestionRepository


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1", reason="requires PostgreSQL with pgvector"
)
def test_repository_replaces_document_atomically_and_records_attempts() -> None:
    engine = get_engine()
    repository = IngestionRepository(engine)
    document_id = f"rails:test-{uuid4().hex}"
    document = ParsedDocument(
        document_id=document_id,
        source="rails",
        version="8.1.4",
        title="Index Guide",
        source_url="https://example.test/guide",
        content_hash="a" * 64,
        sections=(
            ParsedSection(
                ("Index Guide", "Vector search"),
                "https://example.test/guide#vector",
                "Vector indexes help search.",
            ),
        ),
    )
    vector = [0.0] * 383 + [1.0]
    first_chunks = chunk_document(document)

    try:
        repository.replace_document(document, first_chunks, [vector], "b" * 64)
        repository.record_attempt(document_id, "completed")

        with Session(engine) as session:
            stored = session.get(DocumentRecord, document_id)
            chunks = session.scalars(
                select(ChunkRecord).where(ChunkRecord.document_id == document_id)
            ).all()
            searchable = session.scalar(
                text(
                    "SELECT search_vector @@ plainto_tsquery('english', 'vector') "
                    "FROM chunks WHERE document_id = :id"
                ),
                {"id": document_id},
            )
            dimensions = session.scalar(
                text("SELECT vector_dims(embedding) FROM chunks WHERE document_id = :id"),
                {"id": document_id},
            )
        assert stored is not None
        assert (stored.title, stored.content_hash, stored.chunk_count) == (
            "Index Guide",
            "a" * 64,
            1,
        )
        assert repository.current_fingerprint(document_id) == "b" * 64
        assert len(chunks) == 1
        assert chunks[0].section_path == ["Index Guide", "Vector search"]
        assert chunks[0].source_url.endswith("#vector")
        assert searchable is True
        assert dimensions == 384

        changed = replace(
            document,
            title="Updated Guide",
            content_hash="c" * 64,
            sections=(
                ParsedSection(("Updated Guide",), document.source_url, "Different content."),
            ),
        )
        changed_chunks = chunk_document(changed)
        repository.replace_document(changed, changed_chunks, [vector], "d" * 64)
        assert repository.current_fingerprint(document_id) == "d" * 64

        with Session(engine) as session:
            assert session.get(ChunkRecord, first_chunks[0].chunk_id) is None
            assert session.get(ChunkRecord, changed_chunks[0].chunk_id) is not None
            assert session.get(DocumentRecord, document_id).title == "Updated Guide"  # type: ignore[union-attr]

        duplicate = replace(changed_chunks[0], ordinal=1)
        with pytest.raises(IntegrityError):
            repository.replace_document(
                changed, [changed_chunks[0], duplicate], [vector, vector], "e" * 64
            )

        with Session(engine) as session:
            stored = session.get(DocumentRecord, document_id)
            assert stored is not None
            assert stored.ingestion_fingerprint == "d" * 64
            assert session.get(ChunkRecord, changed_chunks[0].chunk_id) is not None
            attempts = session.scalars(
                select(IngestionAttemptRecord).where(
                    IngestionAttemptRecord.document_id == document_id
                )
            ).all()
            assert [attempt.status for attempt in attempts] == ["completed"]
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM ingestion_attempts WHERE document_id = :id"), {"id": document_id}
            )
            connection.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})
