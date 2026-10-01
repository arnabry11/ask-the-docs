import os
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.db.connection import get_engine
from app.ingestion.chunk import chunk_document
from app.ingestion.parse import ParsedDocument, ParsedSection
from app.ingestion.repository import IngestionRepository
from app.retrieval.vector_repository import VectorSearchRepository


def document(document_id: str, title: str, body: str) -> ParsedDocument:
    url = f"https://example.test/{document_id.split(':')[1]}"
    return ParsedDocument(
        document_id=document_id,
        source="rails",
        version="8.1.4",
        title=title,
        source_url=url,
        content_hash="a" * 64,
        sections=(ParsedSection((title, "Search"), f"{url}#search", body),),
    )


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1", reason="requires PostgreSQL with pgvector"
)
def test_cosine_search_orders_chunks_and_returns_citations() -> None:
    engine = get_engine()
    writer = IngestionRepository(engine)
    reader = VectorSearchRepository(engine)
    prefix = uuid4().hex
    close = document(f"rails:{prefix}-close", "Close Guide", "Use a matching index.")
    far = document(f"rails:{prefix}-far", "Far Guide", "A different topic.")
    close_chunk = chunk_document(close)[0]
    far_chunk = chunk_document(far)[0]
    query_vector = [1.0] + [0.0] * 383

    try:
        writer.replace_document(close, [close_chunk], [query_vector], "b" * 64)
        writer.replace_document(far, [far_chunk], [[0.0, 1.0] + [0.0] * 382], "c" * 64)

        results = reader.search(query_vector, 20)
        ids = [result.chunk_id for result in results]
        nearest = reader.search(query_vector, 1)

        assert ids.index(close_chunk.chunk_id) < ids.index(far_chunk.chunk_id)
        assert len(nearest) == 1
        assert nearest[0].chunk_id == close_chunk.chunk_id
        assert nearest[0].document_id == close.document_id
        assert nearest[0].title == "Close Guide"
        assert nearest[0].section_path == ("Close Guide", "Search")
        assert nearest[0].source_url.endswith("#search")
        assert nearest[0].text == "Use a matching index."
        assert nearest[0].cosine_distance == pytest.approx(0.0)
        assert results[ids.index(far_chunk.chunk_id)].cosine_distance == pytest.approx(1.0)
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM documents WHERE id IN (:close_id, :far_id)"),
                {"close_id": close.document_id, "far_id": far.document_id},
            )


def test_vector_search_rejects_invalid_limits_and_vectors() -> None:
    reader = VectorSearchRepository(get_engine())

    with pytest.raises(ValueError, match="limit"):
        reader.search([1.0] * 384, 0)
    with pytest.raises(ValueError, match="limit"):
        reader.search([1.0] * 384, 31)
    with pytest.raises(ValueError, match="nonzero"):
        reader.search([0.0] * 384, 5)
    with pytest.raises(ValueError, match="384"):
        reader.search([1.0], 5)
