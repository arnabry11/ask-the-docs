import os
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.db.connection import get_engine
from app.ingestion.chunk import chunk_document
from app.ingestion.parse import ParsedDocument, ParsedSection
from app.ingestion.repository import IngestionRepository
from app.retrieval.keyword_repository import KeywordSearchRepository


def document(document_id: str, body: str) -> ParsedDocument:
    url = f"https://example.test/{document_id.split(':')[1]}"
    return ParsedDocument(
        document_id=document_id,
        source="rails",
        version="8.1.4",
        title="Index Guide",
        source_url=url,
        content_hash="a" * 64,
        sections=(ParsedSection(("Index Guide", "Search"), f"{url}#search", body),),
    )


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1", reason="requires PostgreSQL with pgvector"
)
def test_keyword_search_uses_web_query_ranking_and_returns_source_metadata() -> None:
    engine = get_engine()
    writer = IngestionRepository(engine)
    reader = KeywordSearchRepository(engine)
    prefix = uuid4().hex
    first = document(f"rails:{prefix}-a", "A GIN index helps full text search.")
    second = document(f"rails:{prefix}-b", "A GIN index helps full text search.")
    unrelated = document(f"rails:{prefix}-c", "Vector similarity compares embeddings.")

    try:
        for item in (first, second, unrelated):
            writer.replace_document(item, chunk_document(item), [[1.0] + [0.0] * 383], "b" * 64)

        phrase = reader.search('"full text search"', 30)
        matches = [match for match in phrase if match.document_id.startswith(f"rails:{prefix}")]
        assert [match.document_id for match in matches] == [first.document_id, second.document_id]
        assert all(match.fts_rank > 0 for match in matches)
        assert matches[0].title == "Index Guide"
        assert matches[0].section_path == ("Index Guide", "Search")
        assert matches[0].source_url.endswith("#search")
        assert matches[0].text == "A GIN index helps full text search."
        assert reader.search('"full text search"', 1)[0].document_id == first.document_id
        assert reader.search("???", 30) == []
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM documents WHERE id IN (:first, :second, :unrelated)"),
                {
                    "first": first.document_id,
                    "second": second.document_id,
                    "unrelated": unrelated.document_id,
                },
            )


def test_keyword_search_rejects_blank_questions_and_invalid_limits() -> None:
    reader = KeywordSearchRepository(get_engine())

    with pytest.raises(ValueError, match="blank"):
        reader.search("  ", 5)
    with pytest.raises(ValueError, match="limit"):
        reader.search("index", 0)
    with pytest.raises(ValueError, match="limit"):
        reader.search("index", 31)
