import os
from collections.abc import Sequence
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.db.connection import get_engine
from app.ingestion.chunk import chunk_document
from app.ingestion.parse import ParsedDocument, ParsedSection
from app.ingestion.repository import IngestionRepository
from app.retrieval.keyword_repository import KeywordSearchRepository
from app.retrieval.rerank import RerankService
from app.retrieval.service import QueryService
from app.retrieval.vector_repository import VectorSearchRepository


class FakeEmbedder:
    def embed_query(self, question: str) -> list[float]:
        return [1.0] + [0.0] * 383


class PreserveFusionOrderScorer:
    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        return [float(len(passages) - index) for index in range(len(passages))]


def document(document_id: str, body: str) -> ParsedDocument:
    url = f"https://example.test/{document_id.split(':')[1]}"
    return ParsedDocument(
        document_id=document_id,
        source="rails",
        version="8.1.4",
        title="Search Guide",
        source_url=url,
        content_hash="a" * 64,
        sections=(ParsedSection(("Search Guide",), url, body),),
    )


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1", reason="requires PostgreSQL with pgvector"
)
def test_hybrid_query_promotes_exact_terms_and_keeps_vector_fallback() -> None:
    engine = get_engine()
    writer = IngestionRepository(engine)
    prefix = uuid4().hex
    vector_doc = document(f"rails:{prefix}-vector", "A planet catalog describes local vectors.")
    keyword_doc = document(f"rails:{prefix}-keyword", "Quasar quasar quasar indexing guidance.")

    try:
        writer.replace_document(
            vector_doc, chunk_document(vector_doc), [[1.0] + [0.0] * 383], "b" * 64
        )
        writer.replace_document(
            keyword_doc, chunk_document(keyword_doc), [[0.0, 1.0] + [0.0] * 382], "c" * 64
        )
        service = QueryService(
            FakeEmbedder(),
            VectorSearchRepository(engine),
            KeywordSearchRepository(engine),
            RerankService(PreserveFusionOrderScorer()),
        )

        exact = service.call("quasar indexing", top_k=1)
        fallback = service.call("???", top_k=1)

        assert exact.chunks[0].chunk.document_id == keyword_doc.document_id
        assert exact.chunks[0].chunk.keyword_rank == 1
        assert exact.chunks[0].chunk.vector_rank == 2
        assert fallback.chunks[0].chunk.document_id == vector_doc.document_id
        assert fallback.chunks[0].chunk.keyword_rank is None
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM documents WHERE id IN (:vector_id, :keyword_id)"),
                {"vector_id": vector_doc.document_id, "keyword_id": keyword_doc.document_id},
            )
