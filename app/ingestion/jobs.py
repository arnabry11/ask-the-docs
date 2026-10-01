import logging
from dataclasses import asdict
from functools import lru_cache

from app.db.connection import get_engine
from app.ingestion.corpus_client import CorpusHttpClient
from app.ingestion.embed import FastEmbedClient
from app.ingestion.repository import IngestionRepository
from app.ingestion.service import IngestDocument
from app.ingestion.sources import load_source_specs

LOGGER = logging.getLogger(__name__)


@lru_cache
def get_embedder() -> FastEmbedClient:
    return FastEmbedClient()


def ingest_document_job(document_id: str) -> dict[str, str | int | None]:
    spec = next((item for item in load_source_specs() if item.document_id == document_id), None)
    if spec is None:
        raise ValueError(f"Unknown document ID: {document_id}")

    with CorpusHttpClient() as client:
        result = IngestDocument(client, IngestionRepository(get_engine()), get_embedder()).call(
            spec
        )
    LOGGER.info("Ingestion %s for %s (%s chunks)", result.status, document_id, result.chunk_count)
    return asdict(result)
