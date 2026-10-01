import math
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.db.models import ChunkRecord, DocumentRecord
from app.ingestion.constants import EMBEDDING_DIMENSIONS

MAX_QUERY_RESULTS = 20


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source: str
    version: str
    title: str
    section_path: tuple[str, ...]
    source_url: str
    text: str
    token_count: int
    cosine_distance: float


class VectorSearchRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def search(self, vector: Sequence[float], limit: int) -> list[RetrievedChunk]:
        if not 1 <= limit <= MAX_QUERY_RESULTS:
            raise ValueError(f"limit must be between 1 and {MAX_QUERY_RESULTS}")
        if (
            len(vector) != EMBEDDING_DIMENSIONS
            or not all(math.isfinite(value) for value in vector)
            or not any(value != 0 for value in vector)
        ):
            raise ValueError(
                f"Query vector must contain {EMBEDDING_DIMENSIONS} finite, nonzero values"
            )

        distance = ChunkRecord.embedding.cosine_distance(list(vector))
        statement = (
            select(
                ChunkRecord.id,
                ChunkRecord.document_id,
                DocumentRecord.source,
                DocumentRecord.version,
                DocumentRecord.title,
                ChunkRecord.section_path,
                ChunkRecord.source_url,
                ChunkRecord.text,
                ChunkRecord.token_count,
                distance.label("cosine_distance"),
            )
            .join(DocumentRecord, DocumentRecord.id == ChunkRecord.document_id)
            .order_by(distance)
            .limit(limit)
        )
        with Session(self.engine) as session:
            rows = session.execute(statement).mappings().all()
        return [
            RetrievedChunk(
                chunk_id=row["id"],
                document_id=row["document_id"],
                source=row["source"],
                version=row["version"],
                title=row["title"],
                section_path=tuple(row["section_path"]),
                source_url=row["source_url"],
                text=row["text"],
                token_count=row["token_count"],
                cosine_distance=float(row["cosine_distance"]),
            )
            for row in rows
        ]
