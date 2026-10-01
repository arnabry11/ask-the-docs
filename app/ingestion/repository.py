import math
from collections.abc import Sequence
from typing import Literal

from sqlalchemy import Engine, delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import ChunkRecord, DocumentRecord, IngestionAttemptRecord
from app.ingestion.chunk import Chunk
from app.ingestion.constants import EMBEDDING_DIMENSIONS
from app.ingestion.parse import ParsedDocument

AttemptStatus = Literal["completed", "skipped", "failed"]


class IngestionRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def current_fingerprint(self, document_id: str) -> str | None:
        with Session(self.engine) as session:
            return session.scalar(
                select(DocumentRecord.ingestion_fingerprint).where(DocumentRecord.id == document_id)
            )

    def replace_document(
        self,
        document: ParsedDocument,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        fingerprint: str,
    ) -> None:
        if len(chunks) != len(embeddings) or len(fingerprint) != 64:
            raise ValueError("Invalid ingestion payload")
        if any(chunk.document_id != document.document_id for chunk in chunks):
            raise ValueError("Chunk belongs to a different document")
        if any(
            len(vector) != EMBEDDING_DIMENSIONS or not all(math.isfinite(value) for value in vector)
            for vector in embeddings
        ):
            raise ValueError(f"Embeddings must contain {EMBEDDING_DIMENSIONS} finite values")

        fields = {
            "id": document.document_id,
            "source": document.source,
            "version": document.version,
            "title": document.title,
            "source_url": document.source_url,
            "content_hash": document.content_hash,
            "ingestion_fingerprint": fingerprint,
            "chunk_count": len(chunks),
        }
        upsert = insert(DocumentRecord).values(**fields)
        upsert = upsert.on_conflict_do_update(
            index_elements=[DocumentRecord.id],
            set_={
                **{key: value for key, value in fields.items() if key != "id"},
                "updated_at": upsert.excluded.updated_at,
            },
        )
        with Session(self.engine) as session, session.begin():
            session.execute(upsert)
            session.execute(
                delete(ChunkRecord).where(ChunkRecord.document_id == document.document_id)
            )
            session.add_all(
                ChunkRecord(
                    id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    ordinal=chunk.ordinal,
                    section_path=list(chunk.section_path),
                    source_url=chunk.source_url,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    embedding=list(vector),
                )
                for chunk, vector in zip(chunks, embeddings, strict=True)
            )

    def record_attempt(
        self, document_id: str, status: AttemptStatus, detail: str | None = None
    ) -> None:
        with Session(self.engine) as session, session.begin():
            session.add(
                IngestionAttemptRecord(document_id=document_id, status=status, detail=detail)
            )
