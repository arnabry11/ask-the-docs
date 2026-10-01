from dataclasses import dataclass

from sqlalchemy import Engine, func, literal_column, select
from sqlalchemy.orm import Session

from app.db.models import ChunkRecord, DocumentRecord

MAX_KEYWORD_CANDIDATES = 30


@dataclass(frozen=True)
class KeywordMatch:
    chunk_id: str
    document_id: str
    source: str
    version: str
    title: str
    section_path: tuple[str, ...]
    source_url: str
    text: str
    token_count: int
    fts_rank: float


class KeywordSearchRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def search(self, question: str, limit: int) -> list[KeywordMatch]:
        if not question.strip():
            raise ValueError("Question must not be blank")
        if not 1 <= limit <= MAX_KEYWORD_CANDIDATES:
            raise ValueError(f"limit must be between 1 and {MAX_KEYWORD_CANDIDATES}")

        query = func.websearch_to_tsquery(literal_column("'english'::regconfig"), question)
        rank = func.ts_rank_cd(ChunkRecord.search_vector, query)
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
                rank.label("fts_rank"),
            )
            .join(DocumentRecord, DocumentRecord.id == ChunkRecord.document_id)
            .where(ChunkRecord.search_vector.op("@@")(query))
            .order_by(rank.desc(), ChunkRecord.id)
            .limit(limit)
        )
        with Session(self.engine) as session:
            rows = session.execute(statement).mappings().all()
        return [
            KeywordMatch(
                chunk_id=row["id"],
                document_id=row["document_id"],
                source=row["source"],
                version=row["version"],
                title=row["title"],
                section_path=tuple(row["section_path"]),
                source_url=row["source_url"],
                text=row["text"],
                token_count=row["token_count"],
                fts_rank=float(row["fts_rank"]),
            )
            for row in rows
        ]
