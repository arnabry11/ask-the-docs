import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, Select, func, literal_column, select
from sqlalchemy.orm import Session

from app.db.models import ChunkRecord, DocumentRecord

MAX_KEYWORD_CANDIDATES = 30
MAX_FALLBACK_TERMS = 5
FALLBACK_STOP_WORDS = frozenset(
    {
        "what",
        "which",
        "when",
        "where",
        "does",
        "how",
        "the",
        "and",
        "for",
        "can",
        "are",
        "with",
        "from",
        "that",
        "this",
        "into",
        "use",
        "using",
    }
)


def fallback_query(question: str) -> str | None:
    if '"' in question or re.search(r"\bOR\b", question):
        return None
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", question.lower())
    terms = list(
        dict.fromkeys(word for word in words if len(word) >= 3 and word not in FALLBACK_STOP_WORDS)
    )
    if len(terms) < 2:
        return None
    selected = sorted(terms, key=len, reverse=True)[:MAX_FALLBACK_TERMS]
    return " OR ".join(selected)


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

        with Session(self.engine) as session:
            rows = session.execute(self._statement(question, limit)).mappings().all()
            broader_query = fallback_query(question)
            if not rows and broader_query is not None:
                rows = session.execute(self._statement(broader_query, limit)).mappings().all()
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

    @staticmethod
    def _statement(question: str, limit: int) -> Select[tuple[Any, ...]]:
        query = func.websearch_to_tsquery(literal_column("'english'::regconfig"), question)
        rank = func.ts_rank_cd(ChunkRecord.search_vector, query)
        return (
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
