"""Persist verified answers for repeat questions."""

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import LlmCacheRecord


@dataclass(frozen=True)
class CachedAnswer:
    key: str
    answer: str
    citations: list[dict[str, object]]
    model: str
    prompt_version: str


def _cached(row: LlmCacheRecord) -> CachedAnswer:
    return CachedAnswer(row.key, row.answer, row.citations, row.model, row.prompt_version)


class GenerationRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def get_cached(self, key: str) -> CachedAnswer | None:
        with Session(self.engine) as session:
            row = session.get(LlmCacheRecord, key)
            return _cached(row) if row is not None else None

    def save(self, answer: CachedAnswer) -> CachedAnswer:
        if len(answer.key) != 64 or not answer.answer.strip() or not answer.model:
            raise ValueError("Verified answer needs a cache key, text, and model")
        with Session(self.engine) as session, session.begin():
            statement = insert(LlmCacheRecord).values(
                key=answer.key,
                answer=answer.answer,
                citations=answer.citations,
                model=answer.model,
                prompt_version=answer.prompt_version,
            )
            session.execute(statement.on_conflict_do_nothing(index_elements=[LlmCacheRecord.key]))
            row = session.get(LlmCacheRecord, answer.key)
            if row is None:
                raise RuntimeError("Saved answer is missing from the cache")
            return _cached(row)
