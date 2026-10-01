import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.db.connection import get_engine
from app.ingestion.constants import EMBEDDING_DIMENSIONS
from app.ingestion.embed import FastEmbedClient, QueryEmbeddingClient
from app.retrieval.vector_repository import (
    MAX_QUERY_RESULTS,
    RetrievedChunk,
    VectorSearchRepository,
)

MAX_QUESTION_LENGTH = 500
DEFAULT_TOP_K = 5


class VectorSearcher(Protocol):
    def search(self, vector: list[float], limit: int) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class QueryResult:
    question: str
    chunks: tuple[RetrievedChunk, ...]


class QueryService:
    def __init__(self, embedder: QueryEmbeddingClient, searcher: VectorSearcher) -> None:
        self.embedder = embedder
        self.searcher = searcher

    def call(self, question: str, top_k: int = DEFAULT_TOP_K) -> QueryResult:
        normalized = " ".join(question.split())
        if not normalized or len(normalized) > MAX_QUESTION_LENGTH:
            raise ValueError(f"Question must contain 1 to {MAX_QUESTION_LENGTH} characters")
        if not 1 <= top_k <= MAX_QUERY_RESULTS:
            raise ValueError(f"top_k must be between 1 and {MAX_QUERY_RESULTS}")

        vector = self.embedder.embed_query(normalized)
        if (
            len(vector) != EMBEDDING_DIMENSIONS
            or not all(math.isfinite(value) for value in vector)
            or not any(value != 0 for value in vector)
        ):
            raise ValueError(
                f"Query embedding must contain {EMBEDDING_DIMENSIONS} finite, nonzero values"
            )
        return QueryResult(normalized, tuple(self.searcher.search(vector, top_k)))


@lru_cache
def get_query_service() -> QueryService:
    return QueryService(FastEmbedClient(), VectorSearchRepository(get_engine()))
