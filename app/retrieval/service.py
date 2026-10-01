import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.config import get_settings
from app.db.connection import get_engine
from app.ingestion.constants import EMBEDDING_DIMENSIONS
from app.ingestion.embed import FastEmbedClient, QueryEmbeddingClient
from app.retrieval.fusion import DEFAULT_RRF_K, FusedChunk, reciprocal_rank_fusion
from app.retrieval.keyword_repository import (
    MAX_KEYWORD_CANDIDATES,
    KeywordMatch,
    KeywordSearchRepository,
)
from app.retrieval.vector_repository import (
    MAX_QUERY_RESULTS,
    MAX_VECTOR_CANDIDATES,
    RetrievedChunk,
    VectorSearchRepository,
)

MAX_QUESTION_LENGTH = 500
DEFAULT_TOP_K = 5


class VectorSearcher(Protocol):
    def search(self, vector: list[float], limit: int) -> list[RetrievedChunk]: ...


class KeywordSearcher(Protocol):
    def search(self, question: str, limit: int) -> list[KeywordMatch]: ...


@dataclass(frozen=True)
class QueryResult:
    question: str
    chunks: tuple[FusedChunk, ...]


class QueryService:
    def __init__(
        self,
        embedder: QueryEmbeddingClient,
        vector_searcher: VectorSearcher,
        keyword_searcher: KeywordSearcher,
        *,
        vector_limit: int = MAX_VECTOR_CANDIDATES,
        keyword_limit: int = MAX_KEYWORD_CANDIDATES,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if not 1 <= vector_limit <= MAX_VECTOR_CANDIDATES:
            raise ValueError("vector_limit is out of range")
        if not 1 <= keyword_limit <= MAX_KEYWORD_CANDIDATES:
            raise ValueError("keyword_limit is out of range")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        self.embedder = embedder
        self.vector_searcher = vector_searcher
        self.keyword_searcher = keyword_searcher
        self.vector_limit = vector_limit
        self.keyword_limit = keyword_limit
        self.rrf_k = rrf_k

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
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(self.vector_searcher.search, vector, self.vector_limit)
            keyword_future = executor.submit(
                self.keyword_searcher.search, normalized, self.keyword_limit
            )
            vector_results = vector_future.result()
            keyword_results = keyword_future.result()
        return QueryResult(
            normalized,
            tuple(reciprocal_rank_fusion(vector_results, keyword_results, top_k, rrf_k=self.rrf_k)),
        )


@lru_cache
def get_query_service() -> QueryService:
    settings = get_settings()
    engine = get_engine()
    return QueryService(
        FastEmbedClient(),
        VectorSearchRepository(engine),
        KeywordSearchRepository(engine),
        vector_limit=settings.top_k_vector,
        keyword_limit=settings.top_k_fts,
        rrf_k=settings.rrf_k,
    )
