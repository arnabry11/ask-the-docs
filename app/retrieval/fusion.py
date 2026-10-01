from collections.abc import Sequence
from dataclasses import dataclass

from app.retrieval.keyword_repository import KeywordMatch
from app.retrieval.vector_repository import RetrievedChunk

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedChunk:
    chunk_id: str
    document_id: str
    source: str
    version: str
    title: str
    section_path: tuple[str, ...]
    source_url: str
    text: str
    token_count: int
    rrf_score: float
    vector_rank: int | None
    keyword_rank: int | None
    cosine_distance: float | None
    fts_rank: float | None


def reciprocal_rank_fusion(
    vector_results: Sequence[RetrievedChunk],
    keyword_results: Sequence[KeywordMatch],
    limit: int,
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[FusedChunk]:
    if limit < 1 or rrf_k < 1:
        raise ValueError("limit and rrf_k must be positive")

    vector_by_id: dict[str, RetrievedChunk] = {}
    keyword_by_id: dict[str, KeywordMatch] = {}
    vector_ranks: dict[str, int] = {}
    keyword_ranks: dict[str, int] = {}
    for rank, vector_chunk in enumerate(vector_results, 1):
        vector_by_id.setdefault(vector_chunk.chunk_id, vector_chunk)
        vector_ranks.setdefault(vector_chunk.chunk_id, rank)
    for rank, keyword_chunk in enumerate(keyword_results, 1):
        keyword_by_id.setdefault(keyword_chunk.chunk_id, keyword_chunk)
        keyword_ranks.setdefault(keyword_chunk.chunk_id, rank)
    fused: list[FusedChunk] = []

    for chunk_id in vector_by_id.keys() | keyword_by_id.keys():
        vector = vector_by_id.get(chunk_id)
        keyword = keyword_by_id.get(chunk_id)
        selected_chunk = vector if vector is not None else keyword
        assert selected_chunk is not None
        vector_rank = vector_ranks.get(chunk_id)
        keyword_rank = keyword_ranks.get(chunk_id)
        score = sum(1 / (rrf_k + rank) for rank in (vector_rank, keyword_rank) if rank is not None)
        fused.append(
            FusedChunk(
                chunk_id=selected_chunk.chunk_id,
                document_id=selected_chunk.document_id,
                source=selected_chunk.source,
                version=selected_chunk.version,
                title=selected_chunk.title,
                section_path=selected_chunk.section_path,
                source_url=selected_chunk.source_url,
                text=selected_chunk.text,
                token_count=selected_chunk.token_count,
                rrf_score=score,
                vector_rank=vector_rank,
                keyword_rank=keyword_rank,
                cosine_distance=vector.cosine_distance if vector is not None else None,
                fts_rank=keyword.fts_rank if keyword is not None else None,
            )
        )

    return sorted(fused, key=lambda chunk: (-chunk.rrf_score, chunk.chunk_id))[:limit]
