from dataclasses import replace

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.keyword_repository import KeywordMatch
from app.retrieval.vector_repository import RetrievedChunk


def vector_match(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"rails:{chunk_id}",
        source="rails",
        version="8.1.4",
        title=f"Guide {chunk_id}",
        section_path=("Guide", chunk_id),
        source_url=f"https://example.test/{chunk_id}",
        text=f"Text for {chunk_id}",
        token_count=4,
        cosine_distance=0.1,
    )


def keyword_match(chunk_id: str) -> KeywordMatch:
    chunk = vector_match(chunk_id)
    return KeywordMatch(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source=chunk.source,
        version=chunk.version,
        title=chunk.title,
        section_path=chunk.section_path,
        source_url=chunk.source_url,
        text=chunk.text,
        token_count=chunk.token_count,
        fts_rank=0.4,
    )


def test_fusion_rewards_overlap_and_retains_each_search_rank() -> None:
    fused = reciprocal_rank_fusion(
        [vector_match("a"), vector_match("b"), vector_match("c")],
        [keyword_match("b"), keyword_match("d"), keyword_match("a")],
        limit=3,
    )

    assert [chunk.chunk_id for chunk in fused] == ["b", "a", "d"]
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert (fused[0].vector_rank, fused[0].keyword_rank) == (2, 1)
    assert fused[0].cosine_distance == 0.1
    assert fused[0].fts_rank == 0.4
    assert (fused[2].vector_rank, fused[2].keyword_rank) == (None, 2)
    assert fused[2].cosine_distance is None
    assert fused[2].title == "Guide d"


def test_fusion_has_stable_ties_and_handles_keyword_free_queries() -> None:
    tied = reciprocal_rank_fusion(
        [vector_match("b"), vector_match("a")],
        [keyword_match("a"), keyword_match("b")],
        limit=2,
    )
    vector_only = reciprocal_rank_fusion([vector_match("b")], [], limit=1)

    assert [chunk.chunk_id for chunk in tied] == ["a", "b"]
    assert vector_only[0].keyword_rank is None
    assert vector_only[0].fts_rank is None
    assert vector_only[0].rrf_score == pytest.approx(1 / 61)


def test_fusion_uses_first_rank_for_duplicate_candidates() -> None:
    duplicate = replace(vector_match("a"), cosine_distance=0.2)

    fused = reciprocal_rank_fusion([vector_match("a"), duplicate], [], limit=1)

    assert fused[0].vector_rank == 1
    assert fused[0].cosine_distance == 0.1
    assert fused[0].rrf_score == pytest.approx(1 / 61)
