import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.ingestion.embed import DEFAULT_MODEL_CACHE
from app.retrieval.fusion import FusedChunk

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
MAX_RERANK_CANDIDATES = 20


class RerankScorer(Protocol):
    def score(self, question: str, passages: Sequence[str]) -> list[float]: ...


@dataclass(frozen=True)
class RerankedChunk:
    chunk: FusedChunk
    score: float


def rerank_text(chunk: FusedChunk) -> str:
    headings = list(chunk.section_path)
    if not headings or headings[0] != chunk.title:
        headings.insert(0, chunk.title)
    return f"{' > '.join(headings)}\n{chunk.text}"


class FastEmbedReranker:
    def __init__(
        self,
        cache_dir: Path = DEFAULT_MODEL_CACHE,
        model_name: str = DEFAULT_RERANK_MODEL,
    ) -> None:
        self.model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = [float(score) for score in self.model.rerank(question, passages)]
        if len(scores) != len(passages) or not all(math.isfinite(score) for score in scores):
            raise ValueError("Reranker must return one finite score per passage")
        return scores


class RerankService:
    def __init__(self, scorer: RerankScorer) -> None:
        self.scorer = scorer

    def call(
        self, question: str, candidates: Sequence[FusedChunk], limit: int
    ) -> tuple[RerankedChunk, ...]:
        if not question.strip() or not 1 <= limit <= MAX_RERANK_CANDIDATES:
            raise ValueError("Question and rerank limit must be valid")
        if not candidates:
            return ()
        if len(candidates) > MAX_RERANK_CANDIDATES:
            raise ValueError(f"Reranker accepts at most {MAX_RERANK_CANDIDATES} candidates")

        scores = self.scorer.score(question, [rerank_text(chunk) for chunk in candidates])
        if len(scores) != len(candidates) or not all(math.isfinite(score) for score in scores):
            raise ValueError("Reranker must return one finite score per candidate")
        ranked = [
            RerankedChunk(chunk, score) for chunk, score in zip(candidates, scores, strict=True)
        ]
        return tuple(sorted(ranked, key=lambda item: (-item.score, item.chunk.chunk_id))[:limit])
