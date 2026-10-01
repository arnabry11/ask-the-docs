from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from app.retrieval import rerank as rerank_module
from app.retrieval.fusion import FusedChunk
from app.retrieval.rerank import FastEmbedReranker, RerankService, rerank_text


def fused_chunk(chunk_id: str) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        document_id=f"rails:{chunk_id}",
        source="rails",
        version="8.1.4",
        title="Index Guide",
        section_path=("Index Guide", "Adding Indexes"),
        source_url=f"https://example.test/{chunk_id}",
        text=f"Documentation for {chunk_id}.",
        token_count=5,
        rrf_score=0.03,
        vector_rank=1,
        keyword_rank=None,
        cosine_distance=0.2,
        fts_rank=None,
    )


class FakeScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((question, list(passages)))
        return self.scores


def test_reranker_orders_scores_and_preserves_source_metadata() -> None:
    scorer = FakeScorer([-2.0, 3.0, 3.0])
    service = RerankService(scorer)
    chunks = [fused_chunk("c"), fused_chunk("b"), fused_chunk("a")]

    ranked = service.call("How do I add an index?", chunks, limit=2)

    assert [item.chunk.chunk_id for item in ranked] == ["a", "b"]
    assert [item.score for item in ranked] == [3.0, 3.0]
    assert ranked[0].chunk.source_url == "https://example.test/a"
    assert scorer.calls == [
        (
            "How do I add an index?",
            [
                "Index Guide > Adding Indexes\nDocumentation for c.",
                "Index Guide > Adding Indexes\nDocumentation for b.",
                "Index Guide > Adding Indexes\nDocumentation for a.",
            ],
        )
    ]


def test_rerank_text_adds_title_when_section_path_omits_it() -> None:
    chunk = fused_chunk("a")
    changed = replace(chunk, section_path=("Adding Indexes",))

    assert rerank_text(changed) == "Index Guide > Adding Indexes\nDocumentation for a."


def test_empty_candidates_skip_model_and_invalid_scores_fail() -> None:
    scorer = FakeScorer([])
    service = RerankService(scorer)

    assert service.call("question", [], limit=5) == ()
    assert scorer.calls == []
    with pytest.raises(ValueError, match="finite"):
        RerankService(FakeScorer([float("nan")])).call("question", [fused_chunk("a")], 1)
    with pytest.raises(ValueError, match="one finite score"):
        RerankService(FakeScorer([])).call("question", [fused_chunk("a")], 1)


def test_fastembed_adapter_uses_local_cross_encoder_and_checks_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[str]]] = []
            self.scores = [1.5, -0.5]

        def rerank(self, question: str, passages: Sequence[str]) -> Iterator[float]:
            self.calls.append((question, list(passages)))
            yield from self.scores

    model = FakeModel()
    options: dict[str, object] = {}

    def fake_encoder(**kwargs: object) -> FakeModel:
        options.update(kwargs)
        return model

    monkeypatch.setattr(rerank_module, "TextCrossEncoder", fake_encoder)
    adapter = FastEmbedReranker(Path("/tmp/unused-model-cache"))

    assert adapter.score("index", ["first", "second"]) == [1.5, -0.5]
    assert model.calls == [("index", ["first", "second"])]
    assert options == {
        "model_name": "Xenova/ms-marco-MiniLM-L-6-v2",
        "cache_dir": "/tmp/unused-model-cache",
    }
    model.scores = [float("inf")]
    with pytest.raises(ValueError, match="finite"):
        adapter.score("index", ["first"])
