import pytest

from app.retrieval.fusion import FusedChunk
from app.retrieval.gate import REFUSAL_MESSAGE, decide_gate
from app.retrieval.rerank import RerankedChunk


def result(score: float) -> RerankedChunk:
    return RerankedChunk(
        FusedChunk(
            chunk_id="rails:a:1",
            document_id="rails:a",
            source="rails",
            version="8.1.4",
            title="Guide",
            section_path=("Guide",),
            source_url="https://example.test/a",
            text="A source passage.",
            token_count=4,
            rrf_score=0.02,
            vector_rank=1,
            keyword_rank=None,
            cosine_distance=0.2,
            fts_rank=None,
        ),
        score,
    )


def test_gate_refuses_when_no_sources_exist() -> None:
    decision = decide_gate([], None)

    assert decision.gated is True
    assert decision.reason == "no_sources"
    assert decision.refusal == REFUSAL_MESSAGE


def test_gate_is_unset_until_calibration() -> None:
    decision = decide_gate([result(-100.0)], None)

    assert decision.gated is False
    assert decision.threshold is None
    assert decision.refusal is None


def test_gate_compares_best_raw_score_to_configured_threshold() -> None:
    assert decide_gate([result(0.2), result(0.4)], 0.3).gated is False
    assert decide_gate([result(0.3)], 0.3).gated is False

    refused = decide_gate([result(0.2)], 0.3)
    assert refused.gated is True
    assert refused.reason == "below_threshold"
    assert refused.refusal == REFUSAL_MESSAGE


def test_gate_rejects_nonfinite_threshold() -> None:
    with pytest.raises(ValueError, match="finite"):
        decide_gate([result(0.2)], float("nan"))
