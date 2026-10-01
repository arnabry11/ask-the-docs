import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.retrieval.rerank import RerankedChunk

REFUSAL_MESSAGE = "The available documentation does not cover this question."
GateReason = Literal["no_sources", "below_threshold"]


@dataclass(frozen=True)
class GateDecision:
    gated: bool
    reason: GateReason | None
    refusal: str | None
    threshold: float | None


def decide_gate(results: Sequence[RerankedChunk], threshold: float | None) -> GateDecision:
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("Gate threshold must be finite")
    if not results:
        return GateDecision(True, "no_sources", REFUSAL_MESSAGE, threshold)
    if threshold is not None and max(item.score for item in results) < threshold:
        return GateDecision(True, "below_threshold", REFUSAL_MESSAGE, threshold)
    return GateDecision(False, None, None, threshold)
