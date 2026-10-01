from dataclasses import dataclass
from typing import Literal

from evals.dataset import EvalQuestion

Mode = Literal["vector", "keyword", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("vector", "keyword", "hybrid", "hybrid_rerank")
DEFAULT_KS = (1, 5, 10)


@dataclass(frozen=True)
class EvalOutcome:
    question_id: str
    rankings: dict[Mode, tuple[str, ...]]
    gated: bool
    gate_reason: str | None
    best_rerank_score: float | None


def evaluate_metrics(
    questions: tuple[EvalQuestion, ...],
    outcomes: tuple[EvalOutcome, ...],
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, object]:
    if not questions or len({item.id for item in questions}) != len(questions):
        raise ValueError("Questions must be nonempty and unique")
    if not ks or any(k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("Cutoffs must be unique positive integers")
    by_id = {outcome.question_id: outcome for outcome in outcomes}
    if len(by_id) != len(outcomes) or set(by_id) != {item.id for item in questions}:
        raise ValueError("Outcomes must match questions exactly")
    if any(set(outcome.rankings) != set(MODES) for outcome in outcomes):
        raise ValueError("Every outcome must include all retrieval modes")

    answerable = [item for item in questions if item.answerable]
    unanswerable = [item for item in questions if not item.answerable]
    if not answerable or not unanswerable:
        raise ValueError("Both answerable and unanswerable questions are required")

    variants: dict[Mode, dict[str, float]] = {}
    for mode in MODES:
        scores: dict[str, float] = {}
        for k in ks:
            hits = recalls = reciprocal_ranks = 0.0
            for item in answerable:
                predicted = by_id[item.id].rankings[mode][:k]
                gold = set(item.gold_doc_ids)
                first_hit = next(
                    (rank for rank, document_id in enumerate(predicted, 1) if document_id in gold),
                    None,
                )
                hits += first_hit is not None
                recalls += len(set(predicted) & gold) / len(gold)
                reciprocal_ranks += 1 / first_hit if first_hit is not None else 0.0
            scores[f"hit_rate@{k}"] = hits / len(answerable)
            scores[f"recall@{k}"] = recalls / len(answerable)
            scores[f"mrr@{k}"] = reciprocal_ranks / len(answerable)
        variants[mode] = scores

    correctly_gated = sum(by_id[item.id].gated for item in unanswerable)
    wrongly_gated = sum(by_id[item.id].gated for item in answerable)
    return {
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "variants": variants,
        "gate": {
            "unanswerable_gated": correctly_gated,
            "unanswerable_gate_rate": correctly_gated / len(unanswerable),
            "answerable_wrongly_gated": wrongly_gated,
            "answerable_false_gate_rate": wrongly_gated / len(answerable),
        },
    }
