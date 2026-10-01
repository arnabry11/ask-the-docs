import pytest

from evals.dataset import EvalQuestion
from evals.metrics import MODES, EvalOutcome, evaluate_metrics


def question(question_id: str, gold: tuple[str, ...]) -> EvalQuestion:
    return EvalQuestion(
        id=question_id,
        question="A test question?",
        gold_doc_ids=gold,
        key_facts=("fact one", "fact two") if gold else (),
        answerable=bool(gold),
        category="multi_doc" if len(gold) > 1 else "factual" if gold else "unanswerable",
    )


def outcome(question_id: str, ranking: tuple[str, ...], gated: bool) -> EvalOutcome:
    return EvalOutcome(
        question_id=question_id,
        rankings={mode: ranking for mode in MODES},
        gated=gated,
        gate_reason="below_threshold" if gated else None,
        best_rerank_score=0.1,
    )


def test_metrics_count_chunks_and_unique_gold_documents() -> None:
    questions = (
        question("Q001", ("a",)),
        question("Q002", ("a", "b")),
        question("Q003", ()),
    )
    outcomes = (
        outcome("Q001", ("x", "a"), False),
        outcome("Q002", ("a", "a", "b"), True),
        outcome("Q003", ("a",), True),
    )

    summary = evaluate_metrics(questions, outcomes, ks=(1, 2, 3))
    vector = summary["variants"]["vector"]

    assert vector["hit_rate@1"] == 0.5
    assert vector["recall@1"] == 0.25
    assert vector["mrr@1"] == 0.5
    assert vector["hit_rate@2"] == 1.0
    assert vector["recall@2"] == 0.75
    assert vector["mrr@2"] == 0.75
    assert vector["recall@3"] == 1.0
    assert summary["gate"]["unanswerable_gated"] == 1
    assert summary["gate"]["answerable_wrongly_gated"] == 1


def test_metrics_reject_missing_or_duplicate_outcomes() -> None:
    questions = (question("Q001", ("a",)), question("Q002", ()))

    with pytest.raises(ValueError, match="match questions"):
        evaluate_metrics(questions, (outcome("Q001", ("a",), False),))
    with pytest.raises(ValueError, match="match questions"):
        evaluate_metrics(
            questions,
            (outcome("Q001", ("a",), False), outcome("Q001", ("a",), False)),
        )
