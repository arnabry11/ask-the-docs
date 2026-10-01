"""Choose a reranker gate threshold from a committed retrieval report."""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ingestion.sources import REPO_ROOT
from evals.dataset import DEFAULT_DATASET, load_dataset
from evals.run_retrieval_evals import DEFAULT_OUTPUT as DEFAULT_REPORT

DEFAULT_OUTPUT = REPO_ROOT / "evals" / "results" / "gate_calibration.json"
MAX_FALSE_GATE_RATE = 0.02


@dataclass(frozen=True)
class GateSample:
    question_id: str
    answerable: bool
    best_score: float | None


def load_samples(
    dataset_path: Path, report_path: Path
) -> tuple[tuple[GateSample, ...], dict[str, Any]]:
    dataset = load_dataset(dataset_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if report.get("dataset_sha256") != dataset_hash:
        raise ValueError("Retrieval report does not match the evaluation dataset")
    rows = report.get("questions")
    if not isinstance(rows, list) or len(rows) != len(dataset):
        raise ValueError("Retrieval report must contain one row per dataset question")
    by_id = {item.id: item for item in dataset}
    if any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows):
        raise ValueError("Retrieval report has an invalid question row")
    if len({row["id"] for row in rows}) != len(rows) or {row["id"] for row in rows} != set(by_id):
        raise ValueError("Retrieval report question IDs must match the dataset exactly")

    samples = []
    for row in rows:
        item = by_id[row["id"]]
        if row.get("answerable") is not item.answerable:
            raise ValueError(f"Answerability label differs for {item.id}")
        if "best_rerank_score" not in row:
            raise ValueError(f"Best reranker score is missing for {item.id}")
        score = row.get("best_rerank_score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
        ):
            raise ValueError(f"Best reranker score must be finite for {item.id}")
        rankings = row.get("rankings")
        reranked = rankings.get("hybrid_rerank") if isinstance(rankings, dict) else None
        if not isinstance(reranked, list) or (score is None) != (not reranked):
            raise ValueError(f"Best reranker score and ranking disagree for {item.id}")
        samples.append(
            GateSample(item.id, item.answerable, float(score) if score is not None else None)
        )
    return tuple(samples), report


def gate_counts(samples: tuple[GateSample, ...], threshold: float | None) -> dict[str, Any]:
    answerable = [item for item in samples if item.answerable]
    unanswerable = [item for item in samples if not item.answerable]
    if not answerable or not unanswerable:
        raise ValueError("Calibration needs both answerable and unanswerable questions")
    gated = [
        item
        for item in samples
        if item.best_score is None or (threshold is not None and item.best_score < threshold)
    ]
    false_gates = [item.question_id for item in gated if item.answerable]
    correct_gates = [item.question_id for item in gated if not item.answerable]
    return {
        "threshold": threshold,
        "unanswerable_gated": len(correct_gates),
        "unanswerable_total": len(unanswerable),
        "answerable_wrongly_gated": len(false_gates),
        "answerable_total": len(answerable),
        "balanced_accuracy": (
            len(correct_gates) / len(unanswerable)
            + (len(answerable) - len(false_gates)) / len(answerable)
        )
        / 2,
        "correct_gate_ids": correct_gates,
        "false_gate_ids": false_gates,
        "unanswerable_not_gated_ids": [
            item.question_id for item in unanswerable if item.question_id not in correct_gates
        ],
    }


def candidate_thresholds(samples: tuple[GateSample, ...]) -> tuple[float, ...]:
    scores = sorted({item.best_score for item in samples if item.best_score is not None})
    if not scores:
        return (0.0,)
    return (
        scores[0] - 1.0,
        *(left + (right - left) / 2 for left, right in zip(scores, scores[1:], strict=False)),
        scores[-1] + 1.0,
    )


def rounded_operating_point(samples: tuple[GateSample, ...], threshold: float) -> dict[str, Any]:
    original = gate_counts(samples, threshold)
    rounded = gate_counts(samples, round(threshold, 1))
    if (
        rounded["correct_gate_ids"] == original["correct_gate_ids"]
        and rounded["false_gate_ids"] == original["false_gate_ids"]
    ):
        return rounded
    return original


def calibrate(
    samples: tuple[GateSample, ...], *, max_false_gate_rate: float = MAX_FALSE_GATE_RATE
) -> dict[str, Any]:
    if not math.isfinite(max_false_gate_rate) or not 0 <= max_false_gate_rate <= 1:
        raise ValueError("Maximum false gate rate must be between 0 and 1")
    answerable_total = sum(item.answerable for item in samples)
    if not answerable_total or answerable_total == len(samples):
        raise ValueError("Calibration needs both answerable and unanswerable questions")
    max_wrong = math.floor(max_false_gate_rate * answerable_total)
    candidates = [gate_counts(samples, threshold) for threshold in candidate_thresholds(samples)]
    eligible = [row for row in candidates if row["answerable_wrongly_gated"] <= max_wrong]
    selected = max(
        eligible,
        key=lambda row: (
            row["unanswerable_gated"],
            -row["answerable_wrongly_gated"],
            -row["threshold"],
        ),
    )
    balanced = max(
        candidates,
        key=lambda row: (
            row["balanced_accuracy"],
            -row["answerable_wrongly_gated"],
            -row["threshold"],
        ),
    )
    return {
        "policy": {
            "max_false_gate_rate": max_false_gate_rate,
            "max_wrongly_gated_answerable": max_wrong,
            "selection": (
                "Most unanswerable gated within the false-gate cap; "
                "ties favor fewer false gates and a lower threshold"
            ),
        },
        "selected": rounded_operating_point(samples, selected["threshold"]),
        "comparisons": {
            "unset": gate_counts(samples, None),
            "zero": gate_counts(samples, 0.0),
            "best_balanced_accuracy": rounded_operating_point(samples, balanced["threshold"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the reranker answerability gate")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    samples, report = load_samples(args.dataset, args.report)
    result = {
        "schema_version": 1,
        "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "dataset_sha256": report["dataset_sha256"],
        "corpus_fingerprint": report["corpus"]["fingerprint"],
        "retrieval_settings": report["settings"],
        **calibrate(samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = result["selected"]
    print(
        f"Threshold {selected['threshold']}: gated "
        f"{selected['unanswerable_gated']}/{selected['unanswerable_total']} unanswerable, "
        f"{selected['answerable_wrongly_gated']}/{selected['answerable_total']} answerable"
    )
    print(f"Wrote calibration to {args.output}")


if __name__ == "__main__":
    main()
