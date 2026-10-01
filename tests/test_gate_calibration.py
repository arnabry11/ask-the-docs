import json
from pathlib import Path

import pytest

from evals.calibrate_gate import GateSample, calibrate, gate_counts, load_samples
from evals.dataset import DEFAULT_DATASET
from evals.run_retrieval_evals import DEFAULT_OUTPUT as DEFAULT_REPORT


def test_selected_threshold_uses_the_committed_scores_and_false_gate_cap() -> None:
    samples, _ = load_samples(DEFAULT_DATASET, DEFAULT_REPORT)

    result = calibrate(samples)

    assert result["policy"]["max_wrongly_gated_answerable"] == 1
    assert result["selected"]["threshold"] == 1.5
    assert result["selected"]["unanswerable_gated"] == 7
    assert result["selected"]["false_gate_ids"] == ["Q025"]
    assert result["selected"]["unanswerable_not_gated_ids"] == ["Q054", "Q055", "Q058"]
    assert result["comparisons"]["best_balanced_accuracy"]["unanswerable_gated"] == 8
    assert result["comparisons"]["best_balanced_accuracy"]["answerable_wrongly_gated"] == 3


def test_no_sources_always_gate_and_equal_scores_pass() -> None:
    samples = (
        GateSample("answerable", True, 1.5),
        GateSample("unanswerable", False, None),
    )

    assert gate_counts(samples, None)["correct_gate_ids"] == ["unanswerable"]
    assert gate_counts(samples, 1.5)["false_gate_ids"] == []
    assert gate_counts(samples, 1.6)["false_gate_ids"] == ["answerable"]


@pytest.mark.parametrize("field", ["dataset_sha256", "answerable", "best_rerank_score", "id"])
def test_calibration_rejects_report_drift(tmp_path: Path, field: str) -> None:
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    if field == "dataset_sha256":
        report[field] = "0" * 64
    elif field == "answerable":
        report["questions"][0][field] = False
    elif field == "best_rerank_score":
        report["questions"][0][field] = float("nan")
    else:
        report["questions"][0][field] = report["questions"][1][field]
    changed = tmp_path / "retrieval.json"
    changed.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError):
        load_samples(DEFAULT_DATASET, changed)


def test_false_gate_rate_must_be_bounded() -> None:
    samples = (GateSample("a", True, 2.0), GateSample("b", False, 1.0))

    with pytest.raises(ValueError, match="between 0 and 1"):
        calibrate(samples, max_false_gate_rate=float("nan"))
