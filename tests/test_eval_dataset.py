import json
from collections import Counter
from pathlib import Path

import pytest

from evals.dataset import DEFAULT_DATASET, load_dataset, validate_evidence


def test_committed_dataset_has_grounded_questions_and_negative_cases() -> None:
    questions = load_dataset(DEFAULT_DATASET)
    counts = Counter(item.category for item in questions)

    assert len(questions) == 60
    assert sum(item.answerable for item in questions) == 50
    assert counts["unanswerable"] == 10
    assert counts["multi_doc"] >= 3
    assert counts["misleading"] >= 3
    assert {item.gold_doc_ids[0] for item in questions if item.answerable} >= {
        "rails:getting_started",
        "postgresql:indexes-types",
    }


def test_loader_rejects_duplicate_questions_and_unknown_documents(tmp_path: Path) -> None:
    row = {
        "id": "Q001",
        "question": "How do I create an index?",
        "gold_doc_ids": ["postgresql:indexes-intro"],
        "key_facts": ["CREATE INDEX", "table"],
        "answerable": True,
        "category": "how_to",
    }
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "id": "Q002"}) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path, catalog_ids={"postgresql:indexes-intro"})

    path.write_text(json.dumps({**row, "gold_doc_ids": ["missing:doc"]}) + "\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_dataset(path, catalog_ids={"postgresql:indexes-intro"})


def test_evidence_validation_checks_gold_document_text(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "document_id": "postgresql:indexes-intro",
                "text": "CREATE INDEX adds an index to a table.",
            }
        )
        + "\n"
    )
    dataset = tmp_path / "dataset.jsonl"
    row = {
        "id": "Q001",
        "question": "How do I create an index?",
        "gold_doc_ids": ["postgresql:indexes-intro"],
        "key_facts": ["CREATE INDEX", "table"],
        "answerable": True,
        "category": "how_to",
    }
    dataset.write_text(json.dumps(row) + "\n")
    questions = load_dataset(dataset, catalog_ids={"postgresql:indexes-intro"})
    validate_evidence(questions, chunks)

    row["key_facts"] = ["CREATE INDEX", "missing phrase"]
    dataset.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="key facts missing"):
        validate_evidence(load_dataset(dataset, catalog_ids={"postgresql:indexes-intro"}), chunks)
