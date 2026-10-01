import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from app.ingestion.sources import load_source_specs

Category = Literal["factual", "how_to", "misleading", "multi_doc", "unanswerable"]
DEFAULT_DATASET = Path(__file__).with_name("dataset.jsonl")
REQUIRED_FIELDS = {"id", "question", "gold_doc_ids", "key_facts", "answerable", "category"}
QUESTION_ID = re.compile(r"Q\d{3}\Z")
VALID_CATEGORIES = {"factual", "how_to", "misleading", "multi_doc", "unanswerable"}


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    gold_doc_ids: tuple[str, ...]
    key_facts: tuple[str, ...]
    answerable: bool
    category: Category


def load_dataset(
    path: Path = DEFAULT_DATASET, *, catalog_ids: set[str] | None = None
) -> tuple[EvalQuestion, ...]:
    if catalog_ids is None:
        catalog_ids = {spec.document_id for spec in load_source_specs()}
    questions: list[EvalQuestion] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {number}") from exc
        if not isinstance(raw, dict) or set(raw) != REQUIRED_FIELDS:
            raise ValueError(f"Invalid fields on line {number}")
        question_id = raw["id"]
        question = raw["question"]
        gold = raw["gold_doc_ids"]
        facts = raw["key_facts"]
        answerable = raw["answerable"]
        category = raw["category"]
        if not isinstance(question_id, str) or not QUESTION_ID.fullmatch(question_id):
            raise ValueError(f"Invalid question ID on line {number}")
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 500:
            raise ValueError(f"Invalid question on line {number}")
        if not isinstance(gold, list) or not all(isinstance(item, str) for item in gold):
            raise ValueError(f"Invalid gold document IDs on line {number}")
        if not isinstance(facts, list) or not all(
            isinstance(item, str) and item.strip() for item in facts
        ):
            raise ValueError(f"Invalid key facts on line {number}")
        if (
            not isinstance(answerable, bool)
            or not isinstance(category, str)
            or category not in VALID_CATEGORIES
        ):
            raise ValueError(f"Invalid answerability or category on line {number}")
        if len(set(gold)) != len(gold) or not set(gold) <= catalog_ids:
            raise ValueError(f"Unknown or repeated gold document on line {number}")
        if len({fact.casefold() for fact in facts}) != len(facts):
            raise ValueError(f"Repeated key fact on line {number}")
        if answerable:
            if not gold or not 2 <= len(facts) <= 3 or category == "unanswerable":
                raise ValueError(f"Invalid answerable record on line {number}")
            if category == "multi_doc" and len(gold) < 2:
                raise ValueError(f"Multi-document record needs two gold documents on line {number}")
        elif gold or facts or category != "unanswerable":
            raise ValueError(f"Invalid unanswerable record on line {number}")

        normalized = " ".join(question.split()).casefold()
        if question_id in seen_ids or normalized in seen_questions:
            raise ValueError(f"Duplicate question ID or text on line {number}")
        seen_ids.add(question_id)
        seen_questions.add(normalized)
        questions.append(
            EvalQuestion(
                id=question_id,
                question=question,
                gold_doc_ids=tuple(gold),
                key_facts=tuple(facts),
                answerable=answerable,
                category=cast(Category, category),
            )
        )

    if not questions:
        raise ValueError("Evaluation dataset is empty")
    return tuple(questions)


def validate_evidence(questions: tuple[EvalQuestion, ...], chunks_path: Path) -> None:
    by_document: dict[str, list[str]] = defaultdict(list)
    for number, line in enumerate(chunks_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            by_document[row["document_id"]].append(row["text"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid prepared chunk on line {number}") from exc

    for item in questions:
        if not item.answerable:
            continue
        missing_docs = [doc_id for doc_id in item.gold_doc_ids if doc_id not in by_document]
        if missing_docs:
            raise ValueError(f"{item.id} lacks prepared chunks for {missing_docs}")
        evidence = " ".join(
            text for doc_id in item.gold_doc_ids for text in by_document[doc_id]
        ).casefold()
        missing_facts = [fact for fact in item.key_facts if fact.casefold() not in evidence]
        if missing_facts:
            raise ValueError(f"{item.id} key facts missing from gold documents: {missing_facts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the retrieval evaluation dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--check-evidence", type=Path, metavar="CHUNKS_JSONL")
    args = parser.parse_args()
    questions = load_dataset(args.dataset)
    if args.check_evidence is not None:
        validate_evidence(questions, args.check_evidence)
    answerable = sum(item.answerable for item in questions)
    print(
        f"Validated {len(questions)} questions: "
        f"{answerable} answerable, {len(questions) - answerable} unanswerable"
    )


if __name__ == "__main__":
    main()
