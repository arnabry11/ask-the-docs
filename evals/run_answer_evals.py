"""Compare marker validation with manually judged answer support, without model calls."""

import argparse
import json
from pathlib import Path

from app.generation.citations import check_citations
from app.generation.prompt import PromptSource

DEFAULT_CASES = Path(__file__).with_name("answer_cases.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = parser.parse_args()

    rows = []
    ids: set[str] = set()
    for line in args.cases.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        case_id = case["id"]
        if case_id in ids:
            raise ValueError(f"Duplicate answer case: {case_id}")
        ids.add(case_id)
        sources = tuple(
            PromptSource(
                marker=source["marker"],
                chunk_id=source["chunk_id"],
                document_id=source["document_id"],
                title=source["title"],
                section_path=tuple(source["section_path"]),
                source_url=source["source_url"],
            )
            for source in case["sources"]
        )
        check = check_citations(case["answer"], sources)
        rows.append(
            (
                case_id,
                case["origin"],
                check.valid,
                case["human_supported"],
                case["should_refuse"],
                check.refusal,
            )
        )

    print("case | origin | markers valid | human supported | should refuse | refused")
    print("--- | --- | --- | --- | --- | ---")
    for case_id, origin, valid, supported, should_refuse, refusal in rows:
        print(f"{case_id} | {origin} | {valid} | {supported} | {should_refuse} | {refusal}")
    missed = sum(valid and not supported for _, _, valid, supported, _, _ in rows)
    print(f"Unsupported answers accepted by marker check: {missed}/{len(rows)}")


if __name__ == "__main__":
    main()
