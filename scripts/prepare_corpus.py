import argparse
from pathlib import Path

from app.ingestion.chunk import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS
from app.ingestion.prepare import prepare_corpus
from app.ingestion.sources import REPO_ROOT, load_source_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and chunk the downloaded documentation")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "chunks.jsonl")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    args = parser.parse_args()

    summary = prepare_corpus(
        load_source_specs(),
        args.input,
        args.output,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    print(
        f"Prepared {summary.documents} documents, {summary.sections} sections, "
        f"{summary.chunks} chunks: {summary.output_path}"
    )


if __name__ == "__main__":
    main()
