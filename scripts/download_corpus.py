import argparse
from pathlib import Path

from app.ingestion.corpus_client import CorpusHttpClient
from app.ingestion.download import download_corpus
from app.ingestion.sources import REPO_ROOT, load_source_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned public documentation corpus")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--refresh", action="store_true", help="Refetch even when files are cached")
    args = parser.parse_args()

    with CorpusHttpClient() as client:
        summary = download_corpus(load_source_specs(), args.output, client, refresh=args.refresh)
    print(f"Downloaded {summary.downloaded}, cached {summary.cached}: {summary.manifest_path}")


if __name__ == "__main__":
    main()
