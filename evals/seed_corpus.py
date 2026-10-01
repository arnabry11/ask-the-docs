import argparse
from pathlib import Path

from app.db.connection import get_engine
from app.ingestion.embed import FastEmbedClient
from app.ingestion.prepare import prepare_corpus
from app.ingestion.repository import IngestionRepository
from app.ingestion.service import IngestDocument
from app.ingestion.sources import REPO_ROOT, SourceSpec, load_source_specs


class CachedCorpusClient:
    def __init__(self, specs: list[SourceSpec], raw_dir: Path) -> None:
        self.paths = {spec.fetch_url: raw_dir / spec.relative_path for spec in specs}

    def fetch(self, url: str) -> bytes:
        if url not in self.paths:
            raise ValueError(f"Unknown corpus URL: {url}")
        return self.paths[url].read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the verified local corpus for evaluation")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    args = parser.parse_args()
    specs = load_source_specs()
    prepared = prepare_corpus(specs, args.raw_dir, REPO_ROOT / "data" / "chunks.jsonl")
    print(f"Verified {prepared.documents} pages and {prepared.chunks} chunks")

    service = IngestDocument(
        CachedCorpusClient(specs, args.raw_dir),
        IngestionRepository(get_engine()),
        FastEmbedClient(),
    )
    for spec in specs:
        result = service.call(spec)
        print(f"{result.document_id}: {result.status} ({result.chunk_count} chunks)")


if __name__ == "__main__":
    main()
