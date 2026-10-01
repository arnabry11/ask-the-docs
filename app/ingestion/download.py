import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

from app.ingestion.sources import SourceSpec

MANIFEST_VERSION = 1


class CorpusClient(Protocol):
    def fetch(self, url: str) -> bytes: ...


@dataclass(frozen=True)
class DownloadSummary:
    downloaded: int
    cached: int
    manifest_path: Path


def download_corpus(
    specs: list[SourceSpec], output_dir: Path, client: CorpusClient, *, refresh: bool = False
) -> DownloadSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    previous = _read_manifest(manifest_path)
    previous_by_id = {entry["document_id"]: entry for entry in previous}
    documents: list[dict[str, object]] = []
    downloaded = cached = 0

    for spec in specs:
        path = output_dir / spec.relative_path
        expected = asdict(spec)
        old = previous_by_id.get(spec.document_id)
        if not refresh and old is not None and _is_cached(path, expected, old):
            documents.append(old)
            cached += 1
            continue

        content = client.fetch(spec.fetch_url)
        content.decode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        if not path.is_file() or path.read_bytes() != content:
            _atomic_write(path, content)
        documents.append({**expected, "sha256": digest, "byte_count": len(content)})
        downloaded += 1

    payload = {"schema_version": MANIFEST_VERSION, "documents": documents}
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if not manifest_path.is_file() or manifest_path.read_bytes() != serialized:
        _atomic_write(manifest_path, serialized)
    return DownloadSummary(downloaded=downloaded, cached=cached, manifest_path=manifest_path)


def _read_manifest(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != MANIFEST_VERSION:
        raise ValueError("Unsupported download manifest version")
    documents = payload["documents"]
    if not isinstance(documents, list) or any(not isinstance(item, dict) for item in documents):
        raise ValueError("Invalid download manifest")
    return cast(list[dict[str, object]], documents)


def _is_cached(path: Path, expected: dict[str, object], old: dict[str, object]) -> bool:
    return (
        path.is_file()
        and all(old.get(key) == value for key, value in expected.items())
        and hashlib.sha256(path.read_bytes()).hexdigest() == old.get("sha256")
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
