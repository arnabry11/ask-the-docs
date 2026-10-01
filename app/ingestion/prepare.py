import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from app.ingestion.chunk import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS, chunk_document
from app.ingestion.parse import parse_document
from app.ingestion.sources import SourceSpec


@dataclass(frozen=True)
class PrepareSummary:
    documents: int
    sections: int
    chunks: int
    output_path: Path


def prepare_corpus(
    specs: list[SourceSpec],
    input_dir: Path,
    output_path: Path,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> PrepareSummary:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("documents"), list):
        raise ValueError("Unsupported download manifest")
    entries = manifest["documents"]
    if len(entries) != len(specs):
        raise ValueError("Download manifest does not match the source catalog")

    rows: list[str] = []
    sections = 0
    for spec, entry in zip(specs, entries, strict=True):
        expected = asdict(spec)
        if not isinstance(entry, dict) or any(
            entry.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"Download manifest does not match {spec.document_id}")
        raw = (input_dir / spec.relative_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry.get("sha256") or len(raw) != entry.get(
            "byte_count"
        ):
            raise ValueError(f"Downloaded page failed integrity check: {spec.document_id}")
        document = parse_document(spec, raw.decode("utf-8"))
        sections += len(document.sections)
        rows.extend(
            json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True)
            for chunk in chunk_document(
                document, max_tokens=max_tokens, overlap_tokens=overlap_tokens
            )
        )

    payload = ("\n".join(rows) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.is_file() or output_path.read_bytes() != payload:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        try:
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
    return PrepareSummary(len(specs), sections, len(rows), output_path)
