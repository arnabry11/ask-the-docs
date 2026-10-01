import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from app.ingestion.chunk import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS, Chunk, chunk_document
from app.ingestion.constants import EMBEDDING_MODEL, INGESTION_PIPELINE_VERSION
from app.ingestion.download import CorpusClient
from app.ingestion.embed import EmbeddingClient, embedding_text
from app.ingestion.parse import ParsedDocument, parse_document
from app.ingestion.repository import AttemptStatus
from app.ingestion.sources import SourceSpec

LOGGER = logging.getLogger(__name__)


class DocumentStore(Protocol):
    def current_fingerprint(self, document_id: str) -> str | None: ...

    def replace_document(
        self,
        document: ParsedDocument,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        fingerprint: str,
    ) -> None: ...

    def record_attempt(
        self, document_id: str, status: AttemptStatus, detail: str | None = None
    ) -> None: ...


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    status: Literal["completed", "skipped"]
    chunk_count: int | None


class IngestDocument:
    def __init__(
        self,
        client: CorpusClient,
        store: DocumentStore,
        embedder: EmbeddingClient,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ) -> None:
        self.client = client
        self.store = store
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def call(self, spec: SourceSpec) -> IngestionResult:
        try:
            raw = self.client.fetch(spec.fetch_url)
            fingerprint = _fingerprint(
                spec,
                hashlib.sha256(raw).hexdigest(),
                self.max_tokens,
                self.overlap_tokens,
            )
            if self.store.current_fingerprint(spec.document_id) == fingerprint:
                self.store.record_attempt(spec.document_id, "skipped")
                return IngestionResult(spec.document_id, "skipped", None)

            document = parse_document(spec, raw.decode("utf-8"))
            chunks = chunk_document(
                document, max_tokens=self.max_tokens, overlap_tokens=self.overlap_tokens
            )
            embeddings = self.embedder.embed([embedding_text(chunk) for chunk in chunks])
            self.store.replace_document(document, chunks, embeddings, fingerprint)
            self.store.record_attempt(spec.document_id, "completed")
            return IngestionResult(spec.document_id, "completed", len(chunks))
        except Exception as exc:
            try:
                self.store.record_attempt(spec.document_id, "failed", str(exc)[:500])
            except Exception:
                LOGGER.exception("Could not record failed ingestion for %s", spec.document_id)
            raise


def _fingerprint(spec: SourceSpec, content_hash: str, max_tokens: int, overlap_tokens: int) -> str:
    payload = {
        "source": asdict(spec),
        "content_hash": content_hash,
        "embedding_model": EMBEDDING_MODEL,
        "pipeline_version": INGESTION_PIPELINE_VERSION,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
