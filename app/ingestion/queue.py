from dataclasses import dataclass
from typing import Protocol

from redis import Redis
from rq import Queue, Retry

from app.config import get_settings
from app.ingestion.sources import SourceSpec, load_source_specs

QUEUE_NAME = "ingestion"
JOB_TIMEOUT_SECONDS = 600
RETRY_INTERVALS_SECONDS = [10, 30, 60]


class JobQueue(Protocol):
    def enqueue(self, document_id: str) -> str: ...


@dataclass(frozen=True)
class EnqueuedDocument:
    document_id: str
    job_id: str


class RedisIngestionQueue:
    def __init__(self, redis_url: str) -> None:
        self.queue = Queue(QUEUE_NAME, connection=Redis.from_url(redis_url))

    def enqueue(self, document_id: str) -> str:
        job = self.queue.enqueue_call(
            "app.ingestion.jobs.ingest_document_job",
            args=(document_id,),
            timeout=JOB_TIMEOUT_SECONDS,
            retry=Retry(max=3, interval=RETRY_INTERVALS_SECONDS),
            failure_ttl=86400,
        )
        return job.id


def get_ingestion_queue() -> RedisIngestionQueue:
    return RedisIngestionQueue(get_settings().redis_url)


def enqueue_corpus(
    queue: JobQueue, document_id: str | None = None, specs: list[SourceSpec] | None = None
) -> list[EnqueuedDocument]:
    catalog = specs if specs is not None else load_source_specs()
    selected = [spec for spec in catalog if document_id is None or spec.document_id == document_id]
    if document_id is not None and not selected:
        raise ValueError(f"Unknown document ID: {document_id}")
    return [
        EnqueuedDocument(spec.document_id, queue.enqueue(spec.document_id)) for spec in selected
    ]
