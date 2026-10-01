from dataclasses import asdict
from secrets import compare_digest
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from redis.exceptions import RedisError

from app.config import Settings, get_settings
from app.ingestion.queue import RedisIngestionQueue, enqueue_corpus, get_ingestion_queue

router = APIRouter()


class IngestRequest(BaseModel):
    document_id: str | None = None


@router.post("/ingest", status_code=202)
def ingest(
    request: IngestRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    queue: Annotated[RedisIngestionQueue, Depends(get_ingestion_queue)],
    admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
) -> dict[str, list[dict[str, str]]]:
    if not settings.ingest_admin_key:
        raise HTTPException(status_code=503, detail="Ingestion admin key is not configured")
    if admin_key is None or not compare_digest(admin_key, settings.ingest_admin_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")
    try:
        jobs = enqueue_corpus(queue, request.document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="Ingestion queue is unavailable") from exc
    return {"jobs": [asdict(job) for job in jobs]}
