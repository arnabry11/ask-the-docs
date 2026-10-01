from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from rq import Retry

from app.config import Settings, get_settings
from app.ingestion.queue import RedisIngestionQueue, enqueue_corpus, get_ingestion_queue
from app.ingestion.sources import SourceSpec
from app.main import create_app


class FakeQueue:
    def __init__(self) -> None:
        self.ids: list[str] = []

    def enqueue(self, document_id: str) -> str:
        self.ids.append(document_id)
        return f"job-{len(self.ids)}"


def specs() -> list[SourceSpec]:
    return [
        SourceSpec(
            document_id=f"rails:{index}",
            source="rails",
            version="8.1.4",
            format="markdown",
            fetch_url=f"https://example.test/{index}.md",
            source_url=f"https://example.test/{index}.html",
            relative_path=f"rails/{index}.md",
        )
        for index in range(2)
    ]


def test_enqueuer_selects_one_or_all_catalog_pages() -> None:
    queue = FakeQueue()

    one = enqueue_corpus(queue, "rails:1", specs())
    all_pages = enqueue_corpus(queue, specs=specs())

    assert [(job.document_id, job.job_id) for job in one] == [("rails:1", "job-1")]
    assert [job.document_id for job in all_pages] == ["rails:0", "rails:1"]
    assert queue.ids == ["rails:1", "rails:0", "rails:1"]
    with pytest.raises(ValueError, match="Unknown document ID"):
        enqueue_corpus(queue, "rails:missing", specs())


def test_redis_queue_sets_retry_schedule_and_timeout() -> None:
    class FakeJob:
        id = "queued-job"

    class FakeRqQueue:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def enqueue_call(self, *args: Any, **kwargs: Any) -> FakeJob:
            self.kwargs = kwargs
            return FakeJob()

    queue = RedisIngestionQueue("redis://localhost:6379/0")
    fake = FakeRqQueue()
    queue.queue = fake  # type: ignore[assignment]

    assert queue.enqueue("rails:example") == "queued-job"
    assert fake.kwargs["args"] == ("rails:example",)
    assert fake.kwargs["timeout"] == 600
    retry = fake.kwargs["retry"]
    assert isinstance(retry, Retry)
    assert retry.max == 3
    assert list(retry.intervals) == [10, 30, 60]


@pytest.fixture
def api_client() -> Iterator[tuple[TestClient, FakeQueue]]:
    app = create_app()
    queue = FakeQueue()
    app.dependency_overrides[get_ingestion_queue] = lambda: queue
    app.dependency_overrides[get_settings] = lambda: Settings(ingest_admin_key="private-test-key")
    with TestClient(app) as client:
        yield client, queue
    app.dependency_overrides.clear()


def test_ingest_api_requires_private_key_and_enqueues(
    api_client: tuple[TestClient, FakeQueue],
) -> None:
    client, queue = api_client

    assert client.post("/ingest", json={}).status_code == 401
    assert client.post("/ingest", json={}, headers={"X-Admin-Key": "wrong"}).status_code == 401
    response = client.post(
        "/ingest",
        json={"document_id": "rails:getting_started"},
        headers={"X-Admin-Key": "private-test-key"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "jobs": [{"document_id": "rails:getting_started", "job_id": "job-1"}]
    }
    assert queue.ids == ["rails:getting_started"]
    unknown = client.post(
        "/ingest",
        json={"document_id": "rails:missing"},
        headers={"X-Admin-Key": "private-test-key"},
    )
    assert unknown.status_code == 404


def test_ingest_api_is_disabled_without_admin_key(api_client: tuple[TestClient, FakeQueue]) -> None:
    client, queue = api_client
    client.app.dependency_overrides[get_settings] = lambda: Settings(ingest_admin_key="")

    response = client.post("/ingest", json={})

    assert response.status_code == 503
    assert queue.ids == []
