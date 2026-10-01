import hashlib
import json
from pathlib import Path

import httpx
import pytest

from app.ingestion.corpus_client import CorpusDownloadError, CorpusHttpClient
from app.ingestion.download import download_corpus
from app.ingestion.sources import SourceSpec, load_source_specs


class FakeClient:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def fetch(self, url: str) -> bytes:
        self.calls.append(url)
        return self.responses[url]


def source(document_id: str, path: str) -> SourceSpec:
    return SourceSpec(
        document_id=document_id,
        source="rails",
        version="8.1.4",
        format="markdown",
        fetch_url=f"https://example.test/{path}",
        source_url=f"https://guides.rubyonrails.org/v8.1/{path}",
        relative_path=f"rails/{path}",
    )


def test_source_catalog_uses_pinned_rails_commit_and_postgresql_16() -> None:
    specs = load_source_specs()

    assert len(specs) == 24
    assert len({spec.document_id for spec in specs}) == len(specs)
    pinned_commit = "/c3466ea00d7121798e3aa3144ffdf7174b81d8cb/"
    assert all(pinned_commit in spec.fetch_url for spec in specs[:12])
    assert all("/docs/16/" in spec.fetch_url for spec in specs[12:])


def test_download_is_cached_and_repairs_tampered_file(tmp_path: Path) -> None:
    specs = [source("rails:first", "first.md"), source("rails:second", "second.md")]
    client = FakeClient({specs[0].fetch_url: b"# First\n", specs[1].fetch_url: b"# Second\n"})

    first = download_corpus(specs, tmp_path, client)
    manifest_before = first.manifest_path.read_bytes()
    second = download_corpus(specs, tmp_path, client)

    assert (first.downloaded, first.cached) == (2, 0)
    assert (second.downloaded, second.cached) == (0, 2)
    assert client.calls == [spec.fetch_url for spec in specs]
    assert second.manifest_path.read_bytes() == manifest_before

    (tmp_path / specs[0].relative_path).write_text("tampered", encoding="utf-8")
    repaired = download_corpus(specs, tmp_path, client)
    manifest = json.loads(repaired.manifest_path.read_text(encoding="utf-8"))

    assert (repaired.downloaded, repaired.cached) == (1, 1)
    assert client.calls[-1] == specs[0].fetch_url
    assert manifest["documents"][0]["sha256"] == hashlib.sha256(b"# First\n").hexdigest()


def test_http_client_retries_transient_response() -> None:
    statuses = iter([503, 200])
    delays: list[float] = []
    transport = httpx.MockTransport(lambda _: httpx.Response(next(statuses), content=b"ok"))

    with CorpusHttpClient(transport=transport, sleep=delays.append) as client:
        assert client.fetch("https://example.test/doc") == b"ok"

    assert delays == [0.5]


def test_http_client_fails_fast_on_missing_page() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(404))

    with CorpusHttpClient(transport=transport) as client:
        with pytest.raises(CorpusDownloadError, match="HTTP 404"):
            client.fetch("https://example.test/missing")
