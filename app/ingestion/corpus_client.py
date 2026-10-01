import time
from collections.abc import Callable
from types import TracebackType

import httpx

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_ATTEMPTS = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class CorpusDownloadError(Exception):
    pass


class CorpusHttpClient:
    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = httpx.Client(
            timeout=DEFAULT_TIMEOUT_SECONDS,
            follow_redirects=True,
            transport=transport,
            headers={"User-Agent": "ask-the-docs/0.1 (+https://github.com/arnabry11/ask-the-docs)"},
        )
        self._sleep = sleep

    def fetch(self, url: str) -> bytes:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._client.get(url)
            except httpx.TransportError as exc:
                if attempt == MAX_ATTEMPTS - 1:
                    raise CorpusDownloadError(f"Could not fetch {url}: {exc}") from exc
            else:
                if response.is_success:
                    if not response.content:
                        raise CorpusDownloadError(f"Empty response from {url}")
                    return response.content
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise CorpusDownloadError(f"HTTP {response.status_code} fetching {url}")
                if attempt == MAX_ATTEMPTS - 1:
                    raise CorpusDownloadError(f"HTTP {response.status_code} fetching {url}")
            self._sleep(0.5 * (2**attempt))
        raise AssertionError("Retry loop exited without a result")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CorpusHttpClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
