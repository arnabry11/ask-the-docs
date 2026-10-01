"""OpenRouter streaming chat client; all provider HTTP stays here."""

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUSES = {429, 500, 502, 503, 504, 529}


class OpenRouterError(Exception):
    def __init__(self, code: str, status_code: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class TextDelta:
    text: str


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 90,
        retries: int = 2,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip() or timeout_seconds <= 0 or not 0 <= retries <= 3:
            raise ValueError("OpenRouter client needs a key, timeout, and limited retries")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.transport = transport
        self.sleep = sleep

    def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> Iterator[TextDelta]:
        if not messages or not model or max_output_tokens < 1:
            raise ValueError("OpenRouter request needs messages, model, and output limit")
        request = {
            "model": model,
            "messages": list(messages),
            "temperature": 0,
            "max_tokens": max_output_tokens,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as http:
            for attempt in range(self.retries + 1):
                try:
                    with http.stream(
                        "POST", "/chat/completions", json=request, headers=headers
                    ) as response:
                        if response.status_code in RETRYABLE_STATUSES and attempt < self.retries:
                            self.sleep(2**attempt)
                            continue
                        if response.status_code != 200:
                            raise OpenRouterError(
                                f"http_{response.status_code}", response.status_code
                            )
                        yield from self._events(response)
                        return
                except httpx.ConnectError as exc:
                    if attempt < self.retries:
                        self.sleep(2**attempt)
                        continue
                    raise OpenRouterError("connection_failed") from exc
                except httpx.TimeoutException as exc:
                    # A read timeout may occur after work begins; retrying could bill twice.
                    raise OpenRouterError("timeout") from exc
                except httpx.TransportError as exc:
                    raise OpenRouterError("transport_failed") from exc

    @staticmethod
    def _events(response: httpx.Response) -> Iterator[TextDelta]:
        done = False
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            try:
                payload: dict[str, Any] = json.loads(data)
            except (ValueError, TypeError) as exc:
                raise OpenRouterError("invalid_stream_event") from exc
            if not isinstance(payload, dict):
                raise OpenRouterError("invalid_stream_event")
            if payload.get("error"):
                raise OpenRouterError("stream_error")
            choices = payload.get("choices") or []
            if not isinstance(choices, list):
                raise OpenRouterError("invalid_stream_event")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise OpenRouterError("invalid_stream_event")
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    raise OpenRouterError("invalid_stream_event")
                content = delta.get("content")
                if content is not None and not isinstance(content, str):
                    raise OpenRouterError("invalid_stream_event")
                if isinstance(content, str) and content:
                    yield TextDelta(content)
        if not done:
            raise OpenRouterError("truncated_stream")
