import json

import httpx
import pytest

from app.generation.openrouter_client import OpenRouterClient, OpenRouterError, TextDelta


def sse(*payloads: str) -> str:
    return "".join(f"data: {payload}\n\n" for payload in payloads)


def test_client_streams_text_with_bounded_request() -> None:
    captured: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            text=sse(
                '{"choices":[{"delta":{"content":"Use "}}]}',
                '{"choices":[{"delta":{"content":"indexes [1]."}}]}',
                "[DONE]",
            ),
        )

    client = OpenRouterClient("test-key", transport=httpx.MockTransport(handle))
    events = list(
        client.stream(
            [{"role": "user", "content": "question"}], model="test/model", max_output_tokens=256
        )
    )

    assert events == [TextDelta("Use "), TextDelta("indexes [1].")]
    request = captured[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["temperature"] == 0
    assert body["stream"] is True
    assert body["max_tokens"] == 256
    assert "provider" not in body
    assert "stream_options" not in body


def test_client_retries_a_pre_stream_rate_limit_only() -> None:
    attempts: list[int] = []
    waits: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, text=sse('{"choices":[]}', "[DONE]"))

    client = OpenRouterClient("test-key", transport=httpx.MockTransport(handle), sleep=waits.append)
    events = list(
        client.stream(
            [{"role": "user", "content": "question"}], model="test/model", max_output_tokens=10
        )
    )

    assert len(attempts) == 2
    assert waits == [1]
    assert events == []


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (sse('{"choices":[{"delta":{"content":"partial"}}]}'), "truncated_stream"),
        (sse('{"error":{"message":"failed"}}', "[DONE]"), "stream_error"),
        (sse('{"choices":[{"delta":{"content":42}}]}', "[DONE]"), "invalid_stream_event"),
    ],
)
def test_client_rejects_incomplete_or_invalid_stream_without_retry(body: str, code: str) -> None:
    attempts: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(200, text=body)

    client = OpenRouterClient("test-key", transport=httpx.MockTransport(handle))
    with pytest.raises(OpenRouterError) as error:
        list(
            client.stream(
                [{"role": "user", "content": "question"}], model="test/model", max_output_tokens=10
            )
        )
    assert error.value.code == code
    assert len(attempts) == 1
