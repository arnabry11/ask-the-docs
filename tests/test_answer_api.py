import json
from collections.abc import Iterator

from fastapi.testclient import TestClient

from app.generation.service import AnswerEvent, get_answer_service
from app.main import create_app


class FakeAnswerService:
    def stream(self, question: str) -> Iterator[AnswerEvent]:
        yield AnswerEvent("sources", {"question": question, "sources": []})
        yield AnswerEvent("token", {"text": "Answer [1].", "provisional": True})
        yield AnswerEvent("done", {"status": "generated", "answer": "Answer [1]."})


def test_answer_api_emits_sse_and_rejects_blank_input() -> None:
    app = create_app()
    app.dependency_overrides[get_answer_service] = FakeAnswerService
    with TestClient(app) as client:
        response = client.post("/answer", json={"question": "How do indexes work?"})
        invalid = client.post("/answer", json={"question": "   "})
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: sources\n" in response.text
    assert "event: token\n" in response.text
    assert "event: done\n" in response.text
    final = json.loads(response.text.split("event: done\ndata: ")[1].split("\n\n")[0])
    assert final["status"] == "generated"
    assert invalid.status_code == 422
    assert (
        "text/event-stream"
        in app.openapi()["paths"]["/answer"]["post"]["responses"]["200"]["content"]
    )
