"""Stream provisional answer tokens and a validated final result."""

import json
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.generation.service import AnswerService, get_answer_service
from app.retrieval.service import MAX_QUESTION_LENGTH

router = APIRouter()


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value


@router.post(
    "/answer",
    response_class=StreamingResponse,
    summary="Stream a cited answer",
    description=(
        "Server-sent events: `progress` reports user-visible stages, `sources` gives the "
        "retrieved passages, `token` carries provisional answer text, and `done` gives the "
        "authoritative final answer or failure status. Use a streaming `fetch()` client; "
        "the interactive API docs show the raw event stream. See `/demo/` for a browser example."
    ),
    responses={200: {"content": {"text/event-stream": {}}}},
)
def answer(
    request: AnswerRequest, service: Annotated[AnswerService, Depends(get_answer_service)]
) -> StreamingResponse:
    def events() -> Iterator[str]:
        for event in service.stream(request.question):
            yield f"event: {event.name}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
