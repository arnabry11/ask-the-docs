from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from app.retrieval.service import (
    DEFAULT_TOP_K,
    MAX_QUESTION_LENGTH,
    QueryService,
    get_query_service,
)
from app.retrieval.vector_repository import MAX_QUERY_RESULTS

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_QUERY_RESULTS)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    version: str
    title: str
    section_path: tuple[str, ...]
    source_url: str
    text: str
    token_count: int
    rrf_score: float
    vector_rank: int | None
    keyword_rank: int | None
    cosine_distance: float | None
    fts_rank: float | None


class QueryResponse(BaseModel):
    question: str
    retrieval: str = "hybrid"
    results: list[SourceChunk]


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest, service: Annotated[QueryService, Depends(get_query_service)]
) -> dict[str, object]:
    result = service.call(request.question, request.top_k)
    return {
        "question": result.question,
        "retrieval": "hybrid",
        "results": [asdict(chunk) for chunk in result.chunks],
    }
