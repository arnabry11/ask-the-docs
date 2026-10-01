"""Coordinate retrieval, caching, provider streaming, and citation checks."""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.db.connection import get_engine
from app.generation.cache_key import answer_cache_key
from app.generation.citations import check_citations
from app.generation.openrouter_client import OpenRouterClient, OpenRouterError, TextDelta
from app.generation.prompt import PROMPT_VERSION, build_prompt
from app.generation.repository import CachedAnswer, GenerationRepository
from app.retrieval.service import QueryService, get_query_service

SOURCE_SNIPPET_CHARS = 1200


@dataclass(frozen=True)
class AnswerEvent:
    name: str
    data: dict[str, object]


class AnswerService:
    def __init__(
        self,
        query_service: QueryService,
        repository: GenerationRepository,
        client: OpenRouterClient | None,
        *,
        model: str,
        context_chunks: int,
        context_tokens_per_chunk: int,
        max_output_tokens: int,
    ) -> None:
        self.query_service = query_service
        self.repository = repository
        self.client = client
        self.model = model
        self.context_chunks = context_chunks
        self.context_tokens_per_chunk = context_tokens_per_chunk
        self.max_output_tokens = max_output_tokens

    def stream(self, question: str) -> Iterator[AnswerEvent]:
        yield AnswerEvent("progress", {"stage": "searching", "message": "Searching documentation"})
        retrieved = self.query_service.call(question, top_k=self.context_chunks)
        prompt = build_prompt(
            retrieved.question,
            retrieved.chunks,
            context_chunks=self.context_chunks,
            tokens_per_chunk=self.context_tokens_per_chunk,
        )
        sources = [
            {
                "marker": source.marker,
                "chunk_id": source.chunk_id,
                "document_id": source.document_id,
                "title": source.title,
                "section_path": list(source.section_path),
                "source_url": source.source_url,
                "text": item.chunk.text[:SOURCE_SNIPPET_CHARS],
                "rerank_score": item.score,
            }
            for source, item in zip(prompt.sources, retrieved.chunks, strict=True)
        ]
        yield AnswerEvent(
            "sources",
            {
                "question": retrieved.question,
                "sources": sources,
                "gated": retrieved.gate.gated,
                "gate_reason": retrieved.gate.reason,
                "gate_threshold": retrieved.gate.threshold,
            },
        )
        yield AnswerEvent("progress", {"stage": "reviewing", "message": "Reviewing sources"})
        if retrieved.gate.gated:
            yield self._done("gated", answer=retrieved.gate.refusal, refusal=True)
            return
        if not self.model:
            yield self._done("not_configured", degraded=True)
            return

        key = answer_cache_key(
            retrieved.question,
            [source.chunk_id for source in prompt.sources],
            self.model,
            PROMPT_VERSION,
            prompt.context,
            self.max_output_tokens,
        )
        cached = self.repository.get_cached(key)
        if cached is not None:
            yield self._cached(cached)
            return
        if self.client is None:
            yield self._done("not_configured", degraded=True)
            return

        yield AnswerEvent("progress", {"stage": "writing", "message": "Writing answer"})
        parts: list[str] = []
        provider_stream: Iterator[TextDelta] | None = None
        try:
            try:
                provider_stream = self.client.stream(
                    prompt.messages,
                    model=self.model,
                    max_output_tokens=self.max_output_tokens,
                )
                for event in provider_stream:
                    parts.append(event.text)
                    yield AnswerEvent("token", {"text": event.text, "provisional": True})
            except OpenRouterError as exc:
                yield self._done("provider_error", degraded=True, error_code=exc.code)
                return
        finally:
            if provider_stream is not None:
                close_stream = getattr(provider_stream, "close", None)
                if callable(close_stream):
                    close_stream()

        answer = "".join(parts).strip()
        yield AnswerEvent(
            "progress", {"stage": "checking_citations", "message": "Checking citations"}
        )
        checked = check_citations(answer, prompt.sources)
        if not checked.valid:
            yield self._done("invalid_citations", degraded=True, error_code=checked.reason)
            return

        entry = self.repository.save(
            CachedAnswer(
                key=key,
                answer=answer,
                citations=list(checked.citations),
                model=self.model,
                prompt_version=PROMPT_VERSION,
            )
        )
        yield self._done(
            "generated",
            answer=entry.answer,
            citations=entry.citations,
            refusal=not entry.citations,
        )

    @staticmethod
    def _done(
        status: str,
        *,
        answer: str | None = None,
        citations: list[dict[str, object]] | None = None,
        refusal: bool = False,
        degraded: bool = False,
        error_code: str | None = None,
        cached: bool = False,
    ) -> AnswerEvent:
        return AnswerEvent(
            "done",
            {
                "status": status,
                "answer": answer,
                "citations": citations or [],
                "refusal": refusal,
                "cached": cached,
                "degraded": degraded,
                "error_code": error_code,
            },
        )

    @staticmethod
    def _cached(entry: CachedAnswer) -> AnswerEvent:
        return AnswerService._done(
            "cached",
            answer=entry.answer,
            citations=entry.citations,
            refusal=not entry.citations,
            cached=True,
        )


@lru_cache
def get_answer_service() -> AnswerService:
    settings = get_settings()
    client = OpenRouterClient(settings.openrouter_api_key) if settings.openrouter_api_key else None
    return AnswerService(
        get_query_service(),
        GenerationRepository(get_engine()),
        client,
        model=settings.llm_model,
        context_chunks=settings.context_chunks,
        context_tokens_per_chunk=settings.context_tokens_per_chunk,
        max_output_tokens=settings.max_output_tokens,
    )
