import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.connection import get_engine
from app.db.models import DocumentRecord
from app.ingestion.constants import EMBEDDING_MODEL
from app.ingestion.embed import FastEmbedClient, QueryEmbeddingClient
from app.ingestion.sources import REPO_ROOT, load_source_specs
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.gate import decide_gate
from app.retrieval.keyword_repository import KeywordSearchRepository
from app.retrieval.rerank import FastEmbedReranker, RerankService
from app.retrieval.service import KeywordSearcher, VectorSearcher
from app.retrieval.vector_repository import VectorSearchRepository
from evals.dataset import DEFAULT_DATASET, EvalQuestion, load_dataset
from evals.metrics import EvalOutcome, Mode, evaluate_metrics

DEFAULT_OUTPUT = REPO_ROOT / "evals" / "results" / "retrieval.json"


class RetrievalEvaluator:
    def __init__(
        self,
        embedder: QueryEmbeddingClient,
        vector_searcher: VectorSearcher,
        keyword_searcher: KeywordSearcher,
        reranker: RerankService,
        *,
        vector_limit: int,
        keyword_limit: int,
        rerank_top_n: int,
        rrf_k: int,
        gate_threshold: float | None,
    ) -> None:
        self.embedder = embedder
        self.vector_searcher = vector_searcher
        self.keyword_searcher = keyword_searcher
        self.reranker = reranker
        self.vector_limit = vector_limit
        self.keyword_limit = keyword_limit
        self.rerank_top_n = rerank_top_n
        self.rrf_k = rrf_k
        self.gate_threshold = gate_threshold

    def call(self, item: EvalQuestion) -> EvalOutcome:
        question = " ".join(item.question.split())
        vector = self.embedder.embed_query(question)
        vector_results = self.vector_searcher.search(vector, self.vector_limit)
        keyword_results = self.keyword_searcher.search(question, self.keyword_limit)
        fused = reciprocal_rank_fusion(
            vector_results, keyword_results, self.rerank_top_n, rrf_k=self.rrf_k
        )
        reranked = self.reranker.call(question, fused, self.rerank_top_n)
        gate = decide_gate(reranked, self.gate_threshold)
        rankings: dict[Mode, tuple[str, ...]] = {
            "vector": tuple(chunk.document_id for chunk in vector_results),
            "keyword": tuple(chunk.document_id for chunk in keyword_results),
            "hybrid": tuple(chunk.document_id for chunk in fused),
            "hybrid_rerank": tuple(result.chunk.document_id for result in reranked),
        }
        return EvalOutcome(
            question_id=item.id,
            rankings=rankings,
            gated=gate.gated,
            gate_reason=gate.reason,
            best_rerank_score=reranked[0].score if reranked else None,
        )


def corpus_metadata(engine: Engine) -> dict[str, object]:
    expected = {spec.document_id for spec in load_source_specs()}
    with Session(engine) as session:
        rows = session.execute(
            select(
                DocumentRecord.id,
                DocumentRecord.version,
                DocumentRecord.content_hash,
                DocumentRecord.ingestion_fingerprint,
                DocumentRecord.chunk_count,
            ).order_by(DocumentRecord.id)
        ).all()
    if {row.id for row in rows} != expected or any(row.chunk_count < 1 for row in rows):
        raise ValueError("Evaluation database must contain exactly the 24 nonempty catalog pages")
    manifest = [
        {
            "id": row.id,
            "version": row.version,
            "content_hash": row.content_hash,
            "ingestion_fingerprint": row.ingestion_fingerprint,
            "chunk_count": row.chunk_count,
        }
        for row in rows
    ]
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {
        "document_count": len(rows),
        "chunk_count": sum(row.chunk_count for row in rows),
        "fingerprint": hashlib.sha256(serialized).hexdigest(),
        "documents": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local retrieval evaluations")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    questions = load_dataset(args.dataset)
    settings = get_settings()
    engine = get_engine()
    corpus = corpus_metadata(engine)
    evaluator = RetrievalEvaluator(
        FastEmbedClient(),
        VectorSearchRepository(engine),
        KeywordSearchRepository(engine),
        RerankService(FastEmbedReranker(model_name=settings.rerank_model)),
        vector_limit=settings.top_k_vector,
        keyword_limit=settings.top_k_fts,
        rerank_top_n=settings.rerank_top_n,
        rrf_k=settings.rrf_k,
        gate_threshold=settings.gate_threshold,
    )
    outcomes = tuple(evaluator.call(item) for item in questions)
    summary = evaluate_metrics(questions, outcomes)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "corpus": corpus,
        "settings": {
            "embedding_model": EMBEDDING_MODEL,
            "rerank_model": settings.rerank_model,
            "top_k_vector": settings.top_k_vector,
            "top_k_fts": settings.top_k_fts,
            "rrf_k": settings.rrf_k,
            "rerank_top_n": settings.rerank_top_n,
            "gate_threshold": settings.gate_threshold,
        },
        "summary": summary,
        "questions": [
            {
                "id": item.id,
                "question": item.question,
                "category": item.category,
                "answerable": item.answerable,
                "gold_doc_ids": item.gold_doc_ids,
                "rankings": outcome.rankings,
                "gated": outcome.gated,
                "gate_reason": outcome.gate_reason,
                "best_rerank_score": outcome.best_rerank_score,
            }
            for item, outcome in zip(questions, outcomes, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(questions)} evaluations to {args.output}")
    variants = cast(dict[Mode, dict[str, float]], summary["variants"])
    for mode, metrics in variants.items():
        print(f"{mode}: hit@5={metrics['hit_rate@5']:.3f} mrr@5={metrics['mrr@5']:.3f}")


if __name__ == "__main__":
    main()
