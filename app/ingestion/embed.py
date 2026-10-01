import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from fastembed import TextEmbedding

from app.ingestion.chunk import Chunk
from app.ingestion.constants import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from app.ingestion.sources import REPO_ROOT

DEFAULT_MODEL_CACHE = REPO_ROOT / "models"
EMBED_BATCH_SIZE = 32


class EmbeddingClient(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class QueryEmbeddingClient(Protocol):
    def embed_query(self, question: str) -> list[float]: ...


def embedding_text(chunk: Chunk) -> str:
    path = " > ".join(chunk.section_path)
    if not chunk.section_path or chunk.section_path[0] != chunk.title:
        path = f"{chunk.title} > {path}"
    return f"passage: {path}\n{chunk.text}"


class FastEmbedClient:
    def __init__(self, cache_dir: Path = DEFAULT_MODEL_CACHE) -> None:
        self.model = TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=str(cache_dir))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = [
            vector.tolist() for vector in self.model.embed(texts, batch_size=EMBED_BATCH_SIZE)
        ]
        if len(vectors) != len(texts) or any(
            len(vector) != EMBEDDING_DIMENSIONS for vector in vectors
        ):
            raise ValueError(f"Embedding model must return {EMBEDDING_DIMENSIONS} values per text")
        return vectors

    def embed_query(self, question: str) -> list[float]:
        vectors = [
            [float(value) for value in vector] for vector in self.model.query_embed(question)
        ]
        if (
            len(vectors) != 1
            or len(vectors[0]) != EMBEDDING_DIMENSIONS
            or not all(math.isfinite(value) for value in vectors[0])
        ):
            raise ValueError(f"Query embedding must contain {EMBEDDING_DIMENSIONS} finite values")
        return vectors[0]
