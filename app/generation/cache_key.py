import hashlib
import json
from collections.abc import Sequence


def answer_cache_key(
    question: str,
    chunk_ids: Sequence[str],
    model: str,
    prompt_version: str,
    context: str,
    max_output_tokens: int,
) -> str:
    normalized = " ".join(question.split())
    if not normalized or not chunk_ids or not model or not prompt_version or max_output_tokens < 1:
        raise ValueError("Answer cache key requires a question, context, model, and output limit")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Context chunk IDs must be unique")
    payload = {
        "question": normalized,
        "chunk_ids": list(chunk_ids),
        "model": model,
        "prompt_version": prompt_version,
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "max_output_tokens": max_output_tokens,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
