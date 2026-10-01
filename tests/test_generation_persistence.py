import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.db.connection import get_engine
from app.generation.cache_key import answer_cache_key
from app.generation.repository import CachedAnswer, GenerationRepository


def test_answer_cache_key_uses_normalized_question_and_exact_context() -> None:
    base = answer_cache_key("  How  do indexes work? ", ["a", "b"], "model", "v1", "passages", 256)

    assert base == answer_cache_key(
        "How do indexes work?", ["a", "b"], "model", "v1", "passages", 256
    )
    assert base != answer_cache_key(
        "How do indexes work?", ["b", "a"], "model", "v1", "passages", 256
    )
    assert base != answer_cache_key(
        "How do indexes work?", ["a", "b"], "model", "v2", "passages", 256
    )
    assert base != answer_cache_key(
        "How do indexes work?", ["a", "b"], "model", "v1", "changed", 256
    )
    assert base != answer_cache_key(
        "How do indexes work?", ["a", "b"], "model", "v1", "passages", 512
    )


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1", reason="requires PostgreSQL with generation tables"
)
def test_cache_saves_verified_answer_once_when_writes_overlap() -> None:
    engine = get_engine()
    repository = GenerationRepository(engine)
    key = uuid4().hex * 2
    answer = CachedAnswer(key, "Use an index [1].", [{"marker": 1}], "test/model", "v1")

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            saved = list(executor.map(repository.save, [answer, answer]))

        assert saved == [answer, answer]
        assert repository.get_cached(key) == answer
    finally:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM llm_cache WHERE key = :key"), {"key": key})
