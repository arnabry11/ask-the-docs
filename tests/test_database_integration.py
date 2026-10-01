import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db.connection import get_engine
from app.main import app


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1", reason="requires a running PostgreSQL database"
)
def test_health_checks_real_database() -> None:
    with get_engine().connect() as connection:
        extension = connection.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one()

    assert extension == "vector"

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}
