from fastapi.testclient import TestClient

from app.db.connection import database_is_ready
from app.main import app


def test_health_returns_ok() -> None:
    app.dependency_overrides[database_is_ready] = lambda: True
    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_returns_unavailable_when_database_is_down() -> None:
    app.dependency_overrides[database_is_ready] = lambda: False
    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": "unavailable"}
