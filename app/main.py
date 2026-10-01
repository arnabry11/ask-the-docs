from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.config import get_settings


def create_app() -> FastAPI:
    app = FastAPI(title=get_settings().app_name)
    app.include_router(health_router)
    app.include_router(ingest_router)
    return app


app = create_app()
