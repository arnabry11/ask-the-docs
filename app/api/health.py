from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.db.connection import database_is_ready

router = APIRouter()


@router.get("/health")
def health(database_ready: Annotated[bool, Depends(database_is_ready)]) -> JSONResponse:
    if not database_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "database": "unavailable"},
        )
    return JSONResponse(content={"status": "ok", "database": "ok"})
