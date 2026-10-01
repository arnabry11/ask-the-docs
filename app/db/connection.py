from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    database_url = make_url(get_settings().database_url).set(drivername="postgresql+psycopg")
    return create_engine(database_url, pool_pre_ping=True)


def database_is_ready() -> bool:
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return False
    return True
