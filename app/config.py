from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Ask the Docs"
    database_url: str = "postgresql://ask:ask@localhost:5432/ask_the_docs?sslmode=disable"
    redis_url: str = "redis://localhost:6379/0"
    ingest_admin_key: str = ""
    top_k_vector: int = Field(default=30, ge=1, le=30)
    top_k_fts: int = Field(default=30, ge=1, le=30)
    rrf_k: int = Field(default=60, ge=1)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
