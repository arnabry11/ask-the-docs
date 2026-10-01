from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Ask the Docs"
    database_url: str = "postgresql://ask:ask@localhost:5432/ask_the_docs?sslmode=disable"
    redis_url: str = "redis://localhost:6379/0"
    cors_origins: str = ""
    ingest_admin_key: str = ""
    top_k_vector: int = Field(default=30, ge=1, le=30)
    top_k_fts: int = Field(default=30, ge=1, le=30)
    rrf_k: int = Field(default=60, ge=1)
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = Field(default=20, ge=1, le=20)
    gate_threshold: float | None = Field(default=1.5, allow_inf_nan=False)
    openrouter_api_key: str = ""
    llm_model: str = ""
    context_chunks: int = Field(default=4, ge=1, le=5)
    context_tokens_per_chunk: int = Field(default=300, ge=1, le=400)
    max_output_tokens: int = Field(default=512, ge=1, le=2048)

    @field_validator("gate_threshold", mode="before")
    @classmethod
    def allow_disabled_gate(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() == "off":
            return None
        return value

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
