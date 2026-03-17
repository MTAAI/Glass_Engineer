"""
Glass Expert AI — Centralised Settings
All configuration loaded once via pydantic-settings.
Never use os.getenv() in routers — import settings instead.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────
    app_name: str = "Glass Expert AI"
    app_version: str = "3.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    debug: bool = False
    environment: str = "development"

    # ── Database ───────────────────────────────────────────────────────────
    database_url: str = Field(
        "postgresql://glassai:glassai_secret@localhost:5432/glass_expert_ai"
    )
    db_pool_min_size: int = 2
    db_pool_max_size: int = 20

    # ── Redis ──────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 3600

    # ── JWT Auth ───────────────────────────────────────────────────────────
    jwt_secret_key: str = Field("", alias="JWT_SECRET_KEY")
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480
    refresh_token_expire_days: int = 7

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_use_fp16: bool = True
    embedding_batch_size: int = 64
    embedding_max_length: int = 512

    # ── Reranker ───────────────────────────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"
    rerank_enabled: bool = True
    rerank_top_n: int = 5

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieval_top_k: int = 30
    retrieval_final_top_k: int = 8
    similarity_threshold: float = 0.30

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "glass-expert"
    llm_api_key: str = "token-glass-ai"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # ── Fallback LLM ──────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    def get_secret_key(self) -> str:
        """Return whichever secret key is set."""
        return self.jwt_secret_key or self.secret_key or ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()