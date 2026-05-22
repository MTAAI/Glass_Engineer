from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://agrocore:agrocore_secret@localhost:5432/agrocore"
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "agrocore"
    minio_secret_key: str = "agrocore_secret"
    minio_bucket: str = "agrocore"

    seed_on_start: bool = True

    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]


settings = Settings()
