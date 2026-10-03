import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Afi Template"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Database
    # Support PostgreSQL (Neon DB) or fallback SQLite async for zero-setup local dev
    DATABASE_URL: Optional[str] = None

    # Security
    SECRET_KEY: str = "development-secret-key-please-change-in-production-123456"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day
    COOKIE_NAME: str = "ai_monolith_session"
    COOKIE_SECURE: bool = False
    # Public origin used in generated API documentation/examples, e.g. https://api.example.com
    PUBLIC_BASE_URL: Optional[str] = None

    # AI Provider Keys & OpenAI SDK Compatibility
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None  # Compatible with OpenAI, Groq, DeepSeek, Cloudflare AI, Ollama
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # S3 & Cloudflare R2 Compatible Object Storage
    STORAGE_BACKEND: str = "local"  # "local", "s3", or "r2"
    S3_ENDPOINT_URL: Optional[str] = None  # e.g. https://<account_id>.r2.cloudflarestorage.com
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET_NAME: str = "fastapi-ai-uploads"
    S3_REGION_NAME: str = "auto"
    S3_PUBLIC_DOMAIN: Optional[str] = None  # e.g. https://pub-xxx.r2.dev or CDN

    # CV & Model Config
    WEIGHTS_DIR: str = "weights"
    CV_MODEL_NAME: str = "yolov8n.onnx"
    CV_CONFIDENCE_THRESHOLD: float = 0.35

    # Runtime capability detection
    HAS_PGVECTOR: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def async_database_url(self) -> str:
        """
        Normalizes database URL to async drivers (asyncpg for PostgreSQL, aiosqlite for SQLite).
        Handles Neon DB connection string format automatically.
        """
        if not self.DATABASE_URL or not self.DATABASE_URL.strip():
            # Zero-config fallback to local SQLite for instant testing
            return "sqlite+aiosqlite:///./local.db"

        url = self.DATABASE_URL.strip()

        # Handle postgresql:// or postgres:// from Neon DB dashboard
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

        # Neon DB connection strings often contain sslmode=require; asyncpg prefers ssl=require
        if "sslmode=require" in url:
            url = url.replace("sslmode=require", "ssl=require")

        return url

    @property
    def is_postgres(self) -> bool:
        return "postgres" in self.async_database_url


settings = Settings()

def public_base_url(fallback: str = "") -> str:
    """Return the configured public origin without a trailing slash."""
    return (settings.PUBLIC_BASE_URL or fallback).rstrip("/")
