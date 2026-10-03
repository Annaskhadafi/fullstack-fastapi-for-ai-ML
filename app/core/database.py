import logging
import uuid
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

logger = logging.getLogger(__name__)

# Engine configuration with pool_pre_ping for Neon DB serverless resilience
connect_args = {}
if not settings.is_postgres:
    connect_args["check_same_thread"] = False

engine = create_async_engine(
    settings.async_database_url,
    echo=False,
    pool_pre_ping=True,  # Crucial for Neon DB serverless cold starts
    connect_args=connect_args
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for obtaining an asynchronous DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initializes database schema.
    Safely tests and enables pgvector on Postgres in an isolated connection
    to prevent transaction aborts, then generates all registered tables.
    """
    if settings.is_postgres:
        try:
            async with engine.connect() as conn:
                await conn.execution_options(isolation_level="AUTOCOMMIT")
                # Check if 'vector' is available on this PostgreSQL server (like Neon DB)
                check_res = await conn.execute(
                    text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector';")
                )
                if check_res.scalar():
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    settings.HAS_PGVECTOR = True
                    logger.info("Ekstensi 'vector' aktif di PostgreSQL (pgvector mode aktif).")
                else:
                    settings.HAS_PGVECTOR = False
                    logger.info("Ekstensi 'vector' tidak tersedia di Postgres ini. Menggunakan adaptive text vector mode.")
        except Exception as e:
            settings.HAS_PGVECTOR = False
            logger.warning(f"Gagal mengaktifkan ekstensi 'vector': {e}. Fallback ke adaptive vector storage.")

    # Run table creation in a clean transaction
    async with engine.begin() as conn:
        from app.models.user import User  # noqa: F401
        from app.models.api_key import ApiKey  # noqa: F401
        from app.models.document import Document  # noqa: F401
        from app.models.ml_model import MLModel  # noqa: F401
        from app.models.face import RegisteredFace  # noqa: F401

        await conn.run_sync(Base.metadata.create_all)

        # Backfill the existing single key without invalidating it.
        users = (await conn.execute(text("SELECT id, api_key FROM users WHERE api_key IS NOT NULL"))).all()
        for user_id, api_key in users:
            exists = await conn.scalar(text("SELECT 1 FROM api_keys WHERE api_key = :api_key"), {"api_key": api_key})
            if not exists:
                await conn.execute(text(
                    "INSERT INTO api_keys (id, user_id, api_key, name, created_at) "
                    "VALUES (:id, :user_id, :api_key, 'Legacy', CURRENT_TIMESTAMP)"
                ), {"id": str(uuid.uuid4()), "user_id": user_id, "api_key": api_key})
        
        # Safely migrate newly added columns in Postgres if tables existed previously
        if settings.is_postgres:
            try:
                await conn.execute(text("ALTER TABLE ml_models ALTER COLUMN features_json DROP NOT NULL;"))
                await conn.execute(text("ALTER TABLE ml_models ALTER COLUMN target_names_json DROP NOT NULL;"))
                await conn.execute(text("ALTER TABLE ml_models ALTER COLUMN metrics_json DROP NOT NULL;"))
                await conn.execute(text("ALTER TABLE ml_models ALTER COLUMN task_type DROP NOT NULL;"))
                await conn.execute(text("ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS display_name VARCHAR(255) DEFAULT '';"))
                await conn.execute(text("ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS category VARCHAR(50) DEFAULT 'machine_learning'"))
                await conn.execute(text("ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS framework VARCHAR(50) DEFAULT 'scikit-learn'"))
            except Exception as e:
                logger.warning(f"Could not auto-migrate ml_models columns: {e}")

        logger.info("Database tables initialized successfully.")
