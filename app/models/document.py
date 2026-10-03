import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Text, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator
from pgvector.sqlalchemy import Vector
from app.core.config import settings
from app.core.database import Base


class AdaptiveVector(TypeDecorator):
    """
    Dynamically maps to pgvector Vector(dim) when on PostgreSQL with vector extension (e.g. Neon DB),
    and falls back to JSON-serialized Text when on SQLite or vanilla Postgres.
    """
    impl = Text
    cache_ok = True

    def __init__(self, dim: int = 768, *args, **kwargs):
        self.dim = dim
        super().__init__(*args, **kwargs)

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and settings.HAS_PGVECTOR:
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql" and settings.HAS_PGVECTOR:
            return value
        if isinstance(value, (list, tuple)):
            return json.dumps(value)
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql" and settings.HAS_PGVECTOR:
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # 768 dimensions standard for Gemini text-embedding-004 & common modern embeddings
    embedding: Mapped[Optional[List[float]]] = mapped_column(AdaptiveVector(dim=768), nullable=True)

    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    owner = relationship("User", back_populates="documents")
