import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(String(50), default="machine_learning")  # "computer_vision" or "machine_learning"
    framework: Mapped[str] = mapped_column(String(50), default="scikit-learn")  # "pytorch_yolo", "onnx", "scikit-learn", "pickle"
    task_type: Mapped[str] = mapped_column(String(50), default="classification")  # object_detection, classification, regression
    file_path: Mapped[str] = mapped_column(String(255), nullable=False)
    features_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list of feature names & types
    target_names_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list of classes
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
