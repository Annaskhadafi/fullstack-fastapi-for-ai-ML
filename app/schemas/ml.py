from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class FeatureDefinition(BaseModel):
    name: str
    label: str
    type: str = "number"
    default: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: Optional[List[str]] = None


class ModelInfo(BaseModel):
    id: str
    name: str
    display_name: str
    description: Optional[str] = None
    task_type: str
    features: List[FeatureDefinition]
    target_names: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None


class PredictRequest(BaseModel):
    features: Union[Dict[str, Any], List[Any]] = Field(..., description="Named or ordered feature inputs")


class PredictResponse(BaseModel):
    success: bool
    model_name: str
    prediction: Any
    prediction_label: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    execution_time_ms: float
