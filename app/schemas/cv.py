from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionResult(BaseModel):
    success: bool
    total_detected: int
    classes_summary: dict[str, int]
    boxes: List[BoundingBox]
    execution_time_ms: float
    annotated_image_base64: Optional[str] = None
    original_dimensions: dict[str, int]


class CVFilterRequest(BaseModel):
    filter_type: str = Field(
        ...,
        description="Filter type: grayscale, canny_edge, gaussian_blur, threshold, contours, face_detect"
    )
    canny_thresh1: Optional[int] = 100
    canny_thresh2: Optional[int] = 200
    blur_kernel: Optional[int] = 5
