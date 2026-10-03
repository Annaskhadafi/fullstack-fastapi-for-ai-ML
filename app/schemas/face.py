from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RegisteredFaceItem(BaseModel):
    id: str
    name: str
    identity_number: Optional[str] = None
    notes: Optional[str] = None
    photo_url: str
    created_at: Optional[str] = None
    source: Optional[str] = "database"


class FaceListResponse(BaseModel):
    success: bool = True
    total: int
    faces: List[RegisteredFaceItem]


class FaceRegisterResponse(BaseModel):
    success: bool = True
    id: str
    name: str
    identity_number: Optional[str] = None
    photo_url: str
    embedding_dim: int
    db_synced: bool
    message: str = "Wajah berhasil didaftarkan ke sistem biometrik"


class RecognizedFaceDetail(BaseModel):
    face_index: int
    status: str = Field(..., description="'REGISTERED', 'UNKNOWN', atau 'SPOOF_ATTACK'")
    is_real: bool = Field(..., description="True jika manusia asli (Live), False jika serangan spoofing (layar HP/foto)")
    liveness_score: float = Field(..., description="Skor liveness anti-spoofing (0.0 - 1.0)")
    is_registered: bool
    name: str
    identity_number: Optional[str] = None
    similarity: float = Field(..., description="Cosine similarity score (0.0 - 1.0)")
    confidence: float
    age: Optional[int] = None
    gender: Optional[str] = None
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2]")


class FaceRecognizeResponse(BaseModel):
    success: bool = True
    total_faces: int
    total_registered_matched: int
    total_unknown: int
    total_spoof_attacks: int = 0
    faces: List[RecognizedFaceDetail]
    annotated_image_base64: Optional[str] = None
    execution_time_ms: float
