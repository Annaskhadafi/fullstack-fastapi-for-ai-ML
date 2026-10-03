from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=10)


class DocumentResponse(BaseModel):
    id: str
    title: str
    content_snippet: str
    created_at: datetime

    class Config:
        from_attributes = True


class RetrievedChunk(BaseModel):
    document_id: str
    title: str
    content: str
    similarity_score: float


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(default=3, ge=1, le=10)


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    model: str
    execution_time_ms: float
