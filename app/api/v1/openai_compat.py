import time
import uuid
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from app.models.user import User
from app.services.auth_service import get_current_user_optional
from app.services.rag_service import get_openai_client, get_embedding
from app.core.config import settings

router = APIRouter(prefix="", tags=["OpenAI SDK Compatible Endpoints"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "gpt-4o-mini"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


class EmbeddingRequest(BaseModel):
    model: Optional[str] = "text-embedding-3-small"
    input: Union[str, List[str]]


@router.post("/chat/completions")
async def openai_chat_completions(
    payload: ChatCompletionRequest,
    user: User = Depends(get_current_user_optional)
):
    """
    OpenAI-Compatible Chat Completions Endpoint.
    Can proxy to OpenAI/Groq/Cloudflare/Ollama or generate local responses.
    """
    client = get_openai_client()
    if client:
        try:
            raw_messages = [{"role": m.role, "content": m.content} for m in payload.messages]
            res = await client.chat.completions.create(
                model=payload.model or settings.OPENAI_MODEL,
                messages=raw_messages,
                temperature=payload.temperature or 0.7
            )
            return res.model_dump()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI upstream error: {str(e)}")

    # Local fallback completion if no upstream key configured
    last_msg = payload.messages[-1].content if payload.messages else ""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.model or "fastapi-local-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Respon simulasi lokal FastAPI AI Monolith: Anda bertanya '{last_msg}'. Untuk respon LLM nyata, set OPENAI_API_KEY di .env."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(last_msg),
            "completion_tokens": 20,
            "total_tokens": len(last_msg) + 20
        }
    }


@router.post("/embeddings")
async def openai_embeddings(
    payload: EmbeddingRequest,
    user: User = Depends(get_current_user_optional)
):
    """
    OpenAI-Compatible Embeddings Endpoint.
    Returns 768-dimensional float embeddings in standard OpenAI JSON format.
    """
    inputs = [payload.input] if isinstance(payload.input, str) else payload.input
    data = []

    for idx, text_item in enumerate(inputs):
        vec = await get_embedding(text_item)
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": vec
        })

    return {
        "object": "list",
        "data": data,
        "model": payload.model or "text-embedding-3-small",
        "usage": {
            "prompt_tokens": sum(len(t) for t in inputs),
            "total_tokens": sum(len(t) for t in inputs)
        }
    }
