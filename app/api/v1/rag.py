from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.user import User
from app.models.document import Document
from app.schemas.rag import DocumentCreate, DocumentResponse, RAGQueryRequest, RAGQueryResponse
from app.services.auth_service import get_current_user_optional
from app.services.rag_service import ingest_document, answer_rag_query

router = APIRouter(prefix="/rag", tags=["RAG & Knowledge Base"])


@router.post("/documents", response_model=List[DocumentResponse], status_code=status.HTTP_201_CREATED)
async def api_ingest_document(
    payload: DocumentCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_optional)
):
    """Chunks and ingests text into pgvector storage."""
    user_id = user.id if user else None
    docs = await ingest_document(db, title=payload.title, content=payload.content, user_id=user_id)
    return [
        DocumentResponse(
            id=d.id,
            title=d.title,
            content_snippet=d.content[:150] + ("..." if len(d.content) > 150 else ""),
            created_at=d.created_at
        )
        for d in docs
    ]


@router.get("/documents", response_model=List[DocumentResponse])
async def api_list_documents(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Lists ingested knowledge base documents."""
    query = select(Document).order_by(Document.created_at.desc()).limit(limit)
    res = await db.execute(query)
    docs = res.scalars().all()
    return [
        DocumentResponse(
            id=d.id,
            title=d.title,
            content_snippet=d.content[:150] + ("..." if len(d.content) > 150 else ""),
            created_at=d.created_at
        )
        for d in docs
    ]


@router.post("/query", response_model=RAGQueryResponse)
async def api_rag_query(
    payload: RAGQueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Performs pgvector semantic search and generates an answer using LLM
    conditioned on retrieved context.
    """
    try:
        response = await answer_rag_query(db, question=payload.question, top_k=payload.top_k)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query error: {str(e)}")
