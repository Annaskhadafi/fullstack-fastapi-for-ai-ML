from fastapi import APIRouter, Request, Form, Depends, UploadFile, File
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services.auth_service import get_current_user_optional
from app.services.rag_service import (
    ingest_document,
    answer_rag_query,
    get_recent_documents_unified
)
from app.web.templates import render_template

router = APIRouter(prefix="/rag", tags=["Web RAG"])


@router.get("", response_class=HTMLResponse)
async def rag_page(
    request: Request,
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    # Fetch recent documents with local fallback
    docs = await get_recent_documents_unified(db, limit=20)

    return render_template(request, "rag/index.html", {
        "user": user,
        "docs": docs
    })


@router.post("/upload-htmx", response_class=HTMLResponse)
async def rag_upload_htmx(
    request: Request,
    title: str = Form(...),
    content: str = Form(""),
    file: UploadFile = File(None),
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    try:
        final_content = content
        if file and file.filename:
            file_bytes = await file.read()
            final_content = file_bytes.decode("utf-8", errors="ignore")

        if not final_content or len(final_content.strip()) < 5:
            return HTMLResponse(
                """<div class="alert alert-warning mb-4"><span>Isi dokumen terlalu pendek atau kosong.</span></div>""",
                status_code=400
            )

        user_id = user.id if user else None
        _, db_synced = await ingest_document(db, title=title.strip(), content=final_content, user_id=user_id)

        # Return updated document list
        docs = await get_recent_documents_unified(db, limit=20)

        sync_note = "pgvector & Local Storage" if db_synced else "Penyimpanan Lokal (Fallback Mode)"
        return render_template(request, "rag/partials/doc_list.html", {
            "docs": docs,
            "success_message": f"Dokumen '{title}' berhasil diindeks ke {sync_note}!"
        })
    except Exception as e:
        return HTMLResponse(
            f"""<div class="alert alert-error mb-4"><span>Gagal mengindeks dokumen: {str(e)}</span></div>""",
            status_code=500
        )


@router.post("/chat-htmx", response_class=HTMLResponse)
async def rag_chat_htmx(
    request: Request,
    question: str = Form(...),
    top_k: int = Form(3),
    db: AsyncSession = Depends(get_db)
):
    if not question.strip():
        return HTMLResponse("")

    response = await answer_rag_query(db, question=question.strip(), top_k=top_k)

    return render_template(request, "rag/partials/chat_message.html", {
        "question": question,
        "response": response
    })
