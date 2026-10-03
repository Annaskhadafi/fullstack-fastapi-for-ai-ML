from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from app.services.auth_service import get_current_user_optional
from app.services.rag_service import extract_pdf_text
from app.services.rag_vectorless_service import add_document, answer, list_documents
from app.web.templates import render_template

router = APIRouter(prefix="/rag/vectorless", tags=["Web RAG Vectorless"])


@router.get("", response_class=HTMLResponse)
async def vectorless_page(request: Request, user=Depends(get_current_user_optional)):
    return render_template(request, "rag/vectorless/index.html", {"user": user, "docs": list_documents()})


@router.post("/documents")
async def vectorless_upload(title: str = Form(...), content: str = Form(""), file: UploadFile | None = File(None), user=Depends(get_current_user_optional)):
    if file and file.filename:
        raw = await file.read()
        content = extract_pdf_text(raw) if file.filename.lower().endswith(".pdf") else raw.decode("utf-8", errors="ignore")
    if len(content.strip()) < 5:
        return JSONResponse({"ok": False, "error": "Isi dokumen terlalu pendek atau kosong."}, status_code=400)
    count = add_document(title.strip(), content, user.id if user else None)
    return {"ok": True, "chunks": count, "documents": list_documents()}


@router.post("/query")
async def vectorless_query(question: str = Form(...), top_k: int = Form(5)):
    if not question.strip():
        return JSONResponse({"ok": False, "error": "Pertanyaan masih kosong."}, status_code=400)
    return {"ok": True, **await answer(question.strip(), top_k)}
