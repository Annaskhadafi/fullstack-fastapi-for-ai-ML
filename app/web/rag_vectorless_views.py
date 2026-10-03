from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from app.services.auth_service import get_current_user_optional
from app.services.rag_service import extract_pdf_text
from app.services.rag_vectorless_service import add_document, answer, inspect_pdf, list_documents, provider_settings, update_provider
from app.web.templates import render_template

router = APIRouter(prefix="/rag/vectorless", tags=["Web RAG Vectorless"])


@router.get("", response_class=HTMLResponse)
async def vectorless_page(request: Request, user=Depends(get_current_user_optional)):
    return render_template(request, "rag/vectorless/index.html", {"user": user, "docs": list_documents(), "provider": provider_settings()})


@router.post("/settings")
async def vectorless_settings(api_key: str = Form(""), model: str = Form(""), base_url: str = Form(""), user=Depends(get_current_user_optional)):
    if not user:
        return JSONResponse({"ok": False, "error": "Login diperlukan untuk mengubah provider AI."}, status_code=401)
    if not model.strip():
        return JSONResponse({"ok": False, "error": "Model AI wajib diisi."}, status_code=400)
    return {"ok": True, "provider": update_provider(api_key, model, base_url)}


@router.post("/documents")
async def vectorless_upload(title: str = Form(...), content: str = Form(""), file: UploadFile | None = File(None), user=Depends(get_current_user_optional)):
    if file and file.filename:
        raw = await file.read()
        content = extract_pdf_text(raw) if file.filename.lower().endswith(".pdf") else raw.decode("utf-8", errors="ignore")
    if len(content.strip()) < 5:
        return JSONResponse({"ok": False, "error": "Isi dokumen terlalu pendek atau kosong."}, status_code=400)
    count = add_document(title.strip(), content, user.id if user else None)
    return {"ok": True, "chunks": count, "documents": list_documents()}


@router.post("/inspect-pdf")
async def vectorless_inspect_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"ok": False, "error": "Pilih file PDF."}, status_code=400)
    try:
        result = inspect_pdf(await file.read())
        result["filename"] = file.filename
        return {"ok": True, "inspector": result}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"PDF tidak bisa dibaca: {exc}"}, status_code=400)


@router.post("/query")
async def vectorless_query(question: str = Form(...), top_k: int = Form(5)):
    if not question.strip():
        return JSONResponse({"ok": False, "error": "Pertanyaan masih kosong."}, status_code=400)
    return {"ok": True, **await answer(question.strip(), top_k)}
