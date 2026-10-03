from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings, public_base_url
from app.core.database import get_db
from app.models.document import Document
from app.models.ml_model import MLModel
from app.models.user import User
from app.services.auth_service import get_current_user_optional, refresh_user_api_key, list_user_api_keys
from app.web.templates import render_template

router = APIRouter(tags=["Web Dashboard"])


@router.get("/", response_class=HTMLResponse)
async def home_page(request: Request, user=Depends(get_current_user_optional)):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(
    request: Request,
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    # Fetch stats safely with local storage fallback
    docs_count = 0
    models_count = 0
    try:
        db_docs = await db.scalar(select(func.count()).select_from(Document))
        db_models = await db.scalar(select(func.count()).select_from(MLModel))
        docs_count = db_docs or 0
        models_count = db_models or 0
    except Exception:
        pass

    # Ensure local count is included if DB returned 0 or failed
    from app.services.model_hub_service import scan_weights_directory
    from app.services.rag_service import load_local_documents
    local_models = len(scan_weights_directory())
    local_docs = len(load_local_documents())
    docs_count = max(docs_count, local_docs)
    models_count = max(models_count, local_models)

    api_keys = await list_user_api_keys(db, user)
    return render_template(request, "dashboard/index.html", {
        "user": user,
        "base_api_url": public_base_url(str(request.base_url)),
        "api_keys": api_keys,
        "app_name": settings.APP_NAME,
        "stats": {
            "docs_count": docs_count,
            "models_count": models_count,
            "db_type": "PostgreSQL (Neon DB)" if settings.is_postgres else "SQLite (Local Dev)",
            "gemini_active": bool(settings.GEMINI_API_KEY),
            "openai_active": bool(settings.OPENAI_API_KEY)
        }
    })


@router.post("/dashboard/regenerate-key", response_class=HTMLResponse)
async def dashboard_regenerate_key(
    request: Request,
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    new_key = await refresh_user_api_key(db, user)
    return render_template(request, "dashboard/partials/api_key_badge.html", {
        "api_key": new_key
    })


@router.post("/dashboard/api-keys/{key_action}", response_class=HTMLResponse)
async def dashboard_revoke_api_key(
    key_action: str,
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if key_action != "revoke":
        raise HTTPException(status_code=404, detail="API key action tidak ditemukan")
    user.api_key = None
    await db.commit()
    return HTMLResponse('<div class="alert alert-success text-sm py-2">API key berhasil direvoke. Generate key baru untuk mengaktifkan kembali.</div>')
