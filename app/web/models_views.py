import os
import logging
from typing import Optional
from fastapi import APIRouter, Request, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.services.auth_service import get_current_user_optional
from app.services.model_hub_service import (
    get_all_models_unified,
    save_uploaded_model,
    delete_model_unified
)
from app.web.templates import render_template

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Web Model Hub"])

ALLOWED_EXTENSIONS = {
    "computer_vision": [".pt", ".onnx"],
    "machine_learning": [".pkl", ".joblib", ".onnx"]
}


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def models_hub_page(
    request: Request,
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    # Fetch all models unified (Postgres DB + Local Registry fallback + loose weights folder)
    models_dict = await get_all_models_unified(db)

    return render_template(request, "models/index.html", {
        "user": user,
        "cv_models": models_dict["cv_models"],
        "ml_models": models_dict["ml_models"],
        "app_name": settings.APP_NAME,
        "weights_dir": settings.WEIGHTS_DIR
    })


@router.post("/upload", response_class=HTMLResponse)
async def upload_model_file(
    request: Request,
    user=Depends(get_current_user_optional),
    file: UploadFile = File(...),
    display_name: str = Form(...),
    category: str = Form(...),
    framework: str = Form(...),
    task_type: str = Form("object_detection"),
    description: Optional[str] = Form(None),
    features_json: Optional[str] = Form(None),
    target_names_json: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    filename = file.filename or "model.bin"
    ext = os.path.splitext(filename)[1].lower()

    allowed = ALLOWED_EXTENSIONS.get(category, [".pt", ".onnx", ".pkl", ".joblib"])
    if ext not in allowed:
        return HTMLResponse(
            f"""<div class="alert alert-error text-sm py-2">
                Format file '{ext}' tidak didukung untuk kategori {category}. Gunakan: {', '.join(allowed)}
               </div>""",
            status_code=400
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            return HTMLResponse(
                """<div class="alert alert-error text-sm py-2">File yang diunggah kosong!</div>""",
                status_code=400
            )

        # Save to local storage (weights/) + local registry + DB sync (with local fallback)
        res = await save_uploaded_model(
            file_bytes=file_bytes,
            original_filename=filename,
            display_name=display_name,
            category=category,
            framework=framework,
            task_type=task_type,
            description=description,
            features_json=features_json,
            target_names_json=target_names_json,
            db=db
        )

        db_note = "Tersinkron ke Database & Local Storage" if res["db_synced"] else "Tersimpan di Local Storage (Fallback Mode)"

        return HTMLResponse(
            f"""<div class="alert alert-success text-sm py-2">
                <i data-lucide="check-circle" class="w-4 h-4 inline-block mr-1"></i>
                Model <strong>{display_name}</strong> berhasil di-upload! ({db_note}).
                Memuat ulang halaman...
                <script>setTimeout(() => window.location.reload(), 1200);</script>
            </div>"""
        )
    except Exception as e:
        logger.error(f"Error handling model upload: {e}", exc_info=True)
        return HTMLResponse(
            f"""<div class="alert alert-error text-sm py-2">Gagal mengunggah model: {str(e)}</div>""",
            status_code=500
        )


@router.post("/{model_id}/delete", response_class=HTMLResponse)
async def delete_model(
    model_id: str,
    request: Request,
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        await delete_model_unified(model_id, db)
        return HTMLResponse(
            """<div class="alert alert-info text-sm py-2">
                Model berhasil dihapus dari sistem.
                <script>setTimeout(() => window.location.reload(), 600);</script>
            </div>"""
        )
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        return HTMLResponse(
            f"""<div class="alert alert-error text-sm py-2">Gagal menghapus model: {str(e)}</div>""",
            status_code=500
        )
