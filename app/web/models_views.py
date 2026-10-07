import os
import io
import zipfile
import logging
from typing import Optional
from fastapi import APIRouter, Request, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.services.auth_service import get_current_user_optional
from app.services.model_hub_service import (
    get_all_models_unified,
    save_uploaded_model,
    delete_model_unified,
    resolve_teachable_asset,
    resolve_model_file_path
)
from app.services.converter_service import convert_cv_model
from app.web.templates import render_template

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Web Model Hub"])

ALLOWED_EXTENSIONS = {
    "computer_vision": [".pt", ".onnx", ".tflite", ".zip"],
    "machine_learning": [".pkl", ".joblib", ".onnx"]
}


@router.get("/teachable/{model_name}/{asset_path:path}")
async def teachable_asset(model_name: str, asset_path: str):
    asset = resolve_teachable_asset(model_name, asset_path)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset Teachable Machine tidak ditemukan")
    return FileResponse(asset)


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
    if framework == "teachable_machine":
        allowed = [".zip"]
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


@router.post("/convert", response_class=HTMLResponse)
async def convert_model_endpoint(
    request: Request,
    user=Depends(get_current_user_optional),
    source_identifier: str = Form(...),
    target_format: str = Form(...),
    display_name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        res = await convert_cv_model(
            source_identifier=source_identifier,
            target_format=target_format,
            display_name=display_name,
            description=description,
            db=db
        )
        db_note = "Tersinkron ke Database & Local Storage" if res["db_synced"] else "Tersimpan di Local Storage"
        return HTMLResponse(
            f"""<div class="alert alert-success text-sm py-3 flex items-start gap-2">
                <i data-lucide="check-circle" class="w-5 h-5 text-emerald-600 shrink-0 mt-0.5"></i>
                <div>
                    <span class="font-bold">Konversi Berhasil!</span>
                    <p class="text-xs text-slate-600 mt-0.5">Model <strong>{res['display_name']}</strong> ({res['framework'].upper()}) berhasil dibuat ({db_note}). Memuat ulang halaman...</p>
                </div>
                <script>
                    setTimeout(() => window.location.reload(), 1500);
                </script>
            </div>"""
        )
    except Exception as e:
        logger.error(f"Error converting model: {e}", exc_info=True)
        return HTMLResponse(
            f"""<div class="alert alert-error text-sm py-3 flex items-start gap-2">
                <i data-lucide="alert-circle" class="w-5 h-5 text-rose-600 shrink-0 mt-0.5"></i>
                <div>
                    <span class="font-bold">Gagal Mengonversi Model</span>
                    <p class="text-xs text-rose-700 mt-0.5">{str(e)}</p>
                </div>
            </div>""",
            status_code=500
        )


@router.get("/{model_id}/download")
@router.get("/download/{model_id}")
async def download_model(
    model_id: str,
    request: Request,
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    resolved_path = await resolve_model_file_path(model_id, db)
    if not resolved_path or not os.path.exists(resolved_path):
        raise HTTPException(status_code=404, detail="File model tidak ditemukan di server.")

    # If it points to model.json (Teachable Machine bundle), package the directory
    if os.path.isfile(resolved_path) and os.path.basename(resolved_path) == "model.json":
        resolved_path = os.path.dirname(resolved_path)

    # If it's a directory (SavedModel or Teachable Machine folder), zip it on the fly
    if os.path.isdir(resolved_path):
        dir_name = os.path.basename(resolved_path)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(resolved_path):
                for file in files:
                    file_full = os.path.join(root, file)
                    arcname = os.path.relpath(file_full, resolved_path)
                    zip_file.write(file_full, arcname)
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{dir_name}.zip"'
            }
        )

    # If it's a single file (.pt, .onnx, .tflite, .joblib, .pkl)
    filename = os.path.basename(resolved_path)
    return FileResponse(
        path=resolved_path,
        filename=filename,
        media_type="application/octet-stream"
    )
