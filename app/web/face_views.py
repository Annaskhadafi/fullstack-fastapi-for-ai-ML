import base64
import logging
from typing import Optional
from fastapi import APIRouter, Request, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.services.auth_service import get_current_user_optional
from app.services.face_service import (
    get_all_registered_faces,
    register_face,
    delete_registered_face,
    recognize_faces
)
from app.web.templates import render_template

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face", tags=["Web Face Recognition"])


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def face_studio_page(
    request: Request,
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    faces = await get_all_registered_faces(db)

    return render_template(request, "face/index.html", {
        "user": user,
        "faces": faces,
        "total_registered": len(faces),
        "app_name": settings.APP_NAME
    })


@router.post("/recognize-htmx", response_class=HTMLResponse)
async def face_recognize_htmx(
    request: Request,
    image_file: Optional[UploadFile] = File(None),
    webcam_base64: Optional[str] = Form(None),
    threshold: float = Form(0.45),
    db: AsyncSession = Depends(get_db)
):
    try:
        raw_bytes = None
        if image_file and image_file.filename:
            raw_bytes = await image_file.read()
        elif webcam_base64:
            if "," in webcam_base64:
                webcam_base64 = webcam_base64.split(",")[1]
            raw_bytes = base64.b64decode(webcam_base64)

        if not raw_bytes:
            return HTMLResponse(
                """<div class="alert alert-error text-xs py-2">
                    <span>Harap unggah gambar atau ambil foto webcam terlebih dahulu!</span>
                </div>""",
                status_code=400
            )

        result = await recognize_faces(
            image_bytes=raw_bytes,
            similarity_threshold=threshold,
            db=db
        )

        return render_template(request, "face/partials/recognition_result.html", {
            "result": result
        })
    except Exception as e:
        logger.error(f"Error in face recognize: {e}", exc_info=True)
        return HTMLResponse(
            f"""<div class="alert alert-error text-xs py-2">
                <span>Gagal menganalisis wajah: {str(e)}</span>
            </div>""",
            status_code=500
        )


@router.post("/register-htmx", response_class=HTMLResponse)
async def face_register_htmx(
    request: Request,
    name: str = Form(...),
    identity_number: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    photo_file: Optional[UploadFile] = File(None),
    webcam_photo_base64: Optional[str] = Form(None),
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        raw_bytes = None
        if photo_file and photo_file.filename:
            raw_bytes = await photo_file.read()
        elif webcam_photo_base64:
            if "," in webcam_photo_base64:
                webcam_photo_base64 = webcam_photo_base64.split(",")[1]
            raw_bytes = base64.b64decode(webcam_photo_base64)

        if not raw_bytes:
            return HTMLResponse(
                """<div class="alert alert-error text-xs py-2">
                    <span>Harap pilih file foto atau ambil snapshot webcam untuk pendaftaran!</span>
                </div>""",
                status_code=400
            )

        res = await register_face(
            name=name,
            image_bytes=raw_bytes,
            identity_number=identity_number,
            notes=notes,
            db=db
        )

        sync_text = "Database & Local Storage" if res["db_synced"] else "Penyimpanan Lokal (Fallback Mode)"

        return HTMLResponse(
            f"""<div class="alert alert-success text-xs py-2.5">
                <i data-lucide="check-circle" class="w-4 h-4 mr-1 inline-block"></i>
                Wajah <strong>{name}</strong> berhasil didaftarkan ({sync_text})!
                Memuat ulang halaman...
                <script>setTimeout(() => window.location.reload(), 1200);</script>
            </div>"""
        )
    except ValueError as e:
        return HTMLResponse(
            f"""<div class="alert alert-warning text-xs py-2"><span>{str(e)}</span></div>""",
            status_code=400
        )
    except Exception as e:
        logger.error(f"Error registering face: {e}", exc_info=True)
        return HTMLResponse(
            f"""<div class="alert alert-error text-xs py-2"><span>Gagal mendaftarkan wajah: {str(e)}</span></div>""",
            status_code=500
        )


@router.post("/{face_id}/delete", response_class=HTMLResponse)
async def face_delete_htmx(
    face_id: str,
    request: Request,
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        await delete_registered_face(face_id, db)
        return HTMLResponse(
            """<div class="alert alert-info text-xs py-2">
                <span>Data wajah berhasil dihapus.</span>
                <script>setTimeout(() => window.location.reload(), 600);</script>
            </div>"""
        )
    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        return HTMLResponse(
            f"""<div class="alert alert-error text-xs py-2"><span>Gagal menghapus: {str(e)}</span></div>""",
            status_code=500
        )
