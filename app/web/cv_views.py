import base64
from typing import Optional
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services.auth_service import get_current_user_optional
from app.services.cv_service import run_yolo_detection, apply_opencv_filter
from app.services.model_hub_service import get_all_models_unified, resolve_cv_model_path
from app.web.templates import render_template

router = APIRouter(prefix="/cv", tags=["Web CV"])


@router.get("", response_class=HTMLResponse)
async def cv_studio_page(
    request: Request,
    user=Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    # Fetch registered CV models (.pt and .onnx) with local fallback
    models_dict = await get_all_models_unified(db)
    cv_models = models_dict["cv_models"]

    return render_template(request, "cv/index.html", {
        "user": user,
        "cv_models": cv_models
    })


@router.post("/detect-htmx", response_class=HTMLResponse)
async def cv_detect_htmx(
    request: Request,
    image_file: Optional[UploadFile] = File(None),
    webcam_base64: Optional[str] = Form(None),
    confidence: float = Form(0.35),
    model_name: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    try:
        raw_bytes = None
        if image_file and image_file.filename:
            raw_bytes = await image_file.read()
        elif webcam_base64:
            # Strip data:image/...;base64, prefix
            if "," in webcam_base64:
                webcam_base64 = webcam_base64.split(",")[1]
            raw_bytes = base64.b64decode(webcam_base64)

        if not raw_bytes:
            return HTMLResponse(
                """<div class="alert alert-error text-sm"><span>Harap unggah gambar atau ambil foto webcam terlebih dahulu!</span></div>""",
                status_code=400
            )

        model_path = await resolve_cv_model_path(model_name, db)

        result = run_yolo_detection(raw_bytes, conf_threshold=confidence, model_path=model_path)
        return render_template(request, "cv/partials/detection_result.html", {
            "result": result
        })
    except Exception as e:
        return HTMLResponse(
            f"""<div class="alert alert-error text-sm"><span>Gagal mendeteksi objek: {str(e)}</span></div>""",
            status_code=500
        )


@router.post("/filter-htmx", response_class=HTMLResponse)
async def cv_filter_htmx(
    request: Request,
    image_file: Optional[UploadFile] = File(None),
    webcam_base64: Optional[str] = Form(None),
    filter_type: str = Form("canny_edge")
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
                """<div class="alert alert-error"><span>Harap unggah gambar atau ambil foto webcam terlebih dahulu!</span></div>""",
                status_code=400
            )

        b64_img, meta = apply_opencv_filter(raw_bytes, filter_type=filter_type)
        return render_template(request, "cv/partials/filter_result.html", {
            "image_base64": b64_img,
            "filter_type": filter_type,
            "meta": meta
        })
    except Exception as e:
        return HTMLResponse(
            f"""<div class="alert alert-error"><span>Gagal memproses filter OpenCV: {str(e)}</span></div>""",
            status_code=500
        )
