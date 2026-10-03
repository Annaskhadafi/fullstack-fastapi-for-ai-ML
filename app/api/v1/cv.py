from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.user import User
from app.schemas.cv import DetectionResult
from app.services.auth_service import get_current_user
from app.services.cv_service import run_yolo_detection, apply_opencv_filter
from app.services.model_hub_service import resolve_cv_model_path

router = APIRouter(prefix="/cv", tags=["Computer Vision"])


@router.post("/detect", response_model=DetectionResult)
async def api_detect_objects(
    file: UploadFile = File(...),
    confidence: float = Form(0.35),
    model_name: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Detects objects in an uploaded image using OpenCV and YOLO ONNX.
    Returns bounding box coordinates, detected classes, and base64 annotated image.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPG, PNG, WebP).")

    if model_name and ("/" in model_name or "\\" in model_name or model_name in {".", ".."}):
        raise HTTPException(status_code=400, detail="Nama model tidak valid.")

    model_path = await resolve_cv_model_path(model_name, db)
    if model_name and not model_path:
        raise HTTPException(status_code=404, detail=f"Model CV tidak ditemukan: {model_name}")

    contents = await file.read()
    try:
        result = run_yolo_detection(contents, conf_threshold=confidence, model_path=model_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat memproses gambar: {str(e)}")


@router.post("/filter")
async def api_apply_filter(
    file: UploadFile = File(...),
    filter_type: str = Form("canny_edge"),
    canny1: int = Form(100),
    canny2: int = Form(200),
    blur_kernel: int = Form(5),
    user: User = Depends(get_current_user)
):
    """
    Applies classic OpenCV filters: grayscale, canny_edge, gaussian_blur, threshold, contours, face_detect.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPG, PNG, WebP).")

    contents = await file.read()
    try:
        b64_img, meta = apply_opencv_filter(
            contents,
            filter_type=filter_type,
            canny1=canny1,
            canny2=canny2,
            blur_kernel=blur_kernel
        )
        return {"success": True, "image_base64": b64_img, "metadata": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat menerapkan filter OpenCV: {str(e)}")
