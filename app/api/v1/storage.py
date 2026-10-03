from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from app.models.user import User
from app.services.auth_service import get_current_user_optional
from app.services.storage_service import storage_service

router = APIRouter(prefix="/storage", tags=["Storage (S3 & Cloudflare R2)"])


@router.post("/upload")
async def api_upload_file(
    file: UploadFile = File(...),
    folder: str = Form("uploads"),
    user: User = Depends(get_current_user_optional)
):
    """
    Uploads a file to Cloudflare R2, AWS S3, or Local Filesystem.
    Returns the file URL, backend used, and file key.
    """
    contents = await file.read()
    content_type = file.content_type or "application/octet-stream"

    result = await storage_service.upload_file(
        file_bytes=contents,
        filename=file.filename or "uploaded_file",
        content_type=content_type,
        folder=folder
    )
    return result


@router.get("/info")
async def api_storage_info():
    """Returns active storage backend status (Cloudflare R2, S3, or Local)."""
    return storage_service.get_storage_info()
