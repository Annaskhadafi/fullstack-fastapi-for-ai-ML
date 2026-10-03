import base64
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.face import (
    FaceListResponse,
    RegisteredFaceItem,
    FaceRegisterResponse,
    FaceRecognizeResponse
)
from app.services.face_service import (
    get_all_registered_faces,
    register_face,
    delete_registered_face,
    recognize_faces
)

router = APIRouter(prefix="/face", tags=["Face Biometrics (UniFace API)"])


@router.get("/list", response_model=FaceListResponse)
async def list_registered_faces_api(db: AsyncSession = Depends(get_db)):
    """
    Returns list of all registered faces and their biometrics metadata.
    Suitable for external service integration.
    """
    faces = await get_all_registered_faces(db)
    items = [
        RegisteredFaceItem(
            id=f["id"],
            name=f["name"],
            identity_number=f.get("identity_number"),
            notes=f.get("notes"),
            photo_url=f["photo_url"],
            created_at=f.get("created_at"),
            source=f.get("source", "database")
        )
        for f in faces
    ]
    return FaceListResponse(success=True, total=len(items), faces=items)


@router.post("/register", response_model=FaceRegisterResponse)
async def register_face_api(
    name: str = Form(..., description="Nama lengkap pemilik wajah"),
    identity_number: Optional[str] = Form(None, description="Nomor Identitas (NIK/NIM/ID Karyawan)"),
    notes: Optional[str] = Form(None, description="Catatan tambahan atau departemen"),
    file: Optional[UploadFile] = File(None, description="File foto wajah (JPG/PNG)"),
    image_base64: Optional[str] = Form(None, description="Base64 encoded string dari gambar"),
    db: AsyncSession = Depends(get_db)
):
    """
    Registers a new person by extracting 512-dim ArcFace embeddings using UniFace.
    Accepts photo file upload or base64 data URI.
    """
    raw_bytes = None
    if file and file.filename:
        raw_bytes = await file.read()
    elif image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        raw_bytes = base64.b64decode(image_base64)

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Harap sertakan file foto atau data image_base64!"
        )

    try:
        res = await register_face(
            name=name,
            image_bytes=raw_bytes,
            identity_number=identity_number,
            notes=notes,
            db=db
        )
        return FaceRegisterResponse(
            success=True,
            id=res["id"],
            name=res["name"],
            identity_number=res.get("identity_number"),
            photo_url=res["photo_url"],
            embedding_dim=res["embedding_dim"],
            db_synced=res["db_synced"],
            message=f"Wajah '{name}' berhasil didaftarkan ke sistem biometrik UniFace."
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal memproses pendaftaran wajah: {str(e)}")


@router.post("/recognize", response_model=FaceRecognizeResponse)
async def recognize_faces_api(
    file: Optional[UploadFile] = File(None, description="File foto yang ingin diidentifikasi"),
    image_base64: Optional[str] = Form(None, description="Base64 encoded string gambar"),
    similarity_threshold: float = Form(0.45, description="Ambang batas kemiripan (0.10 - 0.95, default 0.45)"),
    include_annotated_image: bool = Form(True, description="Apakah mengembalikan gambar beranotasi bounding box base64"),
    db: AsyncSession = Depends(get_db)
):
    """
    Detects and recognizes faces in the submitted photo against the registered database.
    Returns matched names, similarity percentages, age, gender, and bounding boxes.
    """
    raw_bytes = None
    if file and file.filename:
        raw_bytes = await file.read()
    elif image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        raw_bytes = base64.b64decode(image_base64)

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Harap unggah file foto atau sertakan image_base64!"
        )

    try:
        result = await recognize_faces(
            image_bytes=raw_bytes,
            similarity_threshold=similarity_threshold,
            db=db
        )
        if not include_annotated_image:
            result["annotated_image_base64"] = None
        return FaceRecognizeResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal menganalisis wajah: {str(e)}"
        )


@router.delete("/{face_id}")
async def delete_face_api(face_id: str, db: AsyncSession = Depends(get_db)):
    """Deletes a registered face by ID."""
    deleted = await delete_registered_face(face_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Data wajah tidak ditemukan")
    return {"success": True, "message": f"Wajah ID {face_id} berhasil dihapus"}
