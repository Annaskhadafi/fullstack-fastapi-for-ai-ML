from fastapi import APIRouter
from app.api.v1.auth import router as auth_router
from app.api.v1.cv import router as cv_router
from app.api.v1.rag import router as rag_router
from app.api.v1.ml import router as ml_router
from app.api.v1.storage import router as storage_router
from app.api.v1.openai_compat import router as openai_compat_router
from app.api.v1.face import router as face_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(auth_router)
api_v1_router.include_router(cv_router)
api_v1_router.include_router(rag_router)
api_v1_router.include_router(ml_router)
api_v1_router.include_router(storage_router)
api_v1_router.include_router(openai_compat_router)
api_v1_router.include_router(face_router)
