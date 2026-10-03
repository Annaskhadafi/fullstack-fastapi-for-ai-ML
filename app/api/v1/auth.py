from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import create_access_token
from app.models.user import User
from app.schemas.auth import UserRegister, UserLogin, UserResponse, TokenResponse, APIKeyResponse, APIKeyItem
from app.services.auth_service import (
    register_user, authenticate_user, refresh_user_api_key, get_current_user,
    list_user_api_keys, create_user_api_key, revoke_user_api_key,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def api_register(payload: UserRegister, db: AsyncSession = Depends(get_db)):
    user = await register_user(db, payload)
    token = create_access_token({"sub": user.id, "email": user.email})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )


@router.post("/login", response_model=TokenResponse)
async def api_login(payload: UserLogin, db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, payload.email, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password salah."
        )
    token = create_access_token({"sub": user.id, "email": user.email})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )


@router.get("/me", response_model=UserResponse)
async def api_me(user: User = Depends(get_current_user)):
    return UserResponse.model_validate(user)


@router.post("/api-key/regenerate", response_model=APIKeyResponse)
async def api_regenerate_key(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    new_key = await refresh_user_api_key(db, user)
    return APIKeyResponse(
        api_key=new_key,
        message="API Key baru berhasil dibuat. Harap simpan dengan aman."
    )


@router.get("/api-keys", response_model=list[APIKeyItem])
async def api_list_keys(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    return await list_user_api_keys(db, user)


@router.post("/api-keys", response_model=APIKeyResponse)
async def api_create_key(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    key = await create_user_api_key(db, user, name="Generated")
    return APIKeyResponse(api_key=key.api_key, message="API Key berhasil dibuat. Harap simpan dengan aman.")


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def api_revoke_key(key_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if not await revoke_user_api_key(db, user, key_id):
        raise HTTPException(status_code=404, detail="API Key tidak ditemukan atau sudah dicabut.")
