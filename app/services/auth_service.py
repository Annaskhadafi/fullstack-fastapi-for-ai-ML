import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import Request, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import (
    hash_password,
    verify_password,
    generate_api_key,
    decode_access_token,
    get_current_token_from_request,
    api_key_header
)
from app.core.config import settings
from app.models.user import User
from app.models.api_key import ApiKey
from app.schemas.auth import UserRegister

logger = logging.getLogger(__name__)


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    query = select(User).where(User.email == email.lower())
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_user_by_api_key(db: AsyncSession, api_key: str) -> Optional[User]:
    if not api_key:
        return None
    query = (
        select(User)
        .join(ApiKey, ApiKey.user_id == User.id)
        .where(ApiKey.api_key == api_key, ApiKey.revoked_at.is_(None))
    )
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    if user:
        return user
    # Keep keys created before the api_keys table was introduced working.
    result = await db.execute(select(User).where(User.api_key == api_key))
    return result.scalar_one_or_none()


async def list_user_api_keys(db: AsyncSession, user: User) -> list[ApiKey]:
    result = await db.execute(
        select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.created_at.desc())
    )
    return list(result.scalars().all())


async def create_user_api_key(db: AsyncSession, user: User, name: str = "Default") -> ApiKey:
    key = ApiKey(user_id=user.id, api_key=generate_api_key(), name=name)
    db.add(key)
    # The legacy column remains the key shown by existing clients/templates.
    if not user.api_key:
        user.api_key = key.api_key
    await db.commit()
    await db.refresh(key)
    return key


async def revoke_user_api_key(db: AsyncSession, user: User, key_id: str) -> bool:
    result = await db.execute(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id)
    )
    key = result.scalar_one_or_none()
    if not key or key.revoked_at:
        return False
    key.revoked_at = datetime.now(timezone.utc)
    await db.commit()
    return True


async def seed_default_admin(db: AsyncSession) -> Optional[User]:
    """Auto-creates the default administrator / first user account if it does not exist."""
    admin_email = (settings.FIRST_SUPERUSER_EMAIL or "").strip().lower()
    admin_password = (settings.FIRST_SUPERUSER_PASSWORD or "").strip()
    admin_name = (settings.FIRST_SUPERUSER_NAME or "Administrator").strip()

    if not admin_email or not admin_password:
        logger.info("FIRST_SUPERUSER_EMAIL atau FIRST_SUPERUSER_PASSWORD belum diset. Melewati initial user seeding.")
        return None

    admin_query = select(User).where(User.email == admin_email)
    admin_user = (await db.execute(admin_query)).scalar_one_or_none()
    if not admin_user:
        new_api_key = generate_api_key()
        default_admin = User(
            email=admin_email,
            hashed_password=hash_password(admin_password),
            full_name=admin_name,
            is_active=True,
            is_admin=True,
            role="admin",
            api_key=new_api_key
        )
        db.add(default_admin)
        await db.commit()
        await db.refresh(default_admin)

        # Register corresponding ApiKey entry in api_keys table
        admin_key_record = ApiKey(
            user_id=default_admin.id,
            api_key=new_api_key,
            name="Default Master Key"
        )
        db.add(admin_key_record)
        await db.commit()

        logger.info(f"Default initial administrator account created: {admin_email}")
        return default_admin
    else:
        # Pastikan user tersebut memiliki status aktif dan role admin
        updated = False
        if not admin_user.is_admin:
            admin_user.is_admin = True
            updated = True
        if admin_user.role != "admin":
            admin_user.role = "admin"
            updated = True
        if not admin_user.is_active:
            admin_user.is_active = True
            updated = True
        if updated:
            await db.commit()
            await db.refresh(admin_user)
            logger.info(f"Updated privileges for first superuser: {admin_email}")

    return admin_user


async def register_user(db: AsyncSession, payload: UserRegister) -> User:
    # Public registration is disabled per specification
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Pendaftaran akun publik telah dinonaktifkan oleh Administrator. Hubungi admin untuk mendapatkan akun."
    )


async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]:
    user = await get_user_by_email(db, email)
    if not user or not user.is_active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


async def refresh_user_api_key(db: AsyncSession, user: User) -> str:
    key = await create_user_api_key(db, user, name="Generated")
    return key.api_key


async def get_current_user_optional(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Extracts current authenticated user from:
    1. X-API-Key Header
    2. HTTP-Only Cookie or Authorization Bearer token
    Returns None if not authenticated.
    """
    # 1. Check API Key
    if api_key:
        user = await get_user_by_api_key(db, api_key)
        if user and user.is_active:
            return user

    # 2. Check JWT from Cookie or Header
    token = get_current_token_from_request(request)
    if token:
        payload = decode_access_token(token)
        if payload and "sub" in payload:
            user = await get_user_by_id(db, payload["sub"])
            if user and user.is_active:
                return user

    return None


async def get_current_user(
    user: Optional[User] = Depends(get_current_user_optional)
) -> User:
    """Dependency that strictly requires an authenticated user."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Autentikasi diperlukan. Silakan login atau sertakan X-API-Key yang valid."
        )
    return user


async def require_admin(
    user: User = Depends(get_current_user)
) -> User:
    """Dependency that requires the user to have admin / superadmin privileges."""
    if not user.is_superadmin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Akses ditolak: Menu ini hanya dapat diakses oleh Super Admin / Guru."
        )
    return user
