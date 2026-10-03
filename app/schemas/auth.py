from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, description="Password min 6 characters")
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_admin: bool
    role: str = "user"
    api_key: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class APIKeyResponse(BaseModel):
    api_key: str
    message: str


class APIKeyItem(BaseModel):
    id: str
    api_key: str
    name: str
    created_at: datetime
    revoked_at: Optional[datetime] = None

    class Config:
        from_attributes = True
