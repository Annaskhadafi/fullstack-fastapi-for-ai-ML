from fastapi import APIRouter, Request, Form, Depends, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.database import get_db
from app.core.security import create_access_token
from app.schemas.auth import UserRegister
from app.services.auth_service import authenticate_user, register_user, get_current_user_optional
from app.web.templates import render_template

router = APIRouter(tags=["Web Auth"])


@router.get("/login", response_class=HTMLResponse)
async def login_page(
    request: Request,
    info: str = None,
    user=Depends(get_current_user_optional)
):
    if user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    
    info_message = None
    if info == "closed":
        info_message = "Pendaftaran akun publik dinonaktifkan oleh Administrator. Silakan masuk menggunakan akun default/resmi."

    return render_template(request, "auth/login.html", {
        "user": None,
        "error": None,
        "info": info_message
    })


@router.post("/login", response_class=HTMLResponse)
async def login_action(
    request: Request,
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    user = await authenticate_user(db, email.strip(), password)
    if not user:
        return render_template(
            request,
            "auth/login.html",
            {"user": None, "error": "Email atau kata sandi tidak valid.", "email": email, "info": None},
            status_code=status.HTTP_400_BAD_REQUEST
        )

    token = create_access_token({"sub": user.id, "email": user.email})

    # Redirect to dashboard with secure cookie
    redirect_resp = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    redirect_resp.set_cookie(
        key=settings.COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    return redirect_resp


@router.get("/register", response_class=HTMLResponse)
async def register_page():
    # Public registration is disabled
    return RedirectResponse(url="/login?info=closed", status_code=status.HTTP_302_FOUND)


@router.post("/register", response_class=HTMLResponse)
async def register_action():
    # Public registration is disabled
    return RedirectResponse(url="/login?info=closed", status_code=status.HTTP_302_FOUND)


@router.get("/logout")
async def logout_action():
    resp = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    resp.delete_cookie(settings.COOKIE_NAME)
    return resp
