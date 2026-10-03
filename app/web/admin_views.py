from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.user import User
from app.services.auth_service import require_admin
from app.web.templates import render_template

router = APIRouter(prefix="/admin", tags=["Web Admin"])


@router.get("/users", response_class=HTMLResponse)
async def admin_users_page(
    request: Request,
    current_admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin dashboard to view and manage users."""
    query = select(User).order_by(User.created_at.desc())
    result = await db.execute(query)
    users = result.scalars().all()

    return render_template(request, "admin/users.html", {
        "user": current_admin,
        "users": users
    })


@router.post("/users/{user_id}/toggle-role", response_class=HTMLResponse)
async def admin_toggle_role(
    request: Request,
    user_id: str,
    current_admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Toggles user role between 'admin' and 'user' via HTMX."""
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    target_user = result.scalar_one_or_none()

    if not target_user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    # Prevent admin from demoting themselves if they are the only admin
    if target_user.id == current_admin.id and target_user.role == "admin":
        return render_template(request, "admin/partials/user_row.html", {
            "u": target_user,
            "current_user": current_admin,
            "error_msg": "Anda tidak dapat mencabut akses admin Anda sendiri."
        })

    target_user.role = "user" if target_user.role == "admin" else "admin"
    target_user.is_admin = (target_user.role == "admin")
    await db.commit()
    await db.refresh(target_user)

    return render_template(request, "admin/partials/user_row.html", {
        "u": target_user,
        "current_user": current_admin
    })


@router.post("/users/{user_id}/toggle-active", response_class=HTMLResponse)
async def admin_toggle_active(
    request: Request,
    user_id: str,
    current_admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Toggles user active state (ban/unban) via HTMX."""
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    target_user = result.scalar_one_or_none()

    if not target_user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    if target_user.id == current_admin.id:
        return render_template(request, "admin/partials/user_row.html", {
            "u": target_user,
            "current_user": current_admin,
            "error_msg": "Anda tidak dapat menonaktifkan akun Anda sendiri."
        })

    target_user.is_active = not target_user.is_active
    await db.commit()
    await db.refresh(target_user)

    return render_template(request, "admin/partials/user_row.html", {
        "u": target_user,
        "current_user": current_admin
    })
