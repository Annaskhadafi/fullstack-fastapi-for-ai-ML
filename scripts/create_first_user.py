"""
CLI Helper to seed the initial / first user from .env settings into the database.
Can be executed directly via:
    python scripts/create_first_user.py
"""
import asyncio
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.core.database import init_db, AsyncSessionLocal
from app.services.auth_service import seed_default_admin


async def main():
    print(f"[*] Initializing database for {settings.APP_NAME}...")
    await init_db()

    masked_pw = "*" * len(settings.FIRST_SUPERUSER_PASSWORD) if settings.FIRST_SUPERUSER_PASSWORD else "(Not Set)"
    print("[*] Checking initial superuser credentials from .env:")
    print(f"    - Email: {settings.FIRST_SUPERUSER_EMAIL}")
    print(f"    - Name:  {settings.FIRST_SUPERUSER_NAME}")
    print(f"    - Password: {masked_pw}")

    async with AsyncSessionLocal() as session:
        user = await seed_default_admin(session)
        if user:
            print(f"[OK] First superuser ready: {user.email} (Role: {user.role}, Admin: {user.is_admin})")
            print(f"[OK] Master API Key: {user.api_key}")
        else:
            print("[!] Initial user was not created. Please check FIRST_SUPERUSER_EMAIL & FIRST_SUPERUSER_PASSWORD in .env.")


if __name__ == "__main__":
    asyncio.run(main())
