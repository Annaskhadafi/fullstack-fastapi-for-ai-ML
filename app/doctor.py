"""Small, dependency-light environment check for local setup and deployment."""

import asyncio
import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _check(label: str, ok: bool, detail: str, warnings: list[str], errors: list[str]) -> None:
    status = "OK" if ok else "ERROR"
    print(f"[{status}] {label}: {detail}")
    if not ok:
        errors.append(label)


async def _database_check() -> tuple[bool, str]:
    try:
        from sqlalchemy import text
        from app.core.database import engine

        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        await engine.dispose()
        return True, "connection berhasil"
    except Exception as exc:  # doctor must report the actual setup problem
        return False, str(exc).splitlines()[0]


def main() -> int:
    warnings: list[str] = []
    errors: list[str] = []

    _check("Python", sys.version_info >= (3, 10), sys.version.split()[0], warnings, errors)

    env_file = ROOT / ".env"
    if env_file.exists():
        print("[OK] .env: ditemukan")
    else:
        print("[WARN] .env: belum ada; salin .env.example ke .env")
        warnings.append(".env")

    try:
        from app.core.config import settings
        weak_secret = settings.SECRET_KEY.startswith(("replace-with", "development-secret"))
        if weak_secret:
            print("[ERROR] SECRET_KEY: masih memakai nilai default")
            errors.append("SECRET_KEY")
        else:
            print("[OK] SECRET_KEY: sudah diubah")

        if settings.DATABASE_URL:
            print("[OK] Database: PostgreSQL dikonfigurasi")
        else:
            print("[WARN] Database: memakai SQLite lokal")
            warnings.append("Database")

        weight = Path(settings.WEIGHTS_DIR) / settings.CV_MODEL_NAME
        if weight.exists():
            print(f"[OK] Model: {weight}")
        else:
            print(f"[WARN] Model: {weight} belum ada; jalankan python weights/download_weights.py")
            warnings.append("Model")

        s3_values = [settings.S3_ACCESS_KEY_ID, settings.S3_SECRET_ACCESS_KEY]
        if any(s3_values) and not all(s3_values):
            print("[ERROR] Storage: kredensial S3/R2 tidak lengkap")
            errors.append("Storage")
        else:
            print("[OK] Storage: local fallback siap" if not any(s3_values) else "[OK] Storage: S3/R2 dikonfigurasi")
    except Exception as exc:
        print(f"[ERROR] Config: {exc}")
        errors.append("Config")

    for package in ("fastapi", "sqlalchemy", "pydantic_settings"):
        try:
            importlib.import_module(package)
            print(f"[OK] Dependency: {package}")
        except ImportError:
            print(f"[ERROR] Dependency: {package} belum terpasang")
            errors.append(package)

    if not errors:
        ok, detail = asyncio.run(_database_check())
        _check("Database connection", ok, detail, warnings, errors)

    print(f"\nSelesai: {len(errors)} error, {len(warnings)} warning.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
