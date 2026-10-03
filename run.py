"""
Runner Script for Afi Template
Starts local development server with auto-reload.
"""
import os
import sys
import shutil

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def main():
    # If .env does not exist, copy from .env.example
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        print("[*] Creating .env from .env.example...")
        shutil.copy(".env.example", ".env")

    print("=" * 65)
    print(" [*] STARTING AFI TEMPLATE (CV + RAG + ML)")
    print("=" * 65)
    print(" - Web UI:      http://localhost:8000")
    print(" - CV Studio:   http://localhost:8000/cv")
    print(" - RAG Studio:  http://localhost:8000/rag")
    print(" - ML Studio:   http://localhost:8000/ml")
    print(" - Model Hub:   http://localhost:8000/models")
    print(" - Swagger API: http://localhost:8000/docs")
    print("=" * 65)

    try:
        import uvicorn
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    except ImportError:
        print("[!] Uvicorn belum terinstal. Silakan jalankan: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
