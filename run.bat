@echo off
title Afi Template - Server Runner
set PYTHONIOENCODING=utf-8

echo ==============================================================================
echo   MENJALANKAN AFI TEMPLATE (CV + RAG + ML)
echo ==============================================================================
echo.

:: 1. Jika ada venv lokal, gunakan venv
if exist "venv\Scripts\activate.bat" (
    echo [*] Menggunakan virtual environment 'venv'...
    call venv\Scripts\activate.bat
) else (
    echo [*] Menggunakan Python sistem...
)

:: 2. Jalankan aplikasi
python run.py

:: 3. Jika server berhenti atau error, jangan pernah langsung close!
echo.
echo ==============================================================================
echo Server telah berhenti.
echo ==============================================================================
pause
