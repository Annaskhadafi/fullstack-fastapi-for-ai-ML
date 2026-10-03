@echo off
title Afi Template - Setup dan Install Dependensi

echo ==============================================================================
echo   AFI TEMPLATE - INSTALASI DEPENDENSI DAN SETUP LOKAL
echo ==============================================================================
echo.

:: 1. Periksa ketersediaan Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python tidak terdeteksi di sistem Anda!
    echo Silakan install Python 3.10+ dari https://www.python.org/downloads/
    echo Pastikan centang opsi "Add Python to PATH" saat menginstall.
    goto error_exit
)

echo [1/4] Memeriksa Python...
python --version

:: 2. Buat file .env jika belum ada
if not exist ".env" (
    echo.
    echo [2/4] Membuat file konfigurasi .env dari .env.example...
    copy .env.example .env >nul
    echo [*] File .env berhasil dibuat.
) else (
    echo.
    echo [2/4] File .env sudah ada.
)

:: 3. Install requirements
echo.
echo [3/4] Menginstall seluruh dependensi requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Gagal menginstall dependensi. Periksa koneksi internet Anda.
    goto error_exit
)

:: 4. Siapkan bobot model
echo.
echo [4/4] Memeriksa bobot model AI...
python weights\download_weights.py

echo.
echo ==============================================================================
echo   [✓] SETUP SELESAI! SEMUA DEPENDENSI BERHASIL DISIAPKAN.
echo ==============================================================================
echo.
echo Anda sekarang dapat menjalankan server dengan:
echo   - Dobel klik file 'run.bat'
echo.
pause
exit /b 0

:error_exit
echo.
echo [!] Setup berhenti karena terjadi kendala di atas.
pause
exit /b 1
