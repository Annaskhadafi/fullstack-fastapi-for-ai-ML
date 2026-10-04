$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Error "Virtual environment belum ada. Jalankan .\setup.ps1 terlebih dahulu."
}

& ".venv\Scripts\python.exe" -m app doctor
if ($LASTEXITCODE -ne 0) {
    Write-Error "Doctor menemukan error. Perbaiki konfigurasi sebelum menjalankan server."
}

& ".venv\Scripts\python.exe" run.py
