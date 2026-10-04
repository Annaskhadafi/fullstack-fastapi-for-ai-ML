$ErrorActionPreference = "Stop"

Write-Host "[1/6] Membuat virtual environment"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3 -m venv .venv
}
$Python = (Resolve-Path ".venv\Scripts\python.exe").Path

Write-Host "[2/6] Memperbarui pip"
& $Python -m pip install --upgrade pip

Write-Host "[3/6] Memasang dependency"
& $Python -m pip install -r requirements.txt

Write-Host "[4/6] Menyiapkan .env"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}
$Secret = & $Python -c "import secrets; print(secrets.token_urlsafe(32))"
$EnvText = Get-Content ".env" -Raw
if ($EnvText -match "SECRET_KEY=(replace-with|development-secret)") {
    $EnvText = $EnvText -replace "(?m)^SECRET_KEY=.*$", "SECRET_KEY=$Secret"
    Set-Content ".env" -Value $EnvText -Encoding utf8
}

Write-Host "[5/6] Menyiapkan folder runtime dan model"
New-Item -ItemType Directory -Force -Path "data\documents", "data\forecast", "app\static\uploads" | Out-Null
& $Python "weights\download_weights.py"

Write-Host "[6/6] Memeriksa environment"
& $Python -m app doctor
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Setup selesai dengan catatan. Perbaiki error doctor sebelum menjalankan aplikasi."
}

Write-Host "Selesai. Jalankan: .\start.ps1"
