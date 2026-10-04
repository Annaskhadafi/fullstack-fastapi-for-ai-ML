# FastAPI AI Monolith Boilerplate

Boilerplate Python monolith untuk aplikasi AI berbasis FastAPI, Jinja2 + HTMX, Computer Vision, RAG, Machine Learning, autentikasi, dan object storage S3/R2.

## Kemampuan

- Computer Vision: OpenCV, filter gambar, face recognition, dan YOLOv8 ONNX pada /cv, /face, dan /api/v1/cv.
- RAG: upload PDF/teks, embedding lokal atau provider OpenAI/Gemini, pgvector, dan fallback file lokal pada /rag dan /rag/vectorless.
- Machine Learning: Model Hub Joblib/ONNX dan inferensi metadata fitur dinamis pada /models, /ml, dan /api/v1/ml.
- Market Forecast: /grill-me, history SQLite, dan integrasi MetaTrader 5 bila tersedia.
- Auth: HTTP-only cookie session, JWT, bcrypt, dan header X-API-Key.
- Storage: filesystem lokal sebagai default, S3/R2/MinIO sebagai backend eksternal.
- OpenAI-compatible API: /api/v1/chat/completions dan /api/v1/embeddings.

## Struktur

    app/api/v1/       REST API
    app/core/         settings, database, security
    app/models/       SQLAlchemy models
    app/schemas/      Pydantic schemas
    app/services/     CV, RAG, ML, auth, storage, forecast
    app/templates/    Jinja2 + HTMX
    app/web/          HTML routes
    migrations/       Alembic environment dan revision
    weights/          model artifacts; file besar diabaikan Git
    tests/            unittest
    main.py           FastAPI app dan lifespan
    run.py            local runner

## Prasyarat

Python 3.10+, Git, dan Docker Desktop bila ingin PostgreSQL + pgvector. MetaTrader 5 hanya diperlukan untuk market forecast.

## Menjalankan lokal

Untuk Windows, jalur termudah adalah:

    .\setup.ps1
    .\start.ps1

Script tersebut membuat .venv, memasang dependency, membuat .env, menghasilkan SECRET_KEY lokal, menyiapkan folder runtime, mengunduh bobot YOLO, dan menjalankan pemeriksaan konfigurasi. Jika PowerShell memblokir script lokal, jalankan `Set-ExecutionPolicy -Scope Process Bypass` pada terminal itu saja.

setup.bat tetap tersedia untuk kompatibilitas lama, tetapi tidak membuat virtual environment. Untuk kerja tim gunakan langkah manual:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    Copy-Item .env.example .env
    python weights/download_weights.py
    python run.py

Linux/macOS menggunakan source .venv/bin/activate dan cp .env.example .env. Server juga dapat dijalankan dengan:

    uvicorn main:app --reload --port 8000

Periksa konfigurasi kapan saja dengan:

    .\.venv\Scripts\python.exe -m app doctor

Doctor memeriksa Python, dependency, `.env`, `SECRET_KEY`, database, model, dan kredensial storage.

URL utama: /docs (Swagger), /cv, /face, /rag, /rag/vectorless, /ml, /models, /grill-me, dan /api/health.

Tanpa DATABASE_URL, aplikasi memakai sqlite+aiosqlite:///./local.db. Tanpa kredensial S3, upload disimpan di app/static/uploads/.

## Konfigurasi

Salin .env.example ke .env dan ubah SECRET_KEY. Untuk PostgreSQL/Neon isi DATABASE_URL. Untuk pgvector jalankan:

    CREATE EXTENSION IF NOT EXISTS vector;

Untuk R2/S3 isi S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET_NAME, dan opsional S3_PUBLIC_DOMAIN. OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, dan GEMINI_API_KEY bersifat opsional sesuai provider.

## Database dan migrasi

Startup saat ini memanggil Base.metadata.create_all(). Alembic tetap tersedia:

    alembic current
    alembic revision --autogenerate -m "describe change"
    alembic upgrade head

Production sebaiknya menjalankan alembic upgrade head saat release dan memakai revision sebagai sumber perubahan schema.

## API singkat

Endpoint yang membutuhkan autentikasi menerima JWT atau X-API-Key:

    curl http://localhost:8000/api/health
    curl -X POST http://localhost:8000/api/v1/cv/detect -H "X-API-Key: YOUR_API_KEY" -F "file=@foto.jpg"
    curl -X POST http://localhost:8000/api/v1/ml/predict/iris_classifier -H "Content-Type: application/json" -H "X-API-Key: YOUR_API_KEY" -d '{"features":{"sepal_length":5.8,"sepal_width":2.7,"petal_length":5.1,"petal_width":1.9}}'

Gunakan /docs sebagai kontrak request/response yang paling akurat.

## Pengujian

    python -m unittest discover -s tests -p "test_*.py" -v
    python -m compileall app main.py run.py

Sebelum release, uji health, login, upload, prediksi ML, dan query RAG di environment target. Health 200 saja belum membuktikan provider eksternal berfungsi.

## Deployment

Docker Compose menjalankan FastAPI dan pgvector:

    docker compose up --build -d
    docker compose ps
    curl http://localhost:8000/api/health

vercel.json menunjuk main.py ke runtime @vercel/python. Isi environment production di Vercel. Fitur dengan proses panjang, MetaTrader 5, filesystem persisten, atau model besar lebih cocok dijalankan di container/VPS.

cloudflare_pages/functions adalah reverse proxy ke backend FastAPI. Atur BACKEND_URL pada Cloudflare Pages. Detail R2 dan proxy ada di cloudflare_pages/README_CLOUDFLARE.md.

## Penilaian kematangan

Yang sudah baik untuk starter: pemisahan API/web/service/model/schema, fallback lokal, health check, Docker Compose, Alembic environment, tests, dan OpenAPI.

Prioritas production-ready:

1. Satukan requirements.txt dan pyproject.toml, pin lockfile, lalu uji install bersih di CI.
2. Buat initial Alembic revision dan pindahkan perubahan schema ad-hoc dari startup ke migration.
3. Ganti secret default, jangan buat admin@aimonolith.local/admin123 di production, dan batasi CORS wildcard.
4. Pisahkan CV, embedding, upload, dan forecast ke worker saat latency atau volume meningkat. Model AI dapat terduplikasi di memori per worker.
5. Perlakukan joblib.load hanya untuk artifact tepercaya; simpan checksum dan versi model.
6. Tambahkan request ID, structured logging, error tracking, metric latency, dan health provider terpisah.
7. Batasi ukuran/tipe upload, tambah rate limit dan CSRF untuk form mutasi, validasi object key, dan jangan kirim exception mentah ke client.
8. Production gunakan PostgreSQL, object storage persisten, backup, dan restore drill; SQLite/filesystem hanya fallback development.

Urutan paling bernilai: dependency + CI, migrations, security defaults, observability, lalu worker/storage scaling.

## Lisensi

MIT License.
