# 🚀 Afi Template - Python AI Monolith Boilerplate

Boilerplate **100% Python Monolith** modern, cepat, dan modular untuk aplikasi **Computer Vision (OpenCV + YOLO)**, **RAG (PostgreSQL Neon DB + pgvector)**, dan **Machine Learning (Scikit-Learn/ONNX)** dengan antarmuka interaktif **Jinja2 + HTMX + Tailwind CSS**.

> **💡 Mengapa Boilerplate ini Dibuat?**  
> Menjalankan pustaka native seperti OpenCV atau model deteksi objek langsung di lingkungan JavaScript / Node.js (Next.js) sering kali gagal akibat kendala library C++ sistem (`libGL.so`), batasan arsitektur platform, maupun ukuran bundle serverless yang membengkak.  
> Boilerplate ini menyatukan Frontend & Backend dalam **1 runtime Python yang konsisten**, ringan, stabil, dan siap di-deploy baik ke **Local**, **Docker**, maupun **Vercel Serverless**.

---

## ✨ Fitur Utama

- **👁️ Computer Vision Studio**:
  - Deteksi objek multi-kelas (80 kelas COCO) berbasis **YOLOv8 ONNX** yang sangat cepat & hemat memori.
  - Dukungan pemrosesan **Unggah File Gambar** dan **Live Snapshot Kamera Webcam** via browser (`webcam.js`).
  - Pipeline filter klasik **OpenCV Headless**: Canny Edge, Grayscale, Gaussian Blur, Contours, dan Deteksi Wajah (Haar Cascade).
- **📚 pgvector RAG (Retrieval-Augmented Generation)**:
  - Tersambung langsung ke database **PostgreSQL (Neon DB)** dengan ekstensi **pgvector** (tanpa perlu biaya ekstra untuk database vektor terpisah).
  - Adapter modular embedding (Google Gemini `text-embedding-004`, OpenAI, atau local fallback).
  - Chatbot tanya-jawab berbasis konteks dokumen dengan sitasi kemiripan (similarity score).
- **🧠 Machine Learning Studio**:
  - Generic Model Runner (Scikit-Learn / Joblib / ONNX).
  - Formulir input dinamis otomatis berdasarkan skema fitur model.
  - Visualisasi probabilitas kelas dan confidence bar chart secara real-time.
- **🔐 Autentikasi & Keamanan**:
  - Secure HTTP-Only Cookie Session untuk navigasi Web UI.
  - Manajemen **API Key** (`sk_live_...`) untuk memanggil endpoint REST API dari aplikasi eksternal / mobile / IoT.
  - Password hashing dengan `bcrypt` dan verifikasi token `JWT`.
- **⚡ Frontend Reaktif Tanpa Node.js**:
  - Ditenagai **FastAPI + Jinja2 + HTMX + Tailwind CSS + DaisyUI**.
  - Interaksi SPA dinamis tanpa kompilasi webpack/vite atau kerumitan bridging Node.js.
- **🪣 Object Storage Kompatibel S3 & Cloudflare R2**:
  - Dukungan penyimpanan file/gambar ke **Cloudflare R2**, **AWS S3**, **MinIO**, atau fallback lokal otomatis.
  - Zero egress fee dengan Cloudflare R2 untuk menghemat biaya hosting.
- **🤖 OpenAI SDK Compatible & Multi-Provider LLM**:
  - Ditenagai SDK resmi `openai`, mendukung OpenAI (`gpt-4o`), **Groq**, **DeepSeek**, **Cloudflare Workers AI**, dan **Ollama** lokal cukup dengan mengubah `OPENAI_BASE_URL`.
  - Dilengkapi endpoint kompatibel OpenAI (`/api/v1/chat/completions` dan `/api/v1/embeddings`) sehingga server ini bisa dihubungkan ke Cursor, LangChain, atau LibreChat.
- **☁️ Deployment Fleksibel (Vercel, Docker, & Cloudflare Pages)**:
  - Siap di-deploy ke **Vercel Serverless** via `vercel.json` (teroptimasi di bawah batas 250MB).
  - Kompatibel dengan **Cloudflare Pages** via reverse proxy functions (`cloudflare_pages/functions/api/[[path]].js`).
  - Siap dijalankan dengan **Docker** & `docker-compose.yml`.
  - Zero-config local fallback (otomatis menggunakan SQLite async jika belum ada koneksi PostgreSQL).

---

## 🏗️ Struktur Direktori

```text
BOILERPLATE/
├── app/
│   ├── api/v1/                 # REST API Endpoints (/api/v1/auth, /cv, /rag, /ml)
│   ├── core/                   # Konfigurasi Pydantic, Database async, dan Keamanan
│   ├── models/                 # Model database SQLAlchemy (User, Document, MLModel)
│   ├── schemas/                # Skema validasi Pydantic
│   ├── services/               # Core AI logic (OpenCV, YOLO, RAG, Scikit-Learn)
│   ├── static/                 # Static assets (webcam.js, main.js, css)
│   ├── templates/              # Jinja2 HTML templates + HTMX partials
│   └── web/                    # Monolith web controllers (HTML & HTMX views)
├── migrations/                 # Migrasi database Alembic
├── weights/                    # Bobot model AI (yolov8n.onnx, scikit-learn .joblib)
├── .env.example                # Template konfigurasi environment
├── Dockerfile                  # Container build config
├── docker-compose.yml          # Setup container + PostgreSQL pgvector lokal
├── requirements.txt            # Dependensi Python teroptimasi
├── vercel.json                 # Konfigurasi deployment Vercel Serverless
├── main.py                     # Entry point aplikasi FastAPI
└── run.py                      # Script runner lokal
```

---

## 🚀 Panduan Memulai Cepat (Local Development)

### Cara Paling Mudah di Windows (1-Click Batch File):

1. **Install Dependensi & Setup Otomatis**:
   - Cukup dobel klik file **`setup.bat`**  
     *(Script ini akan otomatis membuat `venv`, menginstall seluruh dependensi `requirements.txt`, membuat file `.env`, dan mengunduh bobot model YOLOv8 ONNX).*

2. **Menjalankan Server**:
   - Cukup dobel klik file **`run.bat`**  
     *(Server akan langsung aktif dan menampilkan link web di `http://localhost:8000`).*

---

### Cara Manual via Terminal (PowerShell / Bash):

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan di Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Atau di Linux/macOS:
source venv/bin/activate

# Install dependensi
pip install -r requirements.txt
```

### 2. Konfigurasi Variabel Lingkungan (.env)

Salin file contoh ke `.env`:
```bash
cp .env.example .env
```

Buka `.env` dan sesuaikan pengaturan Anda:
```env
# Koneksi PostgreSQL (Neon DB)
# Contoh: postgresql+asyncpg://user:pass@ep-xyz-pooler.us-east-2.aws.neon.tech/neondb?ssl=require
DATABASE_URL=

# AI Provider Key untuk RAG (Dapatkan gratis di https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_api_key_here
```
> *Catatan: Jika `DATABASE_URL` dikosongkan, boilerplate akan otomatis menggunakan database SQLite lokal (`local.db`) untuk uji coba instan.*

### 3. (Opsional) Download Model YOLOv8 ONNX

Jalankan script helper untuk mengunduh bobot model YOLOv8 Nano (~12MB):
```bash
python weights/download_weights.py
```
*(Jika belum diunduh, sistem tetap dapat mendeteksi wajah dengan OpenCV Haar Cascade bawaan).*

### 4. Jalankan Server Pengembangan

```bash
python run.py
```
atau via uvicorn langsung:
```bash
uvicorn main:app --reload --port 8000
```

Buka browser di:
- **Web UI**: [http://localhost:8000](http://localhost:8000)
- **Computer Vision Studio**: [http://localhost:8000/cv](http://localhost:8000/cv)
- **pgvector RAG Studio**: [http://localhost:8000/rag](http://localhost:8000/rag)
- **Machine Learning Studio**: [http://localhost:8000/ml](http://localhost:8000/ml)
- **Dokumentasi REST API (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🗄️ Menghubungkan ke PostgreSQL (Neon DB)

1. Buat project database baru di [Neon Console](https://console.neon.tech/).
2. Aktifkan ekstensi `vector` jika belum aktif di SQL Editor Neon:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Salin connection string dari dashboard Neon (pilih mode **Pooled**).
4. Masukkan string tersebut ke variabel `DATABASE_URL` di `.env`. Boilerplate ini secara otomatis mengonversi format Neon (`postgresql://` atau `sslmode=require`) ke driver asinkron `postgresql+asyncpg://`.
5. Jalankan migrasi jika diperlukan:
   ```bash
   alembic upgrade head
   ```

---

## ☁️ Panduan Deployment

### A. Deploy ke Vercel Serverless
Proyek ini telah dilengkapi file `vercel.json` yang dikonfigurasi untuk runtime `@vercel/python`:
1. Pastikan repository Anda telah di-push ke GitHub / GitLab.
2. Import project ke [Vercel](https://vercel.com).
3. Tambahkan Environment Variable di Vercel Settings:
   - `DATABASE_URL`: Connection string Neon DB Anda.
   - `SECRET_KEY`: String rahasia acak untuk JWT.
   - `GEMINI_API_KEY`: API key AI Anda.
4. Klik **Deploy**! Vercel akan otomatis mengeksekusi `main.py`.

### B. Deploy dengan Docker & Docker Compose
Untuk menjalankan seluruh stack (termasuk PostgreSQL dengan pgvector lokal) dalam kontainer:
```bash
docker compose up --build -d
```
Aplikasi akan tersedia di `http://localhost:8000` dan PostgreSQL di port `5432`.

---

## 📡 Integrasi REST API Eksternal

Endpoint REST API dapat dipanggil dari aplikasi luar dengan menyertakan header `X-API-Key`:

### Deteksi Objek (cURL)
```bash
curl -X POST "http://localhost:8000/api/v1/cv/detect" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@foto.jpg" \
  -F "confidence=0.4"
```

### Tanya RAG (cURL)
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"question": "Bagaimana cara kerja sistem ini?", "top_k": 3}'
```

### Prediksi ML (cURL)
```bash
curl -X POST "http://localhost:8000/api/v1/ml/predict/iris_classifier" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"features": {"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.9}}'
```

---

## 📄 Lisensi
MIT License - Bebas digunakan dan dimodifikasi untuk keperluan komersial maupun riset.
