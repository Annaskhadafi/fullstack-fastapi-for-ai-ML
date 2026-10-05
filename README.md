# 🚀 Python AI Monolith Boilerplate

Boilerplate **100% Python Monolith** modern, cepat, dan modular untuk aplikasi **Computer Vision (OpenCV + YOLO + Teachable Machine)**, **RAG (PostgreSQL Neon DB + pgvector & Vectorless)**, dan **Machine Learning (Scikit-Learn/ONNX)** dengan antarmuka interaktif **Jinja2 + HTMX + Tailwind CSS**.

> **💡 Mengapa Boilerplate ini Dibuat?**
> Menjalankan pustaka native seperti OpenCV atau model deteksi objek langsung di lingkungan JavaScript / Node.js (Next.js) sering kali gagal akibat kendala library C++ sistem (`libGL.so`), batasan arsitektur platform, maupun ukuran bundle serverless yang membengkak.
> Boilerplate ini menyatukan Frontend & Backend dalam **1 runtime Python yang konsisten**, ringan, stabil, dan siap di-deploy baik ke **Local**, **Docker**, maupun **Vercel Serverless**.

---

## ✨ Fitur Utama

- **👁️ Computer Vision Studio & Teachable Machine**:
  - Deteksi objek multi-kelas (80 kelas COCO) berbasis **YOLOv8 ONNX** yang sangat cepat & hemat memori.
  - Integrasi **Teachable Machine**: Mendukung klasifikasi gambar (*Image Classification*) dan **MediaPipe Pose** landmark.
  - Tampilan prediksi real-time di bawah video preview dengan pembersihan visualisasi landmark (tanpa garis wajah yang mengganggu).
  - Penstabil prediksi cerdas berbasis **Exponential Moving Average (EMA)** dan **Hysteresis State Lock** agar label tidak melompat-lompat saat objek diam.
  - Dukungan pemrosesan **Unggah File Gambar** dan **Live Snapshot Kamera Webcam** via browser (`webcam.js`).
  - Pipeline filter klasik **OpenCV Headless**: Canny Edge, Grayscale, Gaussian Blur, Contours, dan Deteksi Wajah (Haar Cascade).
- **📚 Knowledge Base & pgvector RAG (Retrieval-Augmented Generation)**:
  - Tersambung langsung ke database **PostgreSQL (Neon DB)** dengan ekstensi **pgvector** (tanpa perlu biaya ekstra untuk database vektor terpisah).
  - Pilihan mode **RAG Vectorless** untuk alur dokumen berbasis file lokal tanpa database vektor eksternal.
  - Sinkronisasi terpusat untuk konfigurasi provider LLM kustom (OpenAI, Gemini, Groq, DeepSeek, Ollama).
  - Chatbot tanya-jawab berbasis konteks dokumen dengan sitasi kemiripan (similarity score) dan fitur bulk delete dokumen.
- **🧠 Machine Learning Studio**:
  - Generic Model Runner (Scikit-Learn / Joblib / ONNX).
  - Formulir input dinamis otomatis berdasarkan skema fitur model.
  - Visualisasi probabilitas kelas dan confidence bar chart secara real-time.
- **📈 Market Forecast (Fraktal Pattern)**:
  - Dashboard `/grill-me` dengan analisis pola fraktal (180 candle historis + 30 candle proyeksi horizon) dan evaluasi prediksi vs aktual otomatis ke SQLite.
  - **Pilihan Fleksibel MetaTrader 5 (MT5)**: Dilengkapi toggle "Gunakan MT5" di UI. Secara default MT5 bersifat opsional—jika dimatikan atau MT5 tidak dibuka, sistem otomatis menggunakan generator pergerakan candle multi-siklik & random walk realistis.
  - Mekanisme **Graceful Auto-Fallback** dengan notifikasi informatif apabila koneksi ke MT5 terminal terputus atau gagal, mencegah error 503.
- **🔐 Autentikasi & Keamanan**:
  - Secure HTTP-Only Cookie Session untuk navigasi Web UI dengan branding EZ Template.
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
- **🆕 Setup Otomatis & Environment Doctor**:
  - `setup.ps1` membuat `.venv`, memasang dependensi, menyiapkan `.env` dan folder runtime, serta mencoba mengunduh bobot YOLO.
  - `SECRET_KEY` dihasilkan otomatis jika konfigurasi masih memakai placeholder/default development.
  - `start.ps1` memeriksa environment sebelum menjalankan server.
  - `python -m app doctor` menampilkan status **OK**, **WARN**, atau **ERROR** untuk Python, konfigurasi, dependensi dasar, model, kredensial storage, dan koneksi database.
- **🙂 Face Recognition Studio**:
  - Registrasi, pengenalan, daftar, dan penghapusan wajah melalui `/face` dan `/api/v1/face`.
- **🗂️ Model Hub**:
  - Kelola model lokal melalui `/models`, dukung unggah model kustom / Teachable Machine, dan gunakan metadata fitur dinamis untuk inferensi ML.

---

## 🖼️ Tampilan Aplikasi

Beberapa halaman utama yang tersedia setelah server dijalankan:

### EZ Template Login

![EZ Template Login](docs/screenshots/login-ez-template.png)

### Computer Vision Studio

![Computer Vision Studio](docs/screenshots/computer-vision.png)

### Face Recognition & Anti-Spoofing

![Face Recognition Studio](docs/screenshots/face-recognition.png)

### Knowledge Base & pgvector RAG

![Knowledge Base & pgvector RAG](docs/screenshots/rag.png)

### Model Hub — Daftar Model Computer Vision

![Model Hub Computer Vision](docs/screenshots/model-hub.png)

### Model Hub — Daftar Model Machine Learning

![Model Hub Machine Learning](docs/screenshots/model-hub-ml.png)

### Model Hub — Upload dan Registrasi Model

![Model Hub Upload Model](docs/screenshots/model-hub-upload.png)

> Screenshot diambil dari aplikasi lokal yang berjalan pada `http://localhost:8000`. Data pada gambar adalah data demo/runtime lokal.

---

## 🏗️ Struktur Direktori

```text
BOILERPLATE/
├── app/
│   ├── api/v1/                 # REST API Endpoints (/api/v1/auth, /cv, /rag, /ml)
│   ├── core/                   # Konfigurasi Pydantic, Database async, dan Keamanan
│   ├── models/                 # Model database SQLAlchemy (User, Document, MLModel)
│   ├── schemas/                # Skema validasi Pydantic
│   ├── services/               # Core AI logic (OpenCV, YOLO, RAG, Scikit-Learn, MT5)
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
├── setup.ps1                   # Setup Windows dengan .venv dan pemeriksaan environment
├── start.ps1                   # Jalankan doctor sebelum server
└── run.py                      # Script runner lokal
```

---

## 🚀 Panduan Memulai Cepat (Local Development)

### 🆕 Setup Windows via PowerShell:

Dari folder root proyek, jalankan:

```powershell
.\setup.ps1
.\start.ps1
```

**Prasyarat:** Python 3.10+ dengan launcher `py` tersedia. Setup menggunakan `.venv`, membuat `.env` jika belum ada, dan mengganti `SECRET_KEY` hanya jika masih berupa placeholder/default development. `start.ps1` menghentikan startup jika doctor menemukan error.

Jika kebijakan PowerShell memblokir script lokal, izin dapat diubah untuk terminal saat ini saja:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Untuk memeriksa ulang konfigurasi tanpa menyalakan server:

```powershell
.\.venv\Scripts\python.exe -m app doctor
```

Doctor memeriksa konfigurasi storage, tetapi belum melakukan upload uji ke S3/R2 atau pengujian provider LLM/embedding. Warning seperti SQLite lokal atau bobot YOLO belum tersedia tidak memblokir startup; error seperti secret default atau kredensial S3 sebagian perlu diperbaiki.

---

### Cara Paling Mudah di Windows (1-Click Batch File):

1. **Install Dependensi & Setup Otomatis**:
   - Cukup dobel klik file **`setup.bat`**
     *(Script ini menginstall seluruh dependensi `requirements.txt`, membuat file `.env` jika belum ada, dan mencoba mengunduh bobot model YOLOv8 ONNX. Untuk pembuatan virtual environment otomatis, gunakan `setup.ps1` di atas).*

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

# Akun Pertama / Super Admin Awal
FIRST_SUPERUSER_EMAIL=admin@aimonolith.local
FIRST_SUPERUSER_PASSWORD=admin123
FIRST_SUPERUSER_NAME=Administrator
```
> *Catatan: Jika `DATABASE_URL` dikosongkan, boilerplate akan otomatis menggunakan database SQLite lokal (`local.db`) untuk uji coba instan. Akun superuser awal akan otomatis dibuat saat server pertama kali dijalankan sesuai konfigurasi `.env`.*


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
- **Face Recognition Studio**: [http://localhost:8000/face](http://localhost:8000/face)
- **Model Hub**: [http://localhost:8000/models](http://localhost:8000/models)
- **RAG Vectorless**: [http://localhost:8000/rag/vectorless](http://localhost:8000/rag/vectorless)
- **Market Forecast**: [http://localhost:8000/grill-me](http://localhost:8000/grill-me)
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

## 📝 Changelog

### [2026-10-04] - Pembaruan Fitur & Peningkatan Stabilitas

#### 📈 Market Forecast (`/grill-me`)
- **Opsi Toggle MT5 (Opsional)**: Menambahkan checkbox "Gunakan MT5" pada dashboard Market Forecast (default nonaktif). Pengguna dapat menggunakan seluruh fitur tanpa harus membuka atau memasang MetaTrader 5.
- **Simulasi Candle Realistis**: Implementasi generator pergerakan harga berbasis *multi-frequency cycle wave* + *random walk* yang menghasilkan 500 candle konsisten untuk pemindaian pola fraktal (180 candle historis & 30 candle proyeksi horizon), grafik visualisasi, dan pencatatan riwayat prediksi otomatis ke SQLite.
- **Graceful Auto-Fallback**: Jika opsi MT5 diaktifkan tetapi koneksi terminal gagal atau market tidak ditemukan, sistem otomatis beralih ke simulasi dengan pemberitahuan alert di UI tanpa memunculkan error 503.
- **Background Monitor Asinkron**: Pemantauan evaluasi riwayat prediksi candle berjalan di latar belakang tanpa memblokir lock terminal.

#### 👁️ Computer Vision & Teachable Machine (`/cv`)
- **Dukungan Model Teachable Machine**: Integrasi model klasifikasi citra dan MediaPipe Pose landmark yang diekspor dari Teachable Machine.
- **Layout Tampilan Baru**: Menata posisi live prediction agar tampil rapi langsung di bawah video preview webcam.
- **Pembersihan Visualisasi MediaPipe**: Menghilangkan garis penghubung landmark yang mengganggu pada area wajah (hanya menyisakan titik-titik pose tubuh utama).
- **Stabilisasi Prediksi**: Menerapkan Exponential Moving Average (EMA, `alpha = 0.15`) dan mekanisme *hysteresis state lock* (debounce 4 frame / delta ambang 15%) sehingga label prediksi tidak melompat-lompat saat subjek diam.

#### 📚 Knowledge Base & RAG (`/rag`)
- **Sinkronisasi Provider LLM**: Pengaturan provider kustom (`API Key`, `Model`, `Base URL`) tersinkronisasi otomatis antara mode RAG Vectorless dan RAG pgvector.
- **Manajemen Dokumen**: Dukungan bulk delete dokumen dan pemisahan navigasi daftar dokumen.
- **Automated Tests**: Penambahan pengujian unit untuk validasi shared provider (`tests/test_rag_provider.py`).

#### 🗂️ Model Hub & Template UI
- **Pembaruan Model Hub (`/models`)**: Dukungan upload dan registrasi paket model Teachable Machine dan ONNX kustom.
- **Penyempurnaan Tampilan**: Pembaruan tema login dan navigasi sidebar "EZ Template".
- **Git Ignore**: Pembaruan `.gitignore` untuk mengabaikan artefak folder model upload dinamis di `weights/`.

---

## 📄 Lisensi
MIT License - Bebas digunakan dan dimodifikasi untuk keperluan komersial maupun riset.
