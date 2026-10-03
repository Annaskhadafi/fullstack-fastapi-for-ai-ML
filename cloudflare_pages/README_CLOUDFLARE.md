# ☁️ Panduan Integrasi Cloudflare (Pages + R2 Object Storage + Workers AI)

Boilerplate ini telah dirancang untuk kompatibel secara penuh dengan ekosistem **Cloudflare**:

1. **Cloudflare R2** (S3-Compatible Object Storage tanpa biaya egress)
2. **OpenAI SDK via Cloudflare Workers AI** (Menjalankan model LLM di edge Cloudflare)
3. **Cloudflare Pages** (Hosting frontend edge dengan reverse proxy otomatis ke backend FastAPI)

---

## 🪣 1. Cara Menggunakan Cloudflare R2 (Object Storage)

Cloudflare R2 100% kompatibel dengan API AWS S3. Untuk mengaktifkannya:

1. Buka dashboard Cloudflare -> **R2 Object Storage** -> **Create Bucket** (contoh: `ai-monolith-uploads`).
2. Buat API Token: Klik **Manage R2 API Tokens** -> **Create API Token** (Pilih izin *Object Read & Write*).
3. Buka file `.env` di proyek ini dan masukkan:
   ```env
   # Ganti <account_id> dengan ID Akun Cloudflare Anda
   S3_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
   S3_ACCESS_KEY_ID=your_r2_access_key_id
   S3_SECRET_ACCESS_KEY=your_r2_secret_access_key
   S3_BUCKET_NAME=ai-monolith-uploads
   S3_REGION_NAME=auto
   # (Opsional) Domain publik bucket R2 atau Custom Domain Anda:
   S3_PUBLIC_DOMAIN=https://pub-xxxxxx.r2.dev
   ```
4. Setiap gambar atau file yang diunggah ke `/api/v1/storage/upload` sekarang otomatis tersimpan di Cloudflare R2!

---

## 🤖 2. Cara Menggunakan Cloudflare Workers AI (OpenAI SDK Compatible)

Cloudflare menyediakan endpoint REST yang kompatibel dengan format OpenAI SDK:

1. Buka Cloudflare Dashboard -> **Workers & Pages** -> **AI**.
2. Buat API Token dengan izin *Workers AI*.
3. Buka file `.env` dan konfigurasikan:
   ```env
   OPENAI_BASE_URL=https://api.cloudflare.com/client/v4/accounts/<account_id>/ai/v1
   OPENAI_API_KEY=your_cloudflare_workers_ai_token
   OPENAI_MODEL=@cf/meta/llama-3.1-8b-instruct
   OPENAI_EMBEDDING_MODEL=@cf/baai/bge-small-en-v1.5
   ```
4. RAG Chatbot dan Embeddings otomatis berjalan ditenagai serverless Cloudflare Workers AI!

---

## 🌐 3. Cara Deploy ke Cloudflare Pages

Untuk mendeploy antarmuka ke Cloudflare Pages dengan backend FastAPI:

1. Di Cloudflare Pages Dashboard, pilih **Create a project** -> **Connect to Git**.
2. **Build Settings**:
   - Framework preset: `None`
   - Build output directory: `app/templates` atau `cloudflare_pages`
3. Salin folder `cloudflare_pages/functions` ke root repository jika ingin menggunakan proxy otomatis.
4. Di **Environment Variables** Cloudflare Pages:
   - Tambahkan variabel `BACKEND_URL`: URL server FastAPI Anda (misalnya `https://nama-backend-anda.vercel.app` atau URL server VPS Anda).
5. Cloudflare Pages akan otomatis mem-proxy semua request `/api/*` langsung ke server backend FastAPI Anda!
