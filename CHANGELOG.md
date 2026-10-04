# Changelog

Semua perubahan penting pada proyek ini dicatat dalam dokumen ini.

---

## [Unreleased] - 2026-10-04

### 📈 Market Forecast (`/grill-me`)
- **Added**: Toggle pilihan MetaTrader 5 ("Gunakan MT5") pada header dashboard (default nonaktif / opsional).
- **Added**: Generator simulasi candle sintetis multi-siklik (*sine waves + random walk*) sehingga analisis fraktal 180 bar historis, 30 bar proyeksi, dan chart interaktif tetap dapat diakses penuh tanpa MT5.
- **Added**: Auto-fallback graceful jika MT5 terminal tidak aktif atau gagal dihubungi, dengan notifikasi alert informatif di UI tanpa melempar error 503.
- **Improved**: Background monitor evaluasi SQLite berjalan asinkron dan terisolasi dari lock terminal.

### 👁️ Computer Vision & Teachable Machine (`/cv`)
- **Added**: Integrasi pemrosesan model Teachable Machine (Image Classification & MediaPipe Pose Landmark).
- **Changed**: Penataan layout tampilan; hasil prediksi ditempatkan secara langsung di bawah preview webcam kamera.
- **Fixed**: Pembersihan koneksi landmark MediaPipe pada area wajah untuk menghindari garis-garis silang yang mengganggu.
- **Improved**: Penambahan filter Exponential Moving Average (EMA, `alpha = 0.15`) serta *hysteresis state locking* (debounce 4 frame / delta 15%) untuk menstabilkan prediksi saat objek atau pose dalam posisi diam.

### 📚 Knowledge Base & RAG (`/rag`)
- **Added**: Sinkronisasi pengaturan LLM kustom (`API Key`, `Model`, `Base URL`) secara bersama antara mode Vectorless dan pgvector RAG.
- **Added**: Fitur bulk delete dokumen pada daftar dokumen.
- **Added**: Automated unit tests pada `tests/test_rag_provider.py`.

### 🗂️ Model Hub & Antarmuka (`/models`)
- **Added**: Dukungan upload dan registrasi paket model Teachable Machine dan ONNX kustom.
- **Changed**: Penyesuaian tema navigasi dan branding login "EZ Template".
- **Changed**: Pembaruan `.gitignore` untuk mengabaikan artefak folder model upload di direktori `weights/`.
