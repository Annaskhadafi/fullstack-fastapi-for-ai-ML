# 🎓 Panduan Praktis & Tutorial Belajar: Afi Template

Selamat datang! Panduan ini dirancang khusus untuk siswa, mahasiswa, dan pemula agar dapat langsung menjalankan dan bereksperimen dengan teknologi **Kecerdasan Buatan (AI)** seperti **Computer Vision (OpenCV & YOLO)**, **RAG (Pencarian Dokumen & Chatbot AI)**, dan **Machine Learning** tanpa pusing dengan instalasi yang rumit.

---

## 📌 DAFTAR ISI
1. [Persiapan Awal di Komputer/Laptop](#1-persiapan-awal)
2. [Langkah 1-Klik Menjalankan Aplikasi](#2-langkah-1-klik-menjalankan-aplikasi)
3. [Login Pertama Kali (Akun Default Super Admin)](#3-login-pertama-kali)
4. [Tampilan Baru: Sidebar Navigation & Light Mode](#4-tampilan-baru-sidebar--light-mode)
5. [Eksperimen 1: Computer Vision & Deteksi Objek](#5-eksperimen-1-computer-vision)
6. [Eksperimen 2: Model Hub (Upload Model Sendiri .pt, .onnx, .pkl)](#6-eksperimen-2-model-hub-upload-model-sendiri)
7. [Eksperimen 3: RAG & Chatbot Dokumen Pintar (Neon pgvector)](#7-eksperimen-3-rag--chatbot-dokumen)
8. [Eksperimen 4: Machine Learning & Prediksi Tabular](#8-eksperimen-4-machine-learning)
9. [Eksperimen 5: REST API via Swagger Docs](#9-eksperimen-5-mencoba-rest-api)
10. [Eksperimen 6: Belajar Sistem Hak Akses (Super Admin vs User Standar)](#10-eksperimen-6-belajar-sistem-hak-akses-rbac)
11. [Tanya Jawab & Solusi Masalah Umum (FAQ)](#11-solusi-masalah-umum-faq)

---

## 💻 1. Persiapan Awal

Sebelum mulai, pastikan di laptop Anda sudah terpasang:
1. **Python 3.10 atau versi lebih baru**:
   - Download di: [python.org/downloads](https://www.python.org/downloads/)
   - ⚠️ **PENTING saat Install**: Centang kotak bertuliskan **"Add python.exe to PATH"** di bagian paling bawah installer sebelum menekan tombol Install Now.
2. **Web Browser Modern**: Google Chrome, Microsoft Edge, atau Mozilla Firefox.

---

## 🚀 2. Langkah 1-Klik Menjalankan Aplikasi

Anda tidak perlu mengetik perintah panjang di command prompt! Cukup gunakan file `.bat` yang disediakan:

### Langkah A: Dobel-Klik `setup.bat` (Hanya 1x di awal)
1. Buka folder proyek ini di Windows Explorer.
2. Cari file bernama **`setup.bat`** dan **klik 2 kali (double-click)**.
3. Skrip otomatis memasang pustaka: FastAPI, PyTorch YOLO, OpenCV Headless, pgvector, dan Scikit-Learn.
4. Tunggu sampai muncul tulisan `[✓] SETUP SELESAI!`.

### Langkah B: Dobel-Klik `run.bat`
1. Cari file bernama **`run.bat`** dan **klik 2 kali**.
2. Server otomatis menyala di port `8000`.
3. Buka browser Anda dan kunjungi: **`http://localhost:8000`**

*(💡 Untuk mematikan server, cukup tutup jendela terminal atau tekan `Ctrl + C`).*

---

## 🔐 3. Login Pertama Kali

Sistem ini didesain sebagai **Enterprise Monolith** di mana registrasi publik dinonaktifkan demi keamanan. Pengguna yang pertama kali membuka aplikasi akan langsung diarahkan ke **Halaman Login**.

### Akun Bawaan (Super Admin):
- **Email**: `admin@aimonolith.local`
- **Password**: `admin123`

> 💡 **Fitur 1-Klik Auto-Fill**: Di halaman login terdapat tombol **"Gunakan Akun Admin Ini (1-Klik)"**. Klik tombol tersebut untuk mengisi email dan sandi secara otomatis, lalu tekan **Masuk ke Dashboard**!

---

## 🎨 4. Tampilan Baru: Sidebar & Light Mode

Aplikasi ini menggunakan antarmuka modern yang ramah untuk penggunaan seharian:
- **Default Light Mode**: Desain bersih (*clean slate*) berbasis Tailwind CSS & DaisyUI.
- **Sidebar Navigation**: Menu navigasi tetap di sebelah kiri layar:
  - 📊 **Dashboard**: Ringkasan sistem, dokumen RAG, model ML, dan manajemen API Key pribadi.
  - 📷 **Computer Vision**: Studio deteksi objek & filter OpenCV.
  - 🧠 **pgvector RAG**: Chatbot tanya jawab berbasis dokumen pengetahuan.
  - ⚡ **Machine Learning**: Studio eksekusi model tabular dengan form dinamis.
  - 📦 **Model Hub**: Tempat mengunggah model AI buatan Anda sendiri!
  - 🛡️ **Kelola Pengguna**: Panel Super Admin untuk menambah murid / akun baru.
  - 📖 **API Swagger Docs**: Dokumentasi interaktif REST API OpenAPI.

---

## 👁️ 5. Eksperimen 1: Computer Vision

Buka menu **Computer Vision** di sidebar atau kunjungi: `http://localhost:8000/cv`

### A. Uji Coba Deteksi Objek Gambar:
1. Klik tab **"Unggah Gambar"**.
2. Pilih foto dari laptop Anda (misalnya foto jalan raya, kucing, laptop, atau orang).
3. Pada dropdown **"Pilih Model CV"**, Anda dapat memilih:
   - **YOLOv8 Nano (Default PyTorch / ONNX)**: Deteksi 80 kelas objek COCO dataset.
   - **OpenCV Haar Cascade**: Deteksi wajah manusia bawaan OpenCV tanpa ketergantungan file eksternal.
   - Model kustom Anda yang telah diunggah di Model Hub.
4. Atur slider **Ambang Batas Confidence** (standar: 35%).
5. Klik tombol biru **"Deteksi Objek"**.
6. **Hasilnya**: Kotak pembatas (*bounding box*) berwarna akan digambar di sekitar objek beserta label dan persentase akurasinya!

### B. Uji Coba Filter Klasik OpenCV:
1. Setelah memilih gambar, klik tombol dropdown **"Filter OpenCV"**.
2. Pilih filter yang ingin dicoba:
   - **Grayscale**: Mengubah citra ke hitam putih.
   - **Canny Edge**: Menampilkan garis tepi kontur objek.
   - **Gaussian Blur**: Efek blur penghalus noise.
   - **Contours**: Menemukan kontur garis terluar objek.
   - **Deteksi Wajah (Haar Cascade)**: Deteksi wajah cepat dengan OpenCV.

### C. Uji Coba Live Webcam:
1. Klik tab **"Kamera Webcam"**.
2. Izinkan akses kamera (*Allow Camera*).
3. Klik tombol **"Ambil Foto & Deteksi"** untuk snapshot real-time, atau aktifkan mode live auto-deteksi.

---

## 📦 6. Eksperimen 2: Model Hub (Upload Model Sendiri)

Anda dan murid-murid dapat melatih model di Google Colab, Jupyter Notebook, atau PyTorch, lalu mengunggahnya ke monolith ini:

1. Buka menu **Model Hub** di sidebar (`http://localhost:8000/models`).
2. Klik tombol **"Upload Model Baru"**.
3. Pilih file model dari laptop Anda:
   - **Untuk Computer Vision**: format `.pt` (PyTorch YOLO) atau `.onnx`.
   - **Untuk Machine Learning**: format `.pkl`, `.joblib`, atau `.onnx`.
4. Isi formulir:
   - **Kategori**: Computer Vision atau Machine Learning.
   - **Framework**: PyTorch YOLO, ONNX, Scikit-Learn, atau Pickle.
   - **Nama Tampilan**: Misalnya *"Model Deteksi Helm Proyek"*.
   - **Tipe Tugas**: Object Detection, Classification, dll.
5. Klik **"Simpan & Daftarkan"**.
6. Model Anda langsung tersimpan di folder `weights/` server dan langsung muncul di dropdown Studio CV dan Studio ML!

---

## 🧠 7. Eksperimen 3: RAG & Chatbot Dokumen (Neon pgvector)

Buka menu **pgvector RAG** di sidebar (`http://localhost:8000/rag`)

1. Klik tombol **"Tambah Dokumen"**.
2. Berikan Judul dan isi teks materi pelajaran/SOP (misal: materi tentang tata tertib sekolah atau sejarah komputer).
3. Klik **"Simpan & Embedding"**. Sistem akan membuat representasi vektor dan menyimpannya di database PostgreSQL.
4. Coba ketik pertanyaan di kotak chat bawah, misalnya: *"Jelaskan tentang materi tadi"*.
5. Asisten AI akan mencari potongan paragraf yang paling relevan (*semantic search*) dan menyusun jawabannya.

---

## 👤 8. Eksperimen: Face Recognition & Biometrik (UniFace)

Buka menu **Face Recognition** di sidebar (`http://localhost:8000/face`):

### A. Mendaftarkan Wajah (Face Registration)
1. Klik tab **"Form Daftarkan Wajah"**.
2. Masukkan Nama Lengkap (misal: `Budi Santoso`), No. Identitas (NIM/NIK), dan Catatan.
3. Unggah foto wajah atau klik tombol **"Buka Kamera"** untuk mengambil foto langsung dari webcam.
4. Klik **"Simpan & Daftarkan Wajah"**.
5. Sistem menggunakan model **UniFace (SCRFD + ArcFace)** untuk mendeteksi wajah dan mengekstrak vektor biometrik 512-dimensi yang unik.

### B. Mengenali Wajah (Face Scanner & Recognition)
1. Buka tab **"Scanner & Pengenalan Wajah"**.
2. Unggah foto baru atau aktifkan **Kamera Webcam** (bisa klik **"Live Auto-Scan"** untuk deteksi otomatis real-time).
3. Bila wajah orang yang terdaftar muncul di kamera:
   - Kotak deteksi berwarna **Hijau** dengan label nama orang tersebut (misal: `Budi Santoso (98%)`).
   - Bila orang asing atau belum terdaftar: kotak berwarna **Merah/Kuning** bertuliskan `Wajah Tidak Dikenal (Unknown)`.
   - Estimasi umur dan gender juga otomatis diprediksi oleh UniFace!

### C. Menghubungkan ke Sistem Lain (Microservice REST API)
Sistem ini siap diintegrasikan ke aplikasi luar (aplikasi mobile presensi, IoT absensi, sistem web lain):
- `POST /api/v1/face/recognize`: Mengirimkan foto (multipart form atau base64) dan menerima JSON daftar orang yang cocok.
- `POST /api/v1/face/register`: Mendaftarkan orang baru secara programmatic.
- `GET /api/v1/face/list`: Mengambil seluruh database orang terdaftar.
- Dokumentasi interaktif Swagger: buka **`http://localhost:8000/docs`**.

---

## ⚡ 8. Eksperimen 4: Machine Learning

Buka menu **Machine Learning** di sidebar (`http://localhost:8000/ml`)

1. Pada menu dropdown, pilih model demo bawaan: **Iris Flower Classifier (Scikit-Learn)**.
2. Formulir dinamis akan langsung menampilkan 4 parameter input:
   - *Sepal Length*, *Sepal Width*, *Petal Length*, *Petal Width*.
3. Klik **"Jalankan Prediksi"**.
4. Sistem akan menampilkan nama spesies bunga (*setosa*, *versicolor*, atau *virginica*) beserta grafik persentase probabilitasnya!

---

## 🌐 9. Eksperimen 5: Mencoba REST API

FastAPI secara otomatis menyediakan antarmuka dokumentasi Swagger interaktif:
1. Buka tab baru di browser: **`http://localhost:8000/docs`**
2. Anda dapat menguji endpoint:
   - `POST /api/v1/cv/detect`: Kirim gambar via cURL/Postman.
   - `POST /api/v1/rag/query`: Query semantik pgvector.
   - `POST /api/v1/ml/predict/{model_name}`: Prediksi model ML via JSON.
3. Kunci otentikasi dapat diambil dari Dashboard (**API Key Pribadi Anda**) dan dipasang pada tombol hijau **"Authorize"** di pojok kanan atas Swagger.

---

## 🛡️ 10. Eksperimen 6: Belajar Sistem Hak Akses (RBAC)

1. Saat login sebagai `admin@aimonolith.local`, Anda memiliki peran **Super Admin**.
2. Masuk ke menu **"Kelola Pengguna"** di sidebar (`http://localhost:8000/admin/users`).
3. Buat akun baru untuk murid Anda dengan role **User**.
4. Coba *Logout* dan masuk menggunakan akun murid tersebut:
   - Menu *Kelola Pengguna* tidak akan terlihat.
   - Akun murid tetap dapat menggunakan Computer Vision, RAG, dan Machine Learning secara penuh.

---

## ❓ 11. Solusi Masalah Umum (FAQ)

### Q: Kamera webcam tidak muncul?
- Pastikan Anda membuka web menggunakan alamat `http://localhost:8000` (bukan alamat IP mentah), karena protokol keamanan browser membatasi akses webcam di luar localhost kecuali menggunakan HTTPS.
- Pastikan permission browser untuk kamera berstatus *"Allow"*.

### Q: Komputer tanpa kartu grafis (GPU) apakah bisa menjalankan YOLO?
- **Sangat bisa!** Boilerplate ini menggunakan PyTorch CPU dan ONNX Runtime CPU execution provider yang dioptimalkan secara headless. Inferensi berjalan rata-rata 150-250 ms di laptop standar.

### Q: Mengapa halaman registrasi dialihkan ke login?
- Registrasi publik memang dinonaktifkan agar murid tidak membuat banyak akun sembarangan. Guru/Admin dapat membuat akun murid melalui menu **Kelola Pengguna**.
