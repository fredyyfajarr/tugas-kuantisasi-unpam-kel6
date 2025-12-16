# 📷 Aplikasi Kuantisasi Citra (Histogram Based)

Aplikasi ini dibuat untuk memenuhi **Tugas Besar Mata Kuliah Pengolahan Citra Digital**.
Program ini mengimplementasikan algoritma **Non-Uniform Quantization** menggunakan metode _Equal Frequency Binning_ untuk mengompresi citra dengan tetap menjaga distribusi warna.

## 👨‍💻 Kelompok 6 (Universitas Pamulang)

Anggota Tim:

1. Farid Nuhgraha
2. Fredy Fajar Adi Putra
3. Maulana Aulia Rahman
4. Muhamad Aziz Mufashshal
5. Muhammad Faiz Saputra
6. Ravail Shodikin

## ✨ Fitur Unggulan

- **Arsitektur MVC:** Kode terstruktur rapi (Model-View-Controller).
- **Slider Perbandingan:** Geser untuk melihat _Before-After_ secara real-time.
- **Analisis Mendalam:**
  - Visualisasi Histogram Overlay.
  - Tabel Matriks Data Mentah (Raw Labels).
  - Kamus Warna (Codebook).
  - Statistik Ukuran File & Kompresi.
- **Anti-Lag System:** Otomatis menyesuaikan resolusi citra besar agar performa tetap ringan.

## 🚀 Cara Menjalankan

1.  **Install Library:**
    Pastikan Python sudah terinstall, lalu jalankan:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Jalankan Aplikasi:**
    ```bash
    python -m streamlit run app.py
    ```

---

© 2025 Teknik Informatika - Universitas Pamulang
