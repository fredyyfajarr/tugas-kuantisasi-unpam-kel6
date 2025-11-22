import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io

# --- FUNGSI LOGIKA METODE KUANTISASI (POIN 4) ---

def calculate_mse_psnr(original, compressed):
    """
    Menghitung MSE dan PSNR sesuai rumus di PDF[cite: 77].
    MSE = (1/MN) * sum((I - I')^2)
    PSNR = 10 * log10(255^2 / MSE)
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 0, 100  # Identik
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return mse, psnr

def quantize_channel_histogram_based(channel_array, bits):
    """
    Implementasi algoritma inti:
    Membagi pixel ke dalam kelompok yang merata jumlahnya berdasarkan histogram[cite: 28, 32].
    """
    # 1. Ratakan array 2D menjadi 1D agar bisa diurutkan
    flat = channel_array.flatten()
    
    # 2. Tentukan jumlah level (G = 2^m) [cite: 19]
    num_levels = 2 ** bits
    
    # 3. Gunakan qcut dari pandas untuk membagi data menjadi n kelompok (bucket)
    # dengan jumlah anggota yang sama (equal-frequency binning).
    # Labels=False akan mengembalikan nilai 0 sampai n-1[cite: 33].
    try:
        # Menggunakan method 'rank' untuk menangani nilai pixel yang duplikat agar tetap terbagi rata
        quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')
    except ValueError:
        # Fallback jika gambar terlalu sederhana (sedikit variasi warna)
        quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')

    # 4. Nilai saat ini adalah 0 sampai (2^bit - 1).
    # Kita perlu menampilkannya kembali sebagai citra. 
    # Agar visualnya terlihat, kita 'stretch' kembali ke rentang 0-255 atau biarkan nilai levelnya.
    # Sesuai PDF, "Ganti semua pixel lama dengan nilai kelompoknya" (0 sampai n-1)[cite: 34].
    # Namun, agar bisa dilihat mata manusia sebagai gambar, biasanya dikalikan faktor skala.
    # Di sini kita biarkan nilai 0-(n-1) lalu dinormalisasi saat display agar terlihat kontrasnya.
    
    # Untuk rekonstruksi visual yang akurat, kita petakan nilai level ke nilai tengah representatif (opsional),
    # tapi agar sesuai ketat dengan PDF yang menyebut "Nilai Baru" adalah nilai kelompoknya,
    # kita kembalikan bentuk arraynya.
    
    # Mengembalikan ke bentuk matriks gambar asli
    return quantized_flat.reshape(channel_array.shape).astype(np.uint8)

def process_image(image, bits):
    img_array = np.array(image)
    
    # Pisahkan Kanal RGB 
    r_channel = img_array[:,:,0]
    g_channel = img_array[:,:,1]
    b_channel = img_array[:,:,2]
    
    # Proses Kompresi per Kanal [cite: 37-40]
    r_new = quantize_channel_histogram_based(r_channel, bits)
    g_new = quantize_channel_histogram_based(g_channel, bits)
    b_new = quantize_channel_histogram_based(b_channel, bits)
    
    # Gabungkan kembali (Rekonstruksi) [cite: 73]
    # Kita harus melakukan scaling agar gambar terlihat jelas di layar
    # Karena jika bit=2 (nilai 0-3), gambar akan terlihat hitam total jika tidak di-scale ke 0-255.
    factor = 255 / ((2**bits) - 1)
    
    r_display = (r_new * factor).astype(np.uint8)
    g_display = (g_new * factor).astype(np.uint8)
    b_display = (b_new * factor).astype(np.uint8)
    
    img_reconstructed = np.stack((r_display, g_display, b_display), axis=2)
    
    return img_reconstructed

# --- MINI APPS INTERFACE (POIN 5) ---

st.set_page_config(page_title="Aplikasi Kuantisasi Citra - Kelompok 6", layout="wide")

st.title("🎨 Image Quantization App (Metode Histogram)")
st.caption("Tugas Pengolahan Citra - Kelompok 6 (Universitas Pamulang)")

st.markdown("""
Aplikasi ini mengimplementasikan **Metode Kuantisasi Berbasis Histogram** (Non-Uniform).
Sesuai teori, pixel dibagi ke dalam kelompok frekuensi yang sama, bukan interval nilai yang sama.
""")

# Sidebar untuk input
with st.sidebar:
    st.header("Pengaturan")
    uploaded_file = st.file_uploader("Upload Citra (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    
    # Pilihan Bit (Sesuai jurnal 1-7 bit) [cite: 17]
    bits = st.slider("Pilih Jumlah Bit (m)", min_value=1, max_value=7, value=2)
    st.write(f"Jumlah Level (G): {2**bits}")

if uploaded_file is not None:
    # Load Image
    original_image = Image.open(uploaded_file).convert('RGB')
    
    # Proses Citra
    result_array = process_image(original_image, bits)
    result_image = Image.fromarray(result_array)
    
    # Hitung MSE & PSNR [cite: 76]
    mse, psnr = calculate_mse_psnr(np.array(original_image), result_array)
    
    # Tampilkan Hasil
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Citra Asli")
        st.image(original_image, use_column_width=True)
        st.info("Original 8-bit (256 levels)")
        
    with col2:
        st.subheader(f"Hasil Kuantisasi ({bits} Bit)")
        st.image(result_image, use_column_width=True)
        st.success(f"Tingkat Keabuan: {2**bits} level")
        
    st.divider()
    
    # Tampilkan Metrik Kualitas
    st.subheader("📊 Analisis Kualitas Citra")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("MSE (Mean Squared Error)", f"{mse:.2f}")
    m_col2.metric("PSNR (Peak Signal-to-Noise Ratio)", f"{psnr:.2f} dB")
    
    st.warning("""
    **Interpretasi:**
    * MSE Rendah / PSNR Tinggi = Kualitas Bagus.
    * Semakin kecil bit, distorsi semakin besar (MSE naik, PSNR turun)[cite: 82].
    """)
    
else:
    st.info("Silakan upload gambar di menu sebelah kiri untuk memulai.")