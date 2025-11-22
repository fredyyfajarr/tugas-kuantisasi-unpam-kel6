import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt

# --- KONFIGURASI HALAMAN & CSS ---
st.set_page_config(
    page_title="Aplikasi Kuantisasi - Kelompok 6",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS untuk header warna UNPAM
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            color: #004aad; /* Biru UNPAM */
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI LOGIKA (BACKEND) ---

def calculate_mse_psnr(original, compressed):
    """
    Menghitung MSE dan PSNR.
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 0, 100
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return mse, psnr

def quantize_channel_histogram_based(channel_array, bits):
    """
    Algoritma Inti: Membagi pixel menjadi kelompok dengan jumlah anggota rata (Non-Uniform).
    Menggunakan pandas qcut untuk 'Equal Frequency Binning'.
    """
    flat = channel_array.flatten()
    num_levels = 2 ** bits
    
    try:
        # qcut membagi data menjadi n kelompok dengan jumlah item yang sama
        quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')
    except ValueError:
        # Fallback jika data terlalu seragam
        quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')

    return quantized_flat.reshape(channel_array.shape).astype(np.uint8)

def process_image(image, bits):
    img_array = np.array(image)
    
    # Pisahkan Kanal RGB
    r_channel = img_array[:,:,0]
    g_channel = img_array[:,:,1]
    b_channel = img_array[:,:,2]
    
    # Proses Kompresi per Kanal
    r_new = quantize_channel_histogram_based(r_channel, bits)
    g_new = quantize_channel_histogram_based(g_channel, bits)
    b_new = quantize_channel_histogram_based(b_channel, bits)
    
    # Scaling untuk visualisasi agar terlihat di monitor
    # Nilai 0-3 (2 bit) diubah kembali ke rentang 0-255 agar kontras terlihat
    factor = 255 / ((2**bits) - 1) if bits > 0 else 1
    
    r_display = (r_new * factor).astype(np.uint8)
    g_display = (g_new * factor).astype(np.uint8)
    b_display = (b_new * factor).astype(np.uint8)
    
    # Gabung kembali menjadi citra RGB
    img_reconstructed = np.stack((r_display, g_display, b_display), axis=2)
    
    return img_reconstructed, img_array

# --- UI / TAMPILAN (FRONTEND) ---

# SIDEBAR: Identitas Kelompok & Input
with st.sidebar:
    # Logo Lokal (Diambil dari folder image/logo-unpam.png)
    # Pastikan file gambar benar-benar ada di path tersebut
    try:
        st.image("image/logo-unpam.png", width=100)
    except Exception:
        st.error("Logo tidak ditemukan. Cek folder 'image'.")
    
    st.title("Kelompok 6")
    st.info("**Anggota Kelompok:**\n"
            "1. Farid Nuhgraha\n"
            "2. Fredy Fajar Adi Putra\n"
            "3. Maulana Aulia Rahman\n"
            "4. Muhamad Aziz Mufashshal\n"
            "5. Muhammad Faiz Saputra")
    
    st.header("Pengaturan")
    uploaded_file = st.file_uploader("Upload Citra (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    
    # Slider Bit (1-7 Bit)
    bits = st.slider("Jumlah Bit (m)", 1, 7, 2)
    st.caption(f"Jumlah Level Warna: {2**bits}")

# MAIN CONTENT
st.title("Metode Kuantisasi Citra")
st.markdown("**Implementasi Algoritma Berbasis Histogram (Non-Uniform)**")

if uploaded_file is not None:
    # Load Image
    original_image = Image.open(uploaded_file).convert('RGB')
    
    # Proses Image
    result_array, original_array = process_image(original_image, bits)
    result_image = Image.fromarray(result_array)
    
    # Hitung Metrik
    mse, psnr = calculate_mse_psnr(original_array, result_array)

    # --- TABS VISUALISASI ---
    tab1, tab2, tab3 = st.tabs(["🖼️ Visualisasi Citra", "📊 Analisis Histogram", "📝 Teori Singkat"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Citra Asli")
            st.image(original_image, use_column_width=True)
            st.caption("Original 8-bit (256 level)")
            
        with col2:
            st.subheader(f"Hasil Kuantisasi ({bits} Bit)")
            st.image(result_image, use_column_width=True)
            st.caption(f"Tereduksi menjadi {2**bits} level warna")
        
        # Tombol Download
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="⬇️ Download Hasil Citra",
            data=byte_im,
            file_name=f"hasil_kuantisasi_{bits}bit.png",
            mime="image/png"
        )

    with tab2:
        st.markdown("##### Perbandingan Distribusi Warna (Channel Merah)")
        st.write("Grafik ini membuktikan bahwa metode histogram bekerja dengan mengelompokkan pixel.")
        
        # Ambil sampel channel merah untuk histogram
        r_orig = original_array[:,:,0].flatten()
        r_res = result_array[:,:,0].flatten()
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Histogram Asli**")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(r_orig, bins=256, color='red', alpha=0.5)
            ax.set_title("Distribusi Pixel 0-255")
            st.pyplot(fig)
            
        with c2:
            st.write(f"**Histogram Hasil ({2**bits} Level)**")
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            # Tampilkan histogram hasil
            ax2.hist(r_res, bins=256, color='orange', alpha=0.7)
            ax2.set_title("Pixel Terkelompok (Quantized)")
            st.pyplot(fig2)

    with tab3:
        st.markdown("""
        ### Penjelasan Metode
        
        Metode yang digunakan adalah **Non-Uniform Histogram-Based Quantization**.
        
        1. **Prinsip Dasar:**
           Berbeda dengan pemotongan biasa (uniform), metode ini membagi pixel berdasarkan **frekuensi kemunculan** di histogram.
           
        2. **Langkah Algoritma:**
           * Hitung histogram citra asli.
           * Bagi pixel menjadi $2^m$ kelompok dimana setiap kelompok memiliki jumlah pixel yang sama (Equal Frequency).
           * Ubah nilai pixel lama menjadi nilai kelompok barunya.
           
        3. **Analisis Kualitas:**
           Kami menggunakan MSE dan PSNR untuk mengukur seberapa besar informasi yang hilang (Lossy Compression).
        """)

    # --- METRIK ERROR ---
    st.divider()
    st.subheader("📈 Analisis Kualitas (Error Metrics)")
    
    m1, m2 = st.columns(2)
    m1.metric("MSE (Mean Squared Error)", f"{mse:.2f}", help="Semakin rendah semakin baik")
    m2.metric("PSNR (Peak Signal-to-Noise Ratio)", f"{psnr:.2f} dB", help="Semakin tinggi semakin baik")
    
    if psnr < 30:
        st.warning(f"Note: PSNR {psnr:.2f} dB menunjukkan kualitas citra menurun signifikan, wajar untuk kompresi {bits} bit.")

else:
    st.info("👋 Halo Kelompok 6! Silakan upload gambar di menu sebelah kiri untuk memulai demo.")