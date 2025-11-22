import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt

# --- 1. KONFIGURASI HALAMAN & CSS ---
st.set_page_config(
    page_title="Quantization App - Kelompok 6",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        h1 { color: #004aad; font-weight: 700; }
        div[data-testid="metric-container"] {
            background-color: #ffffff; border: 1px solid #e0e0e0;
            padding: 15px; border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center;
        }
        .footer {
            position: fixed; left: 0; bottom: 0; width: 100%;
            background-color: #f1f1f1; color: #555;
            text-align: center; padding: 10px; font-size: 12px;
            border-top: 1px solid #ddd; z-index: 100;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. FUNGSI LOGIKA (BACKEND) ---

def calculate_mse_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0: return 0, 100
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return mse, psnr

def quantize_channel_histogram_based(channel_array, bits):
    flat = channel_array.flatten()
    num_levels = 2 ** bits
    try:
        # qcut: Membagi data agar setiap bucket punya jumlah item sama (Equal Frequency)
        quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')
    except ValueError:
        quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')
    return quantized_flat  # Mengembalikan raw label (0, 1, 2...)

def process_image(image, bits):
    img_array = np.array(image)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Dapatkan raw labels (0 - 2^m-1)
    r_labels = quantize_channel_histogram_based(r, bits)
    g_labels = quantize_channel_histogram_based(g, bits)
    b_labels = quantize_channel_histogram_based(b, bits)
    
    # Reshape kembali ke dimensi gambar
    r_new = r_labels.reshape(r.shape)
    g_new = g_labels.reshape(g.shape)
    b_new = b_labels.reshape(b.shape)
    
    # Scaling untuk visualisasi (agar nilai 0-3 terlihat kontras jadi 0-255)
    factor = 255 / ((2**bits) - 1) if bits > 0 else 1
    
    r_disp = (r_new * factor).astype(np.uint8)
    g_disp = (g_new * factor).astype(np.uint8)
    b_disp = (b_new * factor).astype(np.uint8)
    
    img_reconstructed = np.stack((r_disp, g_disp, b_disp), axis=2)
    
    # Kembalikan juga raw labels untuk analisis statistik
    return img_reconstructed, img_array, r_new

def extract_dominant_colors(image_array, num_colors=10):
    pixels = image_array.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    if len(unique_colors) > num_colors:
        indices = np.linspace(0, len(unique_colors)-1, num_colors, dtype=int)
        return unique_colors[indices]
    return unique_colors

# --- 3. UI / TAMPILAN (FRONTEND) ---

with st.sidebar:
    try:
        col_logo1, col_logo2, col_logo3 = st.columns([1,2,1])
        with col_logo2:
            st.image("image/logo-unpam.png", use_container_width=True)
    except:
        pass
    
    st.markdown("<h3 style='text-align: center;'>Kelompok 6</h3>", unsafe_allow_html=True)
    
    with st.expander("👥 Anggota Tim", expanded=False):
        st.markdown("""
        1. Farid Nuhgraha
        2. Fredy Fajar Adi Putra
        3. Maulana Aulia Rahman
        4. Muhamad Aziz Mufashshal
        5. Muhammad Faiz Saputra
        """)
    
    st.markdown("---")
    st.header("⚙️ Konfigurasi")
    uploaded_file = st.file_uploader("Upload Citra (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    
    bits = st.slider("Tingkat Kompresi (Bit)", 1, 7, 2)
    levels = 2**bits
    
    # Tabel Referensi PDF 
    st.markdown("##### Tabel Referensi Level ")
    df_ref = pd.DataFrame({
        'Bit': [1, 2, 3, 7, 8],
        'Level': [2, 4, 8, 128, 256],
        'Range': ['0-1', '0-3', '0-7', '0-127', '0-255']
    })
    st.dataframe(df_ref, hide_index=True, use_container_width=True)

# --- MAIN CONTENT ---
st.title("Metode Kuantisasi Citra")
st.markdown("**Implementasi Algoritma Berbasis Histogram (Non-Uniform)**")

if uploaded_file is not None:
    
    with st.spinner('Sedang memproses algoritma...'):
        original_image = Image.open(uploaded_file).convert('RGB')
        # Kita ambil juga r_labels (raw data 0,1,2,3) untuk verifikasi teori
        result_array, original_array, r_labels_raw = process_image(original_image, bits)
        result_image = Image.fromarray(result_array)
        mse, psnr = calculate_mse_psnr(original_array, result_array)
        palette = extract_dominant_colors(result_array, num_colors=8)

    # TABS
    tab1, tab2, tab3 = st.tabs(["🖼️ Hasil Visual", "📊 Analisis & Verifikasi", "📝 Teori PDF"])

    with tab1:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(original_image, use_container_width=True)
            with col2:
                st.subheader(f"Hasil ({bits} Bit)")
                st.image(result_image, use_container_width=True)
        
        st.markdown(f"##### 🎨 Palet Warna ({len(palette)} Warna)")
        cols = st.columns(len(palette) if len(palette) < 10 else 10)
        for idx, color in enumerate(palette[:10]):
            with cols[idx]:
                hex_c = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                st.markdown(f'<div style="background-color:{hex_c};height:30px;border-radius:4px;border:1px solid #ccc;"></div>', unsafe_allow_html=True)

        st.divider()
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            st.download_button("⬇️ Download Hasil", buf.getvalue(), f"kuantisasi_{bits}bit.png", "image/png", use_container_width=True)

    with tab2:
        # 1. Overlay Histogram
        st.subheader("1. Histogram Overlay (Warna)")
        st.caption("Visualisasi bagaimana kurva warna asli (Merah) dipadatkan (Biru).")
        
        r_orig = original_array[:,:,0].flatten()
        r_res = result_array[:,:,0].flatten()
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(r_orig, bins=256, color='red', alpha=0.3, label='Original', density=True)
        ax.hist(r_res, bins=256, color='blue', alpha=0.6, label='Hasil Kuantisasi', histtype='step', linewidth=1.5, density=True)
        ax.legend()
        ax.axis('off')
        st.pyplot(fig)

        st.divider()

        # 2. VERIFIKASI TEORI PDF (PENTING!)
        st.subheader("2. Bukti Verifikasi 'Equal Frequency' [cite: 28]")
        st.markdown("""
        Dokumen PDF menyatakan metode ini **"Membagi pixel ke dalam kelompok yang merata jumlahnya"**.
        Grafik di bawah ini membuktikan apakah algoritma kita benar-benar membagi pixel secara rata atau tidak.
        """)
        
        # Hitung jumlah pixel di setiap level (0 sampai 2^bit - 1)
        # Flatten raw labels dan hitung frekuensinya
        unique, counts = np.unique(r_labels_raw.flatten(), return_counts=True)
        
        # Buat grafik batang
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        bars = ax2.bar(unique, counts, color='#004aad')
        ax2.set_title(f"Jumlah Pixel per Level (Harus Rata / Flat)")
        ax2.set_xlabel(f"Level Kelompok (0 - {levels-1})")
        ax2.set_ylabel("Jumlah Pixel")
        ax2.set_xticks(unique)
        st.pyplot(fig2)
        
        # Kesimpulan Otomatis
        std_dev = np.std(counts)
        avg_pixel = np.mean(counts)
        percent_var = (std_dev / avg_pixel) * 100
        
        if percent_var < 15: # Toleransi 15%
            st.success(f"✅ **TERVERIFIKASI:** Distribusi pixel SANGAT MERATA. Standar Deviasi hanya {percent_var:.1f}%. Ini sesuai dengan teori PDF tentang 'Equal Frequency Binning'.")
        else:
            st.warning("⚠️ Distribusi agak timpang (Wajar jika gambar memiliki warna solid dominan seperti background putih).")

    with tab3:
        st.header("Teori Referensi (Sesuai PDF)")
        
        st.subheader("Interpretasi Kualitas [cite: 79-82]")
        st.info("""
        * **MSE Naik** = Kualitas Turun
        * **PSNR Turun** = Kualitas Buruk
        * **Bit Makin Kecil** = Distorsi Makin Besar
        """)
        
        st.subheader("Simulasi Perhitungan Manual [cite: 86-113]")
        with st.expander("Lihat Perhitungan Data 16 Pixel"):
            st.write("Data Contoh PDF: `[10, 10, 20, 20, 20, 30, 30, 100, 100, 100, 100, 150, 150, 150, 220, 220]`")
            st.write("Total 16 pixel dibagi 4 level = **4 Pixel/Kelompok**.")
            st.markdown("""
            - **Kelompok 0:** 10, 10, 20, 20
            - **Kelompok 1:** 20, 30, 30, 100
            - **Kelompok 2:** 100, 100, 100, 150
            - **Kelompok 3:** 150, 150, 220, 220
            """)
            st.caption("Semua pixel dalam kelompok diganti nilai labelnya.")

    # --- METRIK ERROR ---
    st.subheader("📈 Analisis Kualitas")
    m1, m2 = st.columns(2)
    m1.metric("MSE", f"{mse:.2f}", delta="- Loss", delta_color="inverse")
    m2.metric("PSNR", f"{psnr:.2f} dB", delta="Quality")
    
else:
    st.markdown("<div style='text-align: center; padding: 50px;'><h3>👋 Selamat Datang!</h3><p>Upload gambar untuk memulai.</p></div>", unsafe_allow_html=True)

st.markdown('<div class="footer">Developed by <b>Kelompok 6</b> | Teknik Informatika - Universitas Pamulang © 2025</div>', unsafe_allow_html=True)