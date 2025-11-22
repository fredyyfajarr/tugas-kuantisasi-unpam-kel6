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

# Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
        /* Mengatur padding atas agar tidak terlalu renggang */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        /* Judul Utama berwarna Biru UNPAM */
        h1 {
            color: #004aad;
            font-weight: 700;
        }
        /* Styling untuk METRIC (Kotak MSE/PSNR) agar seperti kartu */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
        }
        /* Styling Footer */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: #555;
            text-align: center;
            padding: 10px;
            font-size: 12px;
            border-top: 1px solid #ddd;
            z-index: 100;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. FUNGSI LOGIKA (BACKEND) ---

def calculate_mse_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 0, 100
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return mse, psnr

def quantize_channel_histogram_based(channel_array, bits):
    flat = channel_array.flatten()
    num_levels = 2 ** bits
    try:
        quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')
    except ValueError:
        quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')
    return quantized_flat.reshape(channel_array.shape).astype(np.uint8)

def process_image(image, bits):
    img_array = np.array(image)
    r_channel, g_channel, b_channel = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    r_new = quantize_channel_histogram_based(r_channel, bits)
    g_new = quantize_channel_histogram_based(g_channel, bits)
    b_new = quantize_channel_histogram_based(b_channel, bits)
    
    factor = 255 / ((2**bits) - 1) if bits > 0 else 1
    r_display = (r_new * factor).astype(np.uint8)
    g_display = (g_new * factor).astype(np.uint8)
    b_display = (b_new * factor).astype(np.uint8)
    
    img_reconstructed = np.stack((r_display, g_display, b_display), axis=2)
    return img_reconstructed, img_array

def extract_dominant_colors(image_array, num_colors=10):
    """Mengambil sampel warna unik yang terbentuk untuk ditampilkan di UI"""
    pixels = image_array.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    if len(unique_colors) > num_colors:
        indices = np.linspace(0, len(unique_colors)-1, num_colors, dtype=int)
        return unique_colors[indices]
    return unique_colors

# --- 3. UI / TAMPILAN (FRONTEND) ---

with st.sidebar:
    # --- LOGO SECTION ---
    try:
        col_logo1, col_logo2, col_logo3 = st.columns([1,2,1])
        with col_logo2:
            # PERBAIKAN: use_column_width diganti use_container_width
            st.image("image/logo-unpam.png", use_container_width=True)
    except Exception:
        st.warning("Logo not found")
    
    st.markdown("<h3 style='text-align: center;'>Kelompok 6</h3>", unsafe_allow_html=True)
    
    with st.expander("👥 Anggota Tim", expanded=True):
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
    
    bits = st.slider("Tingkat Kompresi (Bit)", 1, 7, 2, help="Semakin kecil bit, semakin sedikit variasi warnanya.")
    
    st.info(f"**Status:** {2**bits} Level Warna")


# --- MAIN CONTENT ---
st.title("Metode Kuantisasi Citra")
st.markdown("**Implementasi Algoritma Berbasis Histogram (Non-Uniform)**")

if uploaded_file is not None:
    
    with st.spinner('Sedang memproses histogram pixel...'):
        original_image = Image.open(uploaded_file).convert('RGB')
        result_array, original_array = process_image(original_image, bits)
        result_image = Image.fromarray(result_array)
        mse, psnr = calculate_mse_psnr(original_array, result_array)
        
        palette = extract_dominant_colors(result_array, num_colors=8)

    # --- TABS VISUALISASI ---
    tab1, tab2, tab3 = st.tabs(["🖼️ Hasil & Palet", "📊 Analisis Histogram", "📘 Teori"])

    with tab1:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                # PERBAIKAN: use_column_width diganti use_container_width
                st.image(original_image, use_container_width=True)
            with col2:
                st.subheader(f"Hasil ({bits} Bit)")
                # PERBAIKAN: use_column_width diganti use_container_width
                st.image(result_image, use_container_width=True)
        
        st.markdown("##### 🎨 Palet Warna Dominan yang Terbentuk")
        st.caption(f"Dari jutaan kemungkinan warna, citra dikompresi menjadi kombinasi warna-warna ini (Sampel {len(palette)} warna):")
        
        cols = st.columns(len(palette))
        for idx, color in enumerate(palette):
            with cols[idx]:
                color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                st.markdown(f"""
                <div style="background-color: {color_hex}; height: 40px; border-radius: 5px; border: 1px solid #ccc;" title="RGB: {color}"></div>
                """, unsafe_allow_html=True)

        st.divider()
        
        c_dl1, c_dl2, c_dl3 = st.columns([1,2,1])
        with c_dl2:
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="⬇️ Download Hasil Citra",
                data=byte_im,
                file_name=f"kuantisasi_{bits}bit.png",
                mime="image/png",
                use_container_width=True
            )

    with tab2:
        st.markdown("##### Analisis Distribusi Warna (Red Channel)")
        
        r_orig = original_array[:,:,0].flatten()
        r_res = result_array[:,:,0].flatten()
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Histogram Asli** (Rata/Halus)")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(r_orig, bins=256, color='#ff4b4b', alpha=0.6)
            ax.set_xlim([0,255])
            ax.axis('off')
            st.pyplot(fig)
            
        with c2:
            st.write(f"**Histogram Hasil** (Terkelompok jadi {2**bits} area)")
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.hist(r_res, bins=256, color='#004aad', alpha=0.7)
            ax2.set_xlim([0,255])
            ax2.axis('off')
            st.pyplot(fig2)
            
        st.info("💡 **Insight:** Perhatikan bagaimana histogram kanan memiliki celah kosong. Itu membuktikan warna telah 'ditarik' ke dalam kelompok-kelompok tertentu (Clustering).")

    with tab3:
        st.markdown("""
        ### Penjelasan Singkat
        Metode **Non-Uniform Histogram-Based Quantization** bekerja dengan cara:
        1.  **Scanning:** Menghitung frekuensi setiap warna.
        2.  **Grouping:** Membagi pixel menjadi $2^m$ kelompok yang memiliki jumlah populasi setara.
        3.  **Mapping:** Mengganti nilai pixel asli dengan nilai perwakilan kelompoknya.
        
        Berbeda dengan *Uniform Quantization* (yang membagi rata jarak nilai 0-255), metode ini lebih cerdas karena **mengalokasikan lebih banyak detail pada warna yang paling sering muncul** dalam gambar.
        """)

    # --- METRIK ERROR DI BAWAH ---
    st.subheader("📈 Skor Kualitas Citra")
    m1, m2 = st.columns(2)
    m1.metric("MSE (Error Rate)", f"{mse:.2f}", delta="- Loss" if mse > 0 else "Perfect", delta_color="inverse")
    m2.metric("PSNR (Quality)", f"{psnr:.2f} dB", delta="Low" if psnr < 30 else "High")
    
else:
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h3>👋 Selamat Datang!</h3>
        <p>Silakan upload gambar melalui menu di sebelah kiri untuk memulai demonstrasi.</p>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
    <div class="footer">
        <p>Developed by <b>Kelompok 6</b> | Teknik Informatika - Universitas Pamulang © 2025</p>
    </div>
""", unsafe_allow_html=True)