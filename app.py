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

# Custom CSS untuk UI
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        h1 { color: #004aad; font-weight: 700; }
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
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
        quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')
    except ValueError:
        quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')
    return quantized_flat.reshape(channel_array.shape).astype(np.uint8)

def process_image(image, bits):
    img_array = np.array(image)
    # Pisahkan channel
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Proses kuantisasi per channel
    r_new = quantize_channel_histogram_based(r, bits)
    g_new = quantize_channel_histogram_based(g, bits)
    b_new = quantize_channel_histogram_based(b, bits)
    
    # Scaling visualisasi (0-3 menjadi 0-255)
    factor = 255 / ((2**bits) - 1) if bits > 0 else 1
    
    r_disp = (r_new * factor).astype(np.uint8)
    g_disp = (g_new * factor).astype(np.uint8)
    b_disp = (b_new * factor).astype(np.uint8)
    
    img_reconstructed = np.stack((r_disp, g_disp, b_disp), axis=2)
    return img_reconstructed, img_array

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
    except Exception:
        st.warning("Logo not found")
    
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
    
    bits = st.slider("Tingkat Kompresi (Bit)", 1, 7, 2, help="Mengubah jumlah bit per pixel (m).")
    st.info(f"**Level Warna Baru (G):** {2**bits}")

# --- MAIN CONTENT ---
st.title("Metode Kuantisasi Citra")
st.markdown("**Implementasi Algoritma Berbasis Histogram (Non-Uniform)**")

if uploaded_file is not None:
    
    with st.spinner('Memproses histogram & menghitung MSE...'):
        original_image = Image.open(uploaded_file).convert('RGB')
        result_array, original_array = process_image(original_image, bits)
        result_image = Image.fromarray(result_array)
        mse, psnr = calculate_mse_psnr(original_array, result_array)
        palette = extract_dominant_colors(result_array, num_colors=8)

    # TABS
    tab1, tab2, tab3 = st.tabs(["🖼️ Hasil & Visual", "📊 Analisis Histogram", "📝 Teori & Simulasi"])

    with tab1:
        # Layout Gambar
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(original_image, use_container_width=True)
            with col2:
                st.subheader(f"Hasil ({bits} Bit)")
                st.image(result_image, use_container_width=True)
        
        # Palet Warna
        st.markdown(f"##### 🎨 Palet {len(palette)} Warna Dominan")
        cols = st.columns(len(palette))
        for idx, color in enumerate(palette):
            with cols[idx]:
                hex_c = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                st.markdown(f'<div style="background-color:{hex_c};height:30px;border-radius:4px;border:1px solid #ccc;"></div>', unsafe_allow_html=True)

        st.divider()
        
        # Download
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            st.download_button("⬇️ Download Hasil", buf.getvalue(), f"kuantisasi_{bits}bit.png", "image/png", use_container_width=True)

    with tab2:
        st.markdown("##### Perbandingan Distribusi Pixel (Channel Merah)")
        st.caption("Grafik ini menunjukkan bagaimana distribusi warna asli (Merah) dipaksa mengelompok (Biru).")
        
        r_orig = original_array[:,:,0].flatten()
        r_res = result_array[:,:,0].flatten()
        
        # Matplotlib Overlay Chart
        fig, ax = plt.subplots(figsize=(8, 4))
        # Histogram Asli (Area Merah)
        ax.hist(r_orig, bins=256, color='red', alpha=0.3, label='Original (Detail)', density=True)
        # Histogram Hasil (Garis Biru)
        ax.hist(r_res, bins=256, color='blue', alpha=0.6, label=f'Hasil {bits} Bit (Kasar)', histtype='step', linewidth=1.5, density=True)
        
        ax.set_title("Overlay Histogram: Original vs Quantized")
        ax.set_xlabel("Nilai Intensitas (0-255)")
        ax.set_ylabel("Frekuensi (Density)")
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        st.pyplot(fig)
        st.info(f"Terlihat pada kurva biru, warna tidak lagi halus, melainkan melonjak pada titik-titik tertentu. Inilah efek **Non-Uniform Quantization** dimana {256} level warna dipadatkan menjadi {2**bits} level saja.")

    with tab3:
        st.header("Dasar Teori & Simulasi")
        
        # Penjelasan Rumus dengan LaTeX
        st.subheader("1. Rumus Kualitas Citra")
        st.markdown("Sesuai makalah, kualitas diukur menggunakan MSE dan PSNR [cite: 76-77]:")
        
        c_eq1, c_eq2 = st.columns(2)
        with c_eq1:
            st.latex(r"MSE = \frac{1}{MN}\sum_{i=0}^{M-1}\sum_{j=0}^{N-1}[I(i,j) - I'(i,j)]^2")
            st.caption("**Mean Squared Error:** Rata-rata kesalahan kuadrat.")
        with c_eq2:
            st.latex(r"PSNR = 10 \cdot \log_{10}\left(\frac{255^2}{MSE}\right)")
            st.caption("**Peak Signal-to-Noise Ratio:** Rasio sinyal terhadap noise.")

        st.divider()

        # Simulasi Contoh Manual (Sesuai PDF Hal 5-6)
        st.subheader("2. Simulasi Perhitungan Manual")
        st.markdown("Contoh perhitungan manual algoritma histogram sesuai **halaman 5-6 makalah** [cite: 83-113]:")
        
        with st.expander("Buka Simulasi Data 16 Pixel", expanded=True):
            st.write("**Data Awal (16 Pixel):**")
            # Data contoh dari PDF [cite: 87]
            data_contoh = [10, 10, 20, 20, 20, 30, 30, 100, 100, 100, 100, 150, 150, 150, 220, 220]
            st.code(f"{data_contoh}", language="python")
            
            st.write(f"**Target:** Kompresi ke 2 Bit (4 Level). Maka {16}/4 = **4 pixel per kelompok**.")
            
            st.write("**Hasil Pengelompokan (Equal Frequency):**")
            col_sim1, col_sim2, col_sim3, col_sim4 = st.columns(4)
            col_sim1.success("Kelompok 0\n\n10, 10, 20, 20")
            col_sim2.warning("Kelompok 1\n\n20, 30, 30, 100")
            col_sim3.info("Kelompok 2\n\n100, 100, 100, 150")
            col_sim4.error("Kelompok 3\n\n150, 150, 220, 220")
            
            st.caption("*Catatan: Nilai pixel akan diganti dengan label kelompoknya (0, 1, 2, 3).*")

    # --- METRIK ERROR ---
    st.subheader("📈 Skor Kualitas")
    m1, m2 = st.columns(2)
    m1.metric("MSE", f"{mse:.2f}", delta="- Loss", delta_color="inverse")
    m2.metric("PSNR", f"{psnr:.2f} dB", delta="Quality")

else:
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h3>👋 Selamat Datang!</h3>
        <p>Silakan upload gambar melalui menu di sebelah kiri.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Developed by <b>Kelompok 6</b> | Teknik Informatika - Universitas Pamulang © 2025</div>', unsafe_allow_html=True)