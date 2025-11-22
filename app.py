import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt

# --- 1. KONFIGURASI HALAMAN & CSS ---
st.set_page_config(
    page_title="App Kuantisasi - Kelompok 6",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 3rem; }
        h1 { color: #004aad; font-family: 'Helvetica', sans-serif; font-weight: 800; }
        
        /* Styling Metric Dashboard */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border-left: 5px solid #004aad;
            padding: 10px 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        /* Footer Styling */
        .footer {
            position: fixed; left: 0; bottom: 0; width: 100%;
            background-color: #004aad; color: white;
            text-align: center; padding: 8px; font-size: 13px;
            z-index: 999;
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
    """Implementasi Logic PDF: Equal Frequency Binning"""
    flat = channel_array.flatten()
    num_levels = 2 ** bits
    try:
        # qcut membagi data agar jumlah pixel per kelompok SAMA RATA
        quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')
    except ValueError:
        quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')
    return quantized_flat

def process_image(image, bits):
    img_array = np.array(image)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # 1. Proses Per Kanal (Sesuai PDF Hal 3)
    r_labels = quantize_channel_histogram_based(r, bits)
    g_labels = quantize_channel_histogram_based(g, bits)
    b_labels = quantize_channel_histogram_based(b, bits)
    
    # Reshape kembali ke dimensi gambar
    r_new = r_labels.reshape(r.shape)
    g_new = g_labels.reshape(g.shape)
    b_new = b_labels.reshape(b.shape)
    
    # Scaling untuk visualisasi
    factor = 255 / ((2**bits) - 1) if bits > 0 else 1
    
    r_disp = (r_new * factor).astype(np.uint8)
    g_disp = (g_new * factor).astype(np.uint8)
    b_disp = (b_new * factor).astype(np.uint8)
    
    img_reconstructed = np.stack((r_disp, g_disp, b_disp), axis=2)
    
    # Return detail lengkap untuk visualisasi
    return img_reconstructed, img_array, (r_disp, g_disp, b_disp), r_labels

def extract_palette(image_array, num=10):
    pixels = image_array.reshape(-1, 3)
    unique = np.unique(pixels, axis=0)
    if len(unique) > num:
        idx = np.linspace(0, len(unique)-1, num, dtype=int)
        return unique[idx]
    return unique

# --- 3. UI / TAMPILAN (FRONTEND) ---

with st.sidebar:
    # --- LOGO & TIM ---
    try:
        c1, c2, c3 = st.columns([1,2,1])
        c2.image("image/logo-unpam.png", use_container_width=True)
    except:
        pass
    
    st.markdown("<div style='text-align: center; font-weight: bold;'>KELOMPOK 6</div>", unsafe_allow_html=True)
    
    with st.expander("👨‍💻 Anggota Tim", expanded=False):
        st.markdown("""
        - Farid Nuhgraha
        - Fredy Fajar Adi Putra
        - Maulana Aulia Rahman
        - Muhamad Aziz Mufashshal
        - Muhammad Faiz Saputra
        """)
    
    st.divider()
    st.header("⚙️ Kontrol")
    uploaded_file = st.file_uploader("Upload Citra", type=['jpg', 'png', 'jpeg'])
    
    # Slider
    bits = st.select_slider("Tingkat Kompresi (Bit)", options=[1, 2, 3, 4, 5, 6, 7], value=2)
    levels = 2**bits
    
    st.info(f"**Target:** {levels} Warna")
    
    # --- TABEL REFERENSI (DIKEMBALIKAN SESUAI REQUEST) ---
    st.markdown("---")
    st.markdown("##### 📚 Referensi Level")
    df_ref = pd.DataFrame({
        'Bit': [1, 2, 3, 7, 8],
        'Level': [2, 4, 8, 128, 256],
        'Range': ['0-1', '0-3', '0-7', '0-127', '0-255']
    })
    # Menampilkan tabel referensi sesuai PDF hal 2
    st.dataframe(df_ref, hide_index=True, use_container_width=True)

# --- MAIN CONTENT ---
st.title("Metode Kuantisasi Citra")
st.markdown("**Implementasi Algoritma Histogram (Non-Uniform) - Universitas Pamulang**")

if uploaded_file:
    with st.spinner('Memecah kanal RGB & menghitung histogram...'):
        orig_img = Image.open(uploaded_file).convert('RGB')
        # Panggil fungsi processing
        res_arr, orig_arr, (r_d, g_d, b_d), r_raw = process_image(orig_img, bits)
        res_img = Image.fromarray(res_arr)
        mse, psnr = calculate_mse_psnr(orig_arr, res_arr)
        palette = extract_palette(res_arr, 8)

    # --- METRICS DASHBOARD ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Bit Depth", f"{bits} Bit", f"{levels} Level")
    m2.metric("MSE (Error)", f"{mse:.1f}", delta="-Lossy" if mse>0 else "Perfect", delta_color="inverse")
    m3.metric("PSNR (Quality)", f"{psnr:.2f} dB", delta="Low" if psnr<30 else "High")

    # --- TABS ---
    tab_res, tab_chn, tab_ana, tab_teo = st.tabs(["🖼️ Hasil Akhir", "🎨 Bedah Kanal RGB", "📊 Analisis", "📘 Teori"])

    with tab_res:
        c_orig, c_res = st.columns(2)
        with c_orig:
            st.subheader("Original")
            st.image(orig_img, use_container_width=True)
        with c_res:
            st.subheader(f"Hasil Kuantisasi ({bits} Bit)")
            st.image(res_img, use_container_width=True)
        
        # Palet Warna
        st.markdown(f"**Sampel Palet Warna yang Terbentuk:**")
        cols = st.columns(len(palette))
        for i, color in enumerate(palette):
            hex_c = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            cols[i].markdown(f'<div style="background-color:{hex_c};height:25px;border-radius:3px;"></div>', unsafe_allow_html=True)
        
        # --- TOMBOL DOWNLOAD (CENTERED & DIVIDER) ---
        st.divider()
        c_dl1, c_dl2, c_dl3 = st.columns([1,2,1])
        with c_dl2:
            buf = io.BytesIO()
            res_img.save(buf, format="PNG")
            st.download_button("⬇️ Download Hasil Citra", buf.getvalue(), f"hasil_{bits}bit.png", "image/png", use_container_width=True)

    with tab_chn:
        st.info("Sesuai **PDF Halaman 3 (Poin 1.3)**: Proses dilakukan terpisah pada setiap kanal (R, G, B) lalu digabungkan kembali.")
        col_r, col_g, col_b = st.columns(3)
        with col_r: st.image(r_d, caption=f"Channel Merah", use_container_width=True, clamp=True)
        with col_g: st.image(g_d, caption=f"Channel Hijau", use_container_width=True, clamp=True)
        with col_b: st.image(b_d, caption=f"Channel Biru", use_container_width=True, clamp=True)

    with tab_ana:
        # --- 1. HISTOGRAM OVERLAY (DIKEMBALIKAN) ---
        st.subheader("1. Histogram Overlay (Warna)")
        st.caption("Grafik ini menunjukkan bagaimana kurva warna asli (Merah) dipaksa menjadi kotak-kotak (Biru).")
        
        r_orig = orig_arr[:,:,0].flatten()
        r_res = res_arr[:,:,0].flatten()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        # Plot Asli (Area Merah)
        ax.hist(r_orig, bins=256, color='red', alpha=0.3, label='Original (Detail)', density=True)
        # Plot Hasil (Garis Biru)
        ax.hist(r_res, bins=256, color='blue', alpha=0.7, label=f'Hasil {bits} Bit', histtype='step', linewidth=1.5, density=True)
        
        ax.legend()
        ax.set_title("Perbandingan Distribusi Pixel (Channel Merah)")
        ax.set_xlabel("Nilai Intensitas (0-255)")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

        st.divider()

        # --- 2. BUKTI PEMERATAAN ---
        st.subheader("2. Bukti Pemerataan Pixel (Equal Frequency)")
        st.markdown("Verifikasi bahwa setiap kelompok memiliki jumlah pixel yang rata (Sesuai teori PDF).")
        
        unique, counts = np.unique(r_raw.flatten(), return_counts=True)
        
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        bars = ax2.bar(unique, counts, color='#004aad', alpha=0.8)
        ax2.axhline(y=np.mean(counts), color='red', linestyle='--', label='Target Rata-rata')
        ax2.set_xticks(unique)
        ax2.legend()
        st.pyplot(fig2)

    with tab_teo:
        st.subheader("Simulasi Manual (Sesuai PDF Hal 6)")
        # Tabel Manual persis PDF
        data_manual = {
            'Intensitas Lama': [10, 20, 30, 100, 150, 220],
            'Frekuensi (Jlh Pixel)': [2, 3, 2, 4, 3, 2],
            'Kelompok Baru': ['0', '0 atau 1', '1', '1 atau 2', '2 atau 3', '3'],
            'Nilai Baru (Label)': [0, '0/1', 1, '1/2', '2/3', 3]
        }
        st.table(pd.DataFrame(data_manual))
        st.caption("Tabel ini mereplikasi perhitungan manual dari dokumen referensi.")

else:
    st.markdown("""
    <div style="text-align:center; padding: 40px; background-color: #f8f9fa; border-radius: 10px; margin-top: 20px;">
        <h2 style="color: #004aad;">👋 Selamat Datang, Kelompok 6!</h2>
        <p>Aplikasi ini siap mendemonstrasikan metode <b>Kuantisasi Histogram</b>.</p>
        <p>Silakan upload gambar di menu sebelah kiri (sidebar) untuk memulai.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Teknik Informatika - Universitas Pamulang © 2025 | Kelompok 6</div>', unsafe_allow_html=True)