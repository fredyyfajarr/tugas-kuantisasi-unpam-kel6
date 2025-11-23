import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
from streamlit_image_comparison import image_comparison  # <--- LIBRARY BARU

class AppView:
    """
    VIEW: Menangani tampilan UI.
    """
    def setup_page(self):
        st.set_page_config(page_title="App Kuantisasi MVC", page_icon="🎓", layout="wide")
        st.markdown("""
            <style>
                .block-container { padding-top: 1rem; padding-bottom: 3rem; }
                h1 { color: #004aad; font-family: 'Helvetica', sans-serif; font-weight: 800; }
                div[data-testid="metric-container"] {
                    background-color: #ffffff; border-left: 5px solid #004aad;
                    padding: 10px 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                }
                .footer {
                    position: fixed; left: 0; bottom: 0; width: 100%;
                    background-color: #004aad; color: white; text-align: center;
                    padding: 8px; font-size: 13px; z-index: 999;
                }
            </style>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        with st.sidebar:
            try:
                c1, c2, c3 = st.columns([1,2,1])
                c2.image("image/logo-unpam.png", width="stretch")
            except: pass
            
            st.markdown("<div style='text-align: center; font-weight: bold;'>KELOMPOK 6</div>", unsafe_allow_html=True)
            with st.expander("👨‍💻 Anggota Tim"):
                st.markdown("- Farid Nuhgraha\n- Fredy Fajar Adi Putra\n- Maulana Aulia Rahman\n- Muhamad Aziz Mufashshal\n- Muhammad Faiz Saputra")
            
            st.divider()
            st.header("⚙️ Kontrol")
            uploaded_file = st.file_uploader("Upload Citra", type=['jpg', 'png', 'jpeg'])
            bits = st.select_slider("Tingkat Kompresi (Bit)", options=[1, 2, 3, 4, 5, 6, 7], value=2)
            
            st.markdown("---")
            st.markdown("##### 📚 Referensi Level")
            df_ref = pd.DataFrame({'Bit': [1,2,3,7,8], 'Level': [2,4,8,128,256], 'Range': ['0-1','0-3','0-7','0-127','0-255']})
            st.dataframe(df_ref, hide_index=True, width="stretch")
            
            return uploaded_file, bits

    def render_header(self):
        st.title("Metode Kuantisasi Citra")
        st.markdown("**Implementasi Algoritma Histogram (Non-Uniform) - Universitas Pamulang**")

    def render_empty_state(self):
        st.markdown("""<div style="text-align:center; padding: 40px; background-color: #f8f9fa; border-radius: 10px; margin-top: 20px;">
            <h2 style="color: #004aad;">👋 Selamat Datang!</h2><p>Silakan upload gambar untuk memulai.</p></div>""", unsafe_allow_html=True)

    def render_dashboard(self, bits, mse, psnr):
        levels = 2**bits
        m1, m2, m3 = st.columns(3)
        m1.metric("Bit Depth", f"{bits} Bit", f"{levels} Level")
        m2.metric("MSE (Error)", f"{mse:.1f}", delta="-Lossy" if mse>0 else "Perfect", delta_color="inverse")
        m3.metric("PSNR (Quality)", f"{psnr:.2f} dB", delta="Low" if psnr<30 else "High")

    def render_tabs(self, orig_img, data, bits, palette):
        t1, t2, t3, t4 = st.tabs(["🖼️ Hasil (Slider)", "🎨 Bedah Kanal", "📊 Analisis", "📘 Teori"])
        
        with t1: # Tab Hasil dengan SLIDER KEREN
            st.write("Geser garis vertikal di tengah gambar untuk melihat perbedaan **Original vs Hasil**.")
            
            # --- FITUR BARU: IMAGE COMPARISON SLIDER ---
            # Kita perlu convert PIL Image ke format yang bisa dibaca library ini
            image_comparison(
                img1=orig_img,
                img2=data['reconstructed_img'],
                label1="Original 8-Bit",
                label2=f"Kuantisasi {bits}-Bit",
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )
            # -------------------------------------------
            
            st.divider()
            st.write("**Sampel Palet Warna:**")
            cols = st.columns(len(palette))
            for i, c in enumerate(palette):
                hex_c = '#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2])
                cols[i].markdown(f'<div style="background-color:{hex_c};height:25px;border-radius:3px;"></div>', unsafe_allow_html=True)
            
            st.divider()
            b1, b2, b3 = st.columns([1,2,1])
            with b2:
                buf = io.BytesIO()
                data['reconstructed_img'].save(buf, format="PNG")
                st.download_button("⬇️ Download Hasil", buf.getvalue(), f"hasil_{bits}bit.png", "image/png", use_container_width=True)

        with t2: # Tab Kanal
            st.info("Visualisasi Kanal RGB Terpisah.")
            r, g, b = data['channels_display']
            c_r, c_g, c_b = st.columns(3)
            c_r.image(r, caption="Red", width="stretch", clamp=True)
            c_g.image(g, caption="Green", width="stretch", clamp=True)
            c_b.image(b, caption="Blue", width="stretch", clamp=True)

        with t3: # Tab Analisis
            st.subheader("1. Histogram Overlay")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(data['original_array'][:,:,0].flatten(), bins=256, color='red', alpha=0.3, label='Original', density=True)
            ax.hist(data['reconstructed_array'][:,:,0].flatten(), bins=256, color='blue', alpha=0.7, label='Hasil', histtype='step', linewidth=1.5, density=True)
            ax.legend(); st.pyplot(fig)
            st.divider()
            st.subheader("2. Bukti Pemerataan")
            unique, counts = np.unique(data['raw_labels_r'].flatten(), return_counts=True)
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.bar(unique, counts, color='#004aad', alpha=0.8)
            ax2.axhline(y=np.mean(counts), color='red', linestyle='--', label='Target Rata-rata')
            ax2.legend(); st.pyplot(fig2)

        with t4: # Tab Teori
            st.subheader("Simulasi Manual")
            st.table(pd.DataFrame({
                'Intensitas': [10, 20, 30, 100, 150, 220],
                'Freq': [2, 3, 2, 4, 3, 2],
                'Kelompok': ['0', '0/1', '1', '1/2', '2/3', '3'],
                'Label': ['0', '0/1', '1', '1/2', '2/3', '3']
            }))

    def render_footer(self):
        st.markdown('<div class="footer">Teknik Informatika - Universitas Pamulang © 2025 | Kelompok 6</div>', unsafe_allow_html=True)