import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
from streamlit_image_comparison import image_comparison

class AppView:
    def setup_page(self):
        st.set_page_config(page_title="App Kuantisasi MVC", page_icon="🎓", layout="wide")
        st.markdown("""
            <style>
                .block-container { padding-top: 1rem; padding-bottom: 3rem; }
                h1 { color: #004aad; font-family: 'Helvetica', sans-serif; font-weight: 800; }
                div[data-testid="metric-container"] {
                    background-color: #f8f9fa; 
                    border: 1px solid #e0e0e0;
                    padding: 10px 15px; 
                    border-radius: 8px;
                    border-left: 5px solid #004aad;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
                st.markdown("""
                - Farid Nuhgraha
                - Fredy Fajar Adi Putra
                - Maulana Aulia Rahman
                - Muhamad Aziz Mufashshal
                - Muhammad Faiz Saputra
                - Ravail Shodikin
                """)
            
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

    def format_bytes(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0: return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} GB"

    def render_dashboard(self, bits, mse, psnr, size_stats):
        str_orig = self.format_bytes(size_stats['orig'])
        str_comp = self.format_bytes(size_stats['compressed'])
        str_diff = self.format_bytes(size_stats['diff'])
        
        is_reduced = size_stats['diff'] > 0
        delta_color = "normal" if is_reduced else "inverse"
        delta_sym = "↓" if is_reduced else "↑"
        
        st.subheader("📊 Statistik Kualitas & Ukuran File")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bit Depth", f"{bits} Bit", f"{2**bits} Level")
        m2.metric("MSE (Error)", f"{mse:.1f}", delta="-Lossy" if mse>0 else "Perfect", delta_color="inverse", help="Mean Squared Error. Semakin kecil semakin baik.")
        m3.metric("PSNR (Kualitas)", f"{psnr:.2f} dB", delta="Low" if psnr<30 else "High", help="> 30 dB: Bagus\n20-30 dB: Sedang\n< 20 dB: Buruk")
        m4.metric("Ukuran File", str_comp, f"{delta_sym} {str_diff} ({size_stats['percent']:.1f}%)", delta_color=delta_color, help=f"Asli: {str_orig}\nHasil: {str_comp}")
        st.divider()

    def render_tabs(self, orig_img, data, bits, palette, decode_stats, codebook):
        t1, t2, t3, t4 = st.tabs(["🖼️ Hasil (Slider)", "🎨 Bedah Kanal", "📊 Analisis & Decode", "📘 Teori & Alur"])
        
        with t1:
            st.write("Geser slider di gambar untuk membandingkan:")
            image_comparison(
                img1=orig_img,
                img2=data['reconstructed_img'],
                label1="Original",
                label2=f"Hasil {bits}-Bit",
                starting_position=50,
                show_labels=True, make_responsive=True, in_memory=True
            )
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

        with t2:
            st.info("Visualisasi Kanal RGB Terpisah.")
            r, g, b = data['channels_display']
            c_r, c_g, c_b = st.columns(3)
            c_r.image(r, caption="Red", width="stretch", clamp=True)
            c_g.image(g, caption="Green", width="stretch", clamp=True)
            c_b.image(b, caption="Blue", width="stretch", clamp=True)

        with t3:
            st.subheader("1. Struktur Data Hasil Decode (Raw)")
            st.caption(f"Pembuktian bahwa komputer hanya menyimpan indeks **0 sampai {2**bits-1}**.")
            
            c_raw1, c_raw2, c_raw3 = st.columns([2, 1, 1])
            with c_raw1:
                st.markdown("**Matriks Data (Sampel 15x15):**")
                raw_sample = data['raw_labels_r'][:15, :15]
                st.dataframe(pd.DataFrame(raw_sample).style.background_gradient(cmap='Blues'), height=300, use_container_width=True)
            
            with c_raw2:
                st.markdown("**Statistik Label:**")
                st.dataframe(decode_stats, hide_index=True, use_container_width=True)

            with c_raw3:
                st.markdown("**Kamus (Codebook):**")
                st.dataframe(codebook, hide_index=True, use_container_width=True)
                st.caption("Nilai asli yang diwakili oleh setiap label.")

            st.divider()
            st.subheader("2. Histogram Overlay")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(data['original_array'][:,:,0].flatten(), bins=256, color='red', alpha=0.3, label='Original', density=True)
            ax.hist(data['reconstructed_array'][:,:,0].flatten(), bins=256, color='blue', alpha=0.7, label='Hasil', histtype='step', linewidth=1.5, density=True)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig) # CLEANUP MEMORY

        with t4:
            st.subheader("1. Alur Sistem (Flowchart)")
            st.graphviz_chart("""
                digraph {
                    rankdir=LR;
                    node [shape=box, style=filled, fillcolor="#f0f2f6", fontname="Helvetica"];
                    Start [shape=oval, fillcolor="#d1e7dd", label="Mulai"];
                    Input [label="Upload Citra\n(RGB)"];
                    Resize [label="Resize Otomatis\n(Max 1500px)"];
                    Process [label="Algoritma Kuantisasi\n(Equal Frequency)"];
                    Stats [label="Hitung Metrik\n(MSE, PSNR, Size)"];
                    Output [label="Tampilkan Hasil\n(View)"];
                    End [shape=oval, fillcolor="#f8d7da", label="Selesai"];
                    Start -> Input -> Resize -> Process -> Stats -> Output -> End;
                }
            """)
            st.divider()
            st.subheader("2. Simulasi Manual")
            st.table(pd.DataFrame({
                'Intensitas': [10, 20, 30, 100, 150, 220],
                'Freq': [2, 3, 2, 4, 3, 2],
                'Kelompok': ['0', '0/1', '1', '1/2', '2/3', '3'],
                'Label': ['0', '0/1', '1', '1/2', '2/3', '3']
            }))

    def render_footer(self):
        st.markdown('<div class="footer">Teknik Informatika - Universitas Pamulang © 2025 | Kelompok 6</div>', unsafe_allow_html=True)