import streamlit as st
from PIL import Image
import io  # <--- Perlu ini untuk hitung size
from models.model import QuantizationModel
from views.view import AppView

class AppController:
    """
    CONTROLLER: Penghubung Model & View
    """
    def __init__(self):
        self.model = QuantizationModel()
        self.view = AppView()

    def run(self):
        self.view.setup_page()
        uploaded_file, bits = self.view.render_sidebar()
        self.view.render_header()
        
        if uploaded_file is not None:
            with st.status("Sedang memproses citra...", expanded=True) as status:
                st.write("📂 Membaca file gambar...")
                
                # --- HITUNG SIZE ASLI ---
                # uploaded_file.size memberikan ukuran dalam bytes
                orig_file_size = uploaded_file.size
                # ------------------------
                
                orig_img = Image.open(uploaded_file).convert('RGB')
                
                st.write("📐 Menyesuaikan resolusi (Anti-Lag)...")
                if orig_img.width > 1500 or orig_img.height > 1500:
                     orig_img.thumbnail((1500, 1500))
                
                st.write(f"🧮 Menjalankan algoritma kuantisasi ({bits} Bit)...")
                result_data = self.model.process_image_cached(orig_img, bits)
                
                # --- HITUNG SIZE HASIL ---
                st.write("💾 Menghitung ukuran file hasil...")
                buf = io.BytesIO()
                # Simpan ke memori sebagai PNG (sesuai tombol download) untuk tahu ukurannya
                result_data['reconstructed_img'].save(buf, format="PNG")
                compressed_file_size = buf.tell() # Ukuran dalam bytes
                
                # Hitung Persentase & Selisih
                size_diff = orig_file_size - compressed_file_size
                compression_ratio = (size_diff / orig_file_size) * 100
                
                # Simpan data size ke dictionary result agar rapi
                size_stats = {
                    'orig': orig_file_size,
                    'compressed': compressed_file_size,
                    'diff': size_diff,
                    'percent': compression_ratio
                }
                # -------------------------
                
                st.write("📊 Menghitung statistik & metrik...")
                mse, psnr = self.model.calculate_mse_psnr(
                    result_data['original_array'], 
                    result_data['reconstructed_array']
                )
                palette = self.model.extract_palette(result_data['reconstructed_array'], 8)
                decode_stats = self.model.get_decode_stats(result_data['raw_labels_r'])
                codebook = self.model.get_codebook(
                    result_data['original_array'][:,:,0],
                    result_data['raw_labels_r']
                )

                status.update(label="Proses Selesai!", state="complete", expanded=False)
            
            # Kirim 'size_stats' ke View
            self.view.render_dashboard(bits, mse, psnr, size_stats)
            self.view.render_tabs(orig_img, result_data, bits, palette, decode_stats, codebook)
        
        else:
            self.view.render_empty_state()
            
        self.view.render_footer()