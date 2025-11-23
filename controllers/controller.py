import streamlit as st
from PIL import Image
import time  # <--- 1. Tambah ini

from models.model import QuantizationModel
from views.view import AppView

class AppController:
    def __init__(self):
        self.model = QuantizationModel()
        self.view = AppView()

    def run(self):
        self.view.setup_page()
        uploaded_file, bits = self.view.render_sidebar()
        self.view.render_header()
        
        if uploaded_file is not None:
            # Tidak perlu spinner di sini karena kita mau ukur kecepatan aslinya
            orig_img = Image.open(uploaded_file).convert('RGB')
            
            # --- MULAI STOPWATCH ---
            start_time = time.time() 
            
            # PROSES BERAT (YANG DI-CACHE)
            result_data = self.model.process_image_cached(orig_img, bits)
            
            # --- STOP STOPWATCH ---
            end_time = time.time()
            durasi = end_time - start_time
            
            # CETAK HASIL KE TERMINAL
            print(f"⏱️ [Bit: {bits}] Waktu Proses: {durasi:.6f} detik") 
            # Jika durasi 0.000000, berarti diambil dari CACHE (Super Cepat)

            # Lanjut proses ringan lainnya...
            mse, psnr = self.model.calculate_mse_psnr(
                result_data['original_array'], 
                result_data['reconstructed_array']
            )
            palette = self.model.extract_palette(result_data['reconstructed_array'], 8)
            
            self.view.render_dashboard(bits, mse, psnr)
            self.view.render_tabs(orig_img, result_data, bits, palette)
            
            # Tampilkan info kecepatan di Web juga (Opsional, buat pamer ke dosen)
            if durasi < 0.001:
                st.toast(f"⚡ INSTAN! Diambil dari Cache ({durasi:.4f} det)", icon="🚀")
            else:
                st.toast(f"🐢 Menghitung baru... ({durasi:.4f} det)", icon="⚙️")
        
        else:
            self.view.render_empty_state()
            
        self.view.render_footer()