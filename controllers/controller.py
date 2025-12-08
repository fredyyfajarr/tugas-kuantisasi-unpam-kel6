import streamlit as st
from PIL import Image
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
            # Tidak perlu spinner lama-lama karena sekarang super cepat
            orig_img = Image.open(uploaded_file).convert('RGB')
            
            # --- [SOLUSI ANTI-HANG] ---
            # Resize ke 400px. Ini sangat ringan (hanya 160rb pixel vs 1 juta pixel).
            orig_img.thumbnail((400, 400)) 
            # --------------------------
            
            # 1. Proses Gambar
            result_data = self.model.process_image_cached(orig_img, bits)
            
            # 2. Hitung Metrik
            mse, psnr = self.model.calculate_mse_psnr(
                result_data['original_array'], 
                result_data['reconstructed_array']
            )
            palette = self.model.extract_palette(result_data['reconstructed_array'], 8)
            
            # HAPUS bagian get_3d_plot_data (Ini sumber beratnya)
            
            # 3. Tampilkan View (Tanpa data 3D)
            self.view.render_dashboard(bits, mse, psnr)
            self.view.render_tabs(orig_img, result_data, bits, palette)
        
        else:
            self.view.render_empty_state()
            
        self.view.render_footer()