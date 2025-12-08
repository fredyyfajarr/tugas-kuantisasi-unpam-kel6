import streamlit as st
from PIL import Image
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
            # Tidak pakai spinner agar terasa lebih responsif
            orig_img = Image.open(uploaded_file).convert('RGB')
            
            # --- [UPDATE: KUALITAS HD] ---
            # Kita naikkan batasnya ke 1500px (sebelumnya 400px).
            # Ini membuat gambar TAJAM kembali, tapi tetap mencegah
            # gambar raksasa (misal 4000px) bikin macet aplikasi.
            if orig_img.width > 1500 or orig_img.height > 1500:
                 orig_img.thumbnail((1500, 1500))
            # -----------------------------
            
            # 1. Proses Gambar (Cached)
            result_data = self.model.process_image_cached(orig_img, bits)
            
            # 2. Hitung Metrik & Palette
            mse, psnr = self.model.calculate_mse_psnr(
                result_data['original_array'], 
                result_data['reconstructed_array']
            )
            palette = self.model.extract_palette(result_data['reconstructed_array'], 8)
            
            # 3. Tampilkan View
            self.view.render_dashboard(bits, mse, psnr)
            self.view.render_tabs(orig_img, result_data, bits, palette)
        
        else:
            self.view.render_empty_state()
            
        self.view.render_footer()