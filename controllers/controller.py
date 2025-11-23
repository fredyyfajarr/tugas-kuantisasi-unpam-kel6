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
            with st.spinner('Memproses citra & visualisasi 3D...'):
                orig_img = Image.open(uploaded_file).convert('RGB')
                
                # 1. Proses Gambar
                result_data = self.model.process_image_cached(orig_img, bits)
                
                # 2. Hitung Metrik
                mse, psnr = self.model.calculate_mse_psnr(
                    result_data['original_array'], 
                    result_data['reconstructed_array']
                )
                
                # 3. Ekstrak Data untuk Visualisasi
                palette = self.model.extract_palette(result_data['reconstructed_array'], 8)
                
                # --- DATA 3D BARU ---
                # Mengambil sampel data untuk plot 3D (agar ringan, ambil 500 sampel saja)
                df_3d_orig = self.model.get_3d_plot_data(result_data['original_array'], num_samples=500)
                df_3d_res = self.model.get_3d_plot_data(result_data['reconstructed_array'], num_samples=500)
                # --------------------
            
            # 4. Render UI
            self.view.render_dashboard(bits, mse, psnr)
            # Pass data 3D ke View
            self.view.render_tabs(orig_img, result_data, bits, palette, df_3d_orig, df_3d_res)
        
        else:
            self.view.render_empty_state()
            
        self.view.render_footer()