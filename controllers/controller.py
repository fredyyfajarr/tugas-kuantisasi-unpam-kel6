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
            with st.spinner('Memproses citra...'):
                orig_img = Image.open(uploaded_file).convert('RGB')
                
                # --- [PENTING] RESIZE AGAR TIDAK HANG ---
                # Kecilkan gambar jika terlalu besar (>1000px)
                # Ini mencegah blocking thread saat processing
                orig_img.thumbnail((1000, 1000)) 
                # ----------------------------------------
                
                # 1. Proses Gambar (Cached)
                result_data = self.model.process_image_cached(orig_img, bits)
                
                # 2. Hitung Metrik & Palette
                mse, psnr = self.model.calculate_mse_psnr(
                    result_data['original_array'], 
                    result_data['reconstructed_array']
                )
                palette = self.model.extract_palette(result_data['reconstructed_array'], 8)
                
                # 3. Data Visualisasi 3D (Sampled)
                df_3d_orig = self.model.get_3d_plot_data(result_data['original_array'], num_samples=500)
                df_3d_res = self.model.get_3d_plot_data(result_data['reconstructed_array'], num_samples=500)
            
            # 4. Tampilkan View
            self.view.render_dashboard(bits, mse, psnr)
            self.view.render_tabs(orig_img, result_data, bits, palette, df_3d_orig, df_3d_res)
        
        else:
            self.view.render_empty_state()
            
        self.view.render_footer()