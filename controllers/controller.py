import streamlit as st
from PIL import Image

# Import Modul MVC
from models.model import QuantizationModel
from views.view import AppView

class AppController:
    """
    CONTROLLER: Mengatur alur komunikasi Model <-> View
    """
    def __init__(self):
        self.model = QuantizationModel()
        self.view = AppView()

    def run(self):
        self.view.setup_page()
        uploaded_file, bits = self.view.render_sidebar()
        self.view.render_header()
        
        if uploaded_file is not None:
            # Spinner ini akan muncul sebentar saja karena ada Cache
            with st.spinner('Memproses citra...'):
                orig_img = Image.open(uploaded_file).convert('RGB')
                
                # PANGGIL MODEL (Fungsi Static yang di-Cache)
                result_data = self.model.process_image_cached(orig_img, bits)
                
                # Hitung Metrik & Palette
                mse, psnr = self.model.calculate_mse_psnr(
                    result_data['original_array'], 
                    result_data['reconstructed_array']
                )
                palette = self.model.extract_palette(result_data['reconstructed_array'], 8)
            
            # PANGGIL VIEW
            self.view.render_dashboard(bits, mse, psnr)
            self.view.render_tabs(orig_img, result_data, bits, palette)
        
        else:
            self.view.render_empty_state()
            
        self.view.render_footer()