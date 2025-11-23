import streamlit as st
from PIL import Image

# Import dari folder sebelah (Root level import)
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
            with st.spinner('Memproses citra...'):
                orig_img = Image.open(uploaded_file).convert('RGB')
                
                # Minta Model bekerja
                result_data = self.model.process_image(orig_img, bits)
                mse, psnr = self.model.calculate_mse_psnr(result_data['original_array'], result_data['reconstructed_array'])
                palette = self.model.extract_palette(result_data['reconstructed_array'], 8)
            
            # Minta View tampilkan
            self.view.render_dashboard(bits, mse, psnr)
            self.view.render_tabs(orig_img, result_data, bits, palette)
        else:
            self.view.render_empty_state()
            
        self.view.render_footer()