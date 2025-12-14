import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

class QuantizationModel:
    """
    MODEL: Logika Bisnis & Algoritma
    """

    def calculate_mse_psnr(self, original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if mse == 0: return 0, 100
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return mse, psnr

    def extract_palette(self, image_array, num=10):
        pixels = image_array.reshape(-1, 3)
        unique = np.unique(pixels, axis=0)
        if len(unique) > num:
            idx = np.linspace(0, len(unique)-1, num, dtype=int)
            return unique[idx]
        return unique

    def get_decode_stats(self, raw_labels):
        """Statistik jumlah pixel per kelompok"""
        unique, counts = np.unique(np.array(raw_labels).flatten(), return_counts=True)
        df = pd.DataFrame({
            'Label': unique,
            'Total Pixel': counts,
            'Persentase': (counts / counts.sum() * 100).round(2).astype(str) + '%'
        })
        return df

    def get_codebook(self, original_channel, raw_labels):
        """Kamus Warna: Nilai rata-rata asli untuk setiap label"""
        df = pd.DataFrame({'val': original_channel.flatten(), 'label': raw_labels.flatten()})
        codebook = df.groupby('label')['val'].mean().round(1).reset_index()
        codebook.columns = ['Label Kelompok', 'Nilai Rata-Rata Asli']
        return codebook

    @staticmethod
    @st.cache_data(show_spinner=False)
    def process_image_cached(image, bits):
        img_array = np.array(image)
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        def quantize_channel(flat_channel, b_bits):
            num_levels = 2 ** b_bits
            try:
                return pd.qcut(flat_channel, q=num_levels, labels=False, duplicates='drop')
            except ValueError:
                return pd.qcut(flat_channel, q=num_levels, labels=False, duplicates='drop')

        r_labels = quantize_channel(r.flatten(), bits)
        g_labels = quantize_channel(g.flatten(), bits)
        b_labels = quantize_channel(b.flatten(), bits)
        
        # Reshape ke 2D agar bisa divisualisasikan di View
        r_new = r_labels.reshape(r.shape)
        g_new = g_labels.reshape(g.shape)
        b_new = b_labels.reshape(b.shape)
        
        factor = 255 / ((2**bits) - 1) if bits > 0 else 1
        
        r_disp = (r_new * factor).astype(np.uint8)
        g_disp = (g_new * factor).astype(np.uint8)
        b_disp = (b_new * factor).astype(np.uint8)
        
        img_reconstructed = np.stack((r_disp, g_disp, b_disp), axis=2)
        
        return {
            'reconstructed_img': Image.fromarray(img_reconstructed),
            'reconstructed_array': img_reconstructed,
            'original_array': img_array,
            'channels_display': (r_disp, g_disp, b_disp),
            'raw_labels_r': r_new  # Penting: Mengirim data 2D
        }