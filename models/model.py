import numpy as np
import pandas as pd
from PIL import Image

class QuantizationModel:
    def calculate_mse_psnr(self, original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if mse == 0: return 0, 100
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return mse, psnr

    def _quantize_channel(self, channel_array, bits):
        """Algoritma Inti: Equal Frequency Binning (Non-Uniform)"""
        flat = channel_array.flatten()
        num_levels = 2 ** bits
        try:
            quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')
        except ValueError:
            quantized_flat = pd.qcut(flat, q=num_levels, labels=False, duplicates='drop')
        return quantized_flat

    def process_image(self, image, bits):
        """Logic pemrosesan citra RGB per kanal"""
        img_array = np.array(image)
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        # Proses per kanal
        r_labels = self._quantize_channel(r, bits)
        g_labels = self._quantize_channel(g, bits)
        b_labels = self._quantize_channel(b, bits)
        
        # Reconstruct
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
            'raw_labels_r': r_labels
        }

    def extract_palette(self, image_array, num=10):
        pixels = image_array.reshape(-1, 3)
        unique = np.unique(pixels, axis=0)
        if len(unique) > num:
            idx = np.linspace(0, len(unique)-1, num, dtype=int)
            return unique[idx]
        return unique