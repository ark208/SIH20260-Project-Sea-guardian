"""
Project DRISHTI - Module 1: SAR Vision Preprocessing
Radiometric Calibration, Speckle Filtering (Refined Lee), and Synthetic SAR Swath Generator.
"""
import numpy as np
from typing import Tuple, Dict, Any

class SARPreprocessor:
    def __init__(self, filter_size: int = 7):
        self.filter_size = filter_size

    def calibrate_to_sigma0_db(self, raw_dn: np.ndarray, calibration_factor: float = 0.01) -> np.ndarray:
        """Converts raw Digital Numbers (DN) to backscatter coefficient sigma0 in dB."""
        intensity = (raw_dn.astype(np.float32) ** 2) * calibration_factor + 1e-6
        return 10.0 * np.log10(intensity)

    def refined_lee_filter(self, img: np.ndarray) -> np.ndarray:
        """Applies Lee despeckle filter to suppress multiplicative speckle noise while preserving slick edges."""
        k = self.filter_size
        pad = k // 2
        padded = np.pad(img, pad, mode='reflect')
        filtered = np.zeros_like(img, dtype=np.float32)
        h, w = img.shape
        
        for i in range(h):
            for j in range(w):
                window = padded[i:i+k, j:j+k]
                local_mean = float(np.mean(window))
                local_var = float(np.var(window))
                noise_var = (local_mean * 0.28) ** 2
                weight = max(0.0, 1.0 - (noise_var / (local_var + 1e-6))) if local_var > 0 else 0.0
                filtered[i, j] = local_mean + weight * (img[i, j] - local_mean)
                
        return filtered

    def generate_synthetic_scene(
        self,
        height: int = 256,
        width: int = 256,
        slick_center: Tuple[int, int] = (130, 125),
        slick_radius: int = 28,
        wind_speed_ms: float = 5.5,
        center_lat: float = 18.92,
        center_lon: float = 72.45
    ) -> Dict[str, Any]:
        """Generates realistic synthetic Sentinel-1 SAR scene with backscatter damping, noise and georeferencing."""
        base_db = -18.0 + 3.0 * np.log10(max(wind_speed_ms, 1.0))
        y, x = np.ogrid[:height, :width]
        
        # Elongated slick shape
        angle = np.radians(35.0)
        x_rot = (x - slick_center[0]) * np.cos(angle) + (y - slick_center[1]) * np.sin(angle)
        y_rot = -(x - slick_center[0]) * np.sin(angle) + (y - slick_center[1]) * np.cos(angle)
        dist_from_center = np.sqrt((x_rot / 1.8)**2 + (y_rot * 1.2)**2)
        
        slick_mask = dist_from_center < slick_radius
        
        db_map = np.full((height, width), base_db, dtype=np.float32)
        # Capillary wave damping: -9.5 dB drop
        db_map[slick_mask] -= 9.5
        
        # Gamma speckle noise
        speckle = np.random.gamma(shape=4.0, scale=0.25, size=(height, width))
        noisy_linear = (10.0 ** (db_map / 10.0)) * speckle
        noisy_db = 10.0 * np.log10(np.maximum(noisy_linear, 1e-5))
        
        lat_step, lon_step = 0.0001, 0.0001 # ~10m resolution
        
        return {
            "sar_image_db": noisy_db,
            "raw_linear": noisy_linear,
            "ground_truth_mask": slick_mask.astype(np.uint8),
            "wind_speed_ms": wind_speed_ms,
            "bounds": {
                "min_lat": center_lat - (height / 2) * lat_step,
                "max_lat": center_lat + (height / 2) * lat_step,
                "min_lon": center_lon - (width / 2) * lon_step,
                "max_lon": center_lon + (width / 2) * lon_step,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "pixel_size_m": 10.0,
                "height": height,
                "width": width
            }
        }
