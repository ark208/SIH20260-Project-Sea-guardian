"""
Project DRISHTI - Module 2: Wave-Induced Stokes Drift
Calculates Stokes surface drift velocity vector from WaveWatch III wave spectra (Hs, Tp, wave direction).
"""
import numpy as np
from typing import Tuple, Dict, Any

class StokesDriftCalculator:
    def __init__(self, g: float = 9.81):
        self.g = g

    def calculate_stokes_drift(
        self,
        hs_m: float = 1.8,          # Significant wave height (meters)
        tp_sec: float = 8.5,        # Peak wave period (seconds)
        wave_dir_deg: float = 240.0 # Wave propagation direction (degrees from North)
    ) -> Dict[str, Any]:
        """
        Computes wave Stokes drift: u_Stokes = (1/16) * Hs^2 * omega_p * k_p
        Where omega_p = 2*pi/Tp and deep-water wavenumber k_p = omega_p^2 / g
        """
        omega_p = (2.0 * np.pi) / max(tp_sec, 1.0)
        k_p = (omega_p ** 2) / self.g
        
        # Surface Stokes drift speed (m/s)
        u_stokes_mag = (1.0 / 16.0) * (hs_m ** 2) * omega_p * k_p
        
        # Convert nautical direction (direction waves are coming from / traveling toward)
        # Mathematical angle in radians (East = 0, North = 90)
        rad = np.radians(90.0 - wave_dir_deg)
        u_stokes_east = u_stokes_mag * np.cos(rad)
        u_stokes_north = u_stokes_mag * np.sin(rad)
        
        return {
            "u_stokes_mag_ms": float(u_stokes_mag),
            "u_stokes_east_ms": float(u_stokes_east),
            "u_stokes_north_ms": float(u_stokes_north),
            "wave_dir_deg": wave_dir_deg,
            "hs_m": hs_m,
            "tp_sec": tp_sec
        }
