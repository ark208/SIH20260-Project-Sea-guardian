"""
Project DRISHTI - Module 2: Ocean Environment Service
Simulates / integrates Copernicus CMEMS ocean surface currents and ERA5 10m wind fields.
"""
import numpy as np
from typing import Dict, Any

class OceanEnvironmentService:
    def __init__(
        self,
        windage_coeff: float = 0.032,     # 3.2% windage
        coriolis_deflection_deg: float = 15.0 # Clockwise deflection in Northern Hemisphere
    ):
        self.windage_coeff = windage_coeff
        self.coriolis_deflection_deg = coriolis_deflection_deg

    def get_ocean_state(self, lat: float, lon: float, timestamp_hours_ago: float = 0.0) -> Dict[str, Any]:
        """
        Returns Eulerian current vector (u_curr, v_curr) in m/s,
        10m wind vector (u_wind, v_wind) in m/s, and wave parameters.
        Includes gentle tidal oscillation and spatial gradient.
        """
        # Base CMEMS Eulerian current: ~0.25 m/s toward North-East
        phase = (timestamp_hours_ago % 12.42) * (2 * np.pi / 12.42) # Semidiurnal tide
        u_curr = 0.18 + 0.06 * np.cos(phase) + 0.02 * (lat - 18.9)
        v_curr = 0.22 + 0.05 * np.sin(phase) - 0.02 * (lon - 72.4)
        
        # ERA5 10m Wind field: ~5.8 m/s South-Westerly
        u_wind = 4.2 + 0.5 * np.sin(phase * 0.5)
        v_wind = 4.0 + 0.4 * np.cos(phase * 0.5)
        wind_speed = float(np.hypot(u_wind, v_wind))
        
        # Wave parameters
        hs = 1.6 + 0.1 * wind_speed
        tp = 7.5 + 0.2 * wind_speed
        wave_dir = float(np.degrees(np.arctan2(u_wind, v_wind)) % 360.0)
        
        return {
            "current": {"u_ms": float(u_curr), "v_ms": float(v_curr)},
            "wind": {"u_ms": float(u_wind), "v_ms": float(v_wind), "speed_ms": wind_speed},
            "wave": {"hs_m": float(hs), "tp_sec": float(tp), "dir_deg": wave_dir}
        }

    def compute_total_drift_vector(
        self,
        lat: float,
        lon: float,
        timestamp_hours_ago: float,
        stokes_calculator,
        enable_stokes: bool = True,
        wind_perturbation_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculates total drift vector U_drift = u_current + alpha * R(theta) * u_wind + u_Stokes.
        """
        env = self.get_ocean_state(lat, lon, timestamp_hours_ago)
        u_curr = env["current"]["u_ms"]
        v_curr = env["current"]["v_ms"]
        
        # Windage with Coriolis deflection
        u_w = env["wind"]["u_ms"] * wind_perturbation_factor
        v_w = env["wind"]["v_ms"] * wind_perturbation_factor
        theta_rad = np.radians(self.coriolis_deflection_deg)
        
        # Rotate wind vector
        u_w_rot = u_w * np.cos(theta_rad) - v_w * np.sin(theta_rad)
        v_w_rot = u_w * np.sin(theta_rad) + v_w * np.cos(theta_rad)
        
        u_windage = self.windage_coeff * u_w_rot
        v_windage = self.windage_coeff * v_w_rot
        
        # Stokes drift
        if enable_stokes:
            stokes = stokes_calculator.calculate_stokes_drift(
                hs_m=env["wave"]["hs_m"],
                tp_sec=env["wave"]["tp_sec"],
                wave_dir_deg=env["wave"]["dir_deg"]
            )
            u_stokes = stokes["u_stokes_east_ms"]
            v_stokes = stokes["u_stokes_north_ms"]
        else:
            u_stokes = 0.0
            v_stokes = 0.0
            
        u_total = u_curr + u_windage + u_stokes
        v_total = v_curr + v_windage + v_stokes
        
        return {
            "u_total_ms": float(u_total),
            "v_total_ms": float(v_total),
            "u_curr": float(u_curr),
            "v_curr": float(v_curr),
            "u_windage": float(u_windage),
            "v_windage": float(v_windage),
            "u_stokes": float(u_stokes),
            "v_stokes": float(v_stokes)
        }
