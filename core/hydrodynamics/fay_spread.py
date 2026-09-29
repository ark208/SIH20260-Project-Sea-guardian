"""
Project DRISHTI - Module 2: Fay's 3-Phase Spreading Model
Models Phase 1 (Gravity-Inertia), Phase 2 (Gravity-Viscous), and Phase 3 (Surface Tension-Viscous).
Inverts observed slick area to estimate discharge age T_spill.
"""
import numpy as np
from typing import Dict, Any

class FaySpreadingModel:
    def __init__(
        self,
        rho_oil: float = 880.0,       # kg/m^3 (Medium Crude)
        rho_water: float = 1025.0,    # kg/m^3 (Seawater)
        nu_water: float = 1.14e-6,    # m^2/s kinematic viscosity
        sigma_net: float = 0.025,     # N/m net spreading coefficient
        g: float = 9.81               # m/s^2
    ):
        self.rho_oil = rho_oil
        self.rho_water = rho_water
        self.delta = 1.0 - (rho_oil / rho_water)
        self.nu_water = nu_water
        self.sigma_net = sigma_net
        self.g = g
        
        # Fay empirical constants
        self.k1 = 1.14  # Gravity-Inertia
        self.k2 = 1.45  # Gravity-Viscous
        self.k3 = 2.30  # Surface Tension-Viscous

    def radius_at_time(self, t_sec: float, volume_m3: float = 25.0) -> Dict[str, float]:
        """Calculates slick radius across 3 Fay phases at time t_sec."""
        t = max(t_sec, 1.0)
        
        # Phase 1: Gravity - Inertia
        r1 = self.k1 * ((self.delta * self.g * volume_m3 * (t**2)) ** 0.25)
        
        # Phase 2: Gravity - Viscous
        r2 = self.k2 * (((self.delta * self.g * (volume_m3**2) * (t**1.5)) / (self.nu_water**0.5)) ** (1.0/6.0))
        
        # Phase 3: Surface Tension - Viscous
        r3 = self.k3 * (((self.sigma_net**2 * (t**3)) / ((self.rho_water**2) * self.nu_water)) ** 0.25)
        
        # Transition times
        t_trans_1_2 = ((self.delta * self.g * volume_m3) / (self.nu_water**2)) ** (1.0/3.0) * 0.05
        t_trans_2_3 = ((self.rho_water**2 * self.delta * self.g * (volume_m3**2)) / (self.sigma_net**2)) ** (1.0/3.0) * 0.1
        
        if t < t_trans_1_2:
            phase = 1
            r_active = r1
        elif t < t_trans_2_3:
            phase = 2
            r_active = r2
        else:
            phase = 3
            r_active = r3
            
        return {
            "r1": float(r1),
            "r2": float(r2),
            "r3": float(r3),
            "r_active": float(r_active),
            "area_m2": float(np.pi * (r_active**2)),
            "phase": phase,
            "t_sec": float(t)
        }

    def estimate_discharge_age(self, observed_area_m2: float, estimated_volume_m3: float = 25.0) -> Dict[str, Any]:
        """
        Inverts observed slick area to estimate discharge age T_spill (hours)
        and associated uncertainty sigma_T.
        """
        observed_r = np.sqrt(observed_area_m2 / np.pi)
        
        # Binary search for t_sec
        low, high = 60.0, 72.0 * 3600.0  # 1 min to 72 hours
        for _ in range(50):
            mid = (low + high) / 2.0
            pred = self.radius_at_time(mid, estimated_volume_m3)
            if pred["r_active"] < observed_r:
                low = mid
            else:
                high = mid
                
        t_opt_sec = (low + high) / 2.0
        t_opt_hours = t_opt_sec / 3600.0
        
        # Uncertainty envelope +/- 20%
        sigma_hours = t_opt_hours * 0.20
        
        return {
            "spill_age_hours": round(float(t_opt_hours), 2),
            "spill_age_seconds": round(float(t_opt_sec), 1),
            "uncertainty_hours": round(float(sigma_hours), 2),
            "min_age_hours": max(0.5, round(float(t_opt_hours - sigma_hours), 2)),
            "max_age_hours": round(float(t_opt_hours + sigma_hours), 2),
            "estimated_volume_m3": estimated_volume_m3,
            "observed_area_m2": observed_area_m2
        }
