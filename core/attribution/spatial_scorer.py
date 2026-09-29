"""
Project DRISHTI - Module 4: Spatial Overlap Scorer (35% Weight)
Computes spatial match score between vessel trajectory/uncertainty ellipse
and backward rewind origin probability envelope P_origin(x, y, t).
"""
import numpy as np
from typing import Dict, Any

class SpatialOverlapScorer:
    def __init__(self, earth_radius_m: float = 6371000.0):
        self.r_earth = earth_radius_m

    def compute_distance_meters(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance between two coordinates in meters."""
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return float(self.r_earth * c)

    def compute_spatial_score(
        self,
        vessel_pos_at_discharge: Dict[str, Any],
        origin_envelope: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates spatial overlap integral between Gaussian origin distribution and vessel position.
        S_spatial = exp(-0.5 * (d / sigma_total)^2)
        """
        v_lat = vessel_pos_at_discharge["lat"]
        v_lon = vessel_pos_at_discharge["lon"]
        v_sigma = vessel_pos_at_discharge.get("sigma_m", 300.0)
        
        o_lat = origin_envelope["origin_lat"]
        o_lon = origin_envelope["origin_lon"]
        o_radius = origin_envelope.get("origin_radius_m", 1500.0)
        
        dist_m = self.compute_distance_meters(v_lat, v_lon, o_lat, o_lon)
        
        # Combined standard deviation
        sigma_combined = np.sqrt((o_radius / 2.0)**2 + v_sigma**2)
        
        # Gaussian overlap match score (0.0 to 1.0)
        raw_score = float(np.exp(-0.5 * (dist_m / (sigma_combined + 1e-6))**2))
        
        return {
            "spatial_score": round(float(raw_score), 4),
            "distance_to_origin_m": round(float(dist_m), 1),
            "distance_to_origin_km": round(float(dist_m / 1000.0), 2),
            "sigma_combined_m": round(float(sigma_combined), 1)
        }
