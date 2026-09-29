"""
Project DRISHTI - Module 4: AIS Dark Scorer (30% Weight)
Evaluates intentional transponder blackout near the discharge zone:
S_dark = I_dark * [1 - exp(-Delta_t_dark / tau_0)] * P_STGNN(in_zone)
"""
import numpy as np
from typing import Dict, Any

class AISDarkScorer:
    def __init__(self, tau_0_hours: float = 2.0):
        self.tau_0 = tau_0_hours

    def compute_dark_score(
        self,
        vessel: Dict[str, Any],
        st_gnn_result: Dict[str, Any],
        discharge_time_hours_ago: float,
        spatial_distance_km: float
    ) -> Dict[str, Any]:
        """Calculates AIS blackout anomaly score."""
        has_dark = vessel.get("has_dark_gap", False)
        dark_duration_h = vessel.get("dark_duration_hours", 0.0)
        
        if not has_dark or dark_duration_h <= 0.2:
            return {
                "dark_score": 0.0,
                "is_dark_during_discharge": False,
                "dark_duration_hours": 0.0,
                "blackout_penalty": 0.0
            }
            
        dark_min_h = min(vessel["dark_interval"])
        dark_max_h = max(vessel["dark_interval"])
        
        # Check if discharge happened while dark
        is_dark_at_discharge = (dark_min_h <= discharge_time_hours_ago <= dark_max_h)
        
        # Time duration penalty [0, 1]
        duration_factor = 1.0 - float(np.exp(-dark_duration_h / self.tau_0))
        
        # Proximity relevance (if dark 50km away, lower relevance)
        proximity_factor = float(np.exp(-0.5 * (spatial_distance_km / 15.0)**2))
        
        # Dark score
        base_weight = 1.0 if is_dark_at_discharge else 0.4
        score = base_weight * duration_factor * proximity_factor
        
        return {
            "dark_score": round(float(score), 4),
            "is_dark_during_discharge": bool(is_dark_at_discharge),
            "dark_duration_hours": round(float(dark_duration_h), 2),
            "duration_factor": round(float(duration_factor), 3),
            "proximity_factor": round(float(proximity_factor), 3)
        }
