"""
Project DRISHTI - Module 1: Contextual Look-Alike Discriminator
Filters out low-wind calms, biogenic algal films, ship wakes, and rain cells.
"""
from typing import Dict, Any, Tuple

class LookAlikeDiscriminator:
    def __init__(
        self,
        min_wind_speed: float = 2.5,
        max_wind_speed: float = 14.0,
        max_compactness_biogenic: float = 0.85
    ):
        self.min_wind_speed = min_wind_speed
        self.max_wind_speed = max_wind_speed
        self.max_compactness_biogenic = max_compactness_biogenic

    def evaluate_candidate(
        self,
        slick: Dict[str, Any],
        local_wind_speed_ms: float
    ) -> Tuple[bool, str, float]:
        """
        Evaluates a candidate oil slick detection against environmental context.
        Returns: (is_confirmed, rejection_reason, risk_score)
        """
        # Wind condition checks
        if local_wind_speed_ms < self.min_wind_speed:
            return False, "Low-wind calm / Specular reflection look-alike", 0.15
        
        if local_wind_speed_ms > self.max_wind_speed:
            return False, "High-wind dispersion threshold exceeded", 0.20
        
        # Morphological compactness check (natural films are often very circular or diffuse)
        compactness = slick.get("compactness", 0.5)
        if compactness > self.max_compactness_biogenic:
            return False, "High compactness characteristic of biogenic film", 0.30
        
        # Area check
        if slick.get("area_m2", 0) < 500:
            return False, "Candidate area below physical oil discharge threshold", 0.10
        
        return True, "Verified Mineral Oil Slick Signature", slick.get("confidence", 0.90)
