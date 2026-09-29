"""
Project DRISHTI - Module 4: Behavior & Vessel Risk Profiler (20% & 15% Weights)
Evaluates kinematic anomalies (speed drops, loitering) and prior static vessel risk (DWT, flag state, age, PSC).
"""
import numpy as np
from typing import Dict, Any

class BehaviorAndRiskScorer:
    def compute_behavior_score(self, vessel: Dict[str, Any]) -> Dict[str, Any]:
        track = vessel.get("track", [])
        speeds = [p["speed_knots"] for p in track]
        if not speeds:
            return {"behavior_score": 0.0, "speed_drop_knots": 0.0, "is_loitering": False}
        min_speed = float(np.min(speeds))
        max_speed = float(np.max(speeds))
        speed_delta = max_speed - min_speed
        speed_drop_score = min(1.0, speed_delta / 10.0)
        is_loitering = (min_speed < 6.0 and max_speed > 11.0)
        loiter_score = 0.85 if is_loitering else 0.10
        behavior_score = 0.6 * speed_drop_score + 0.4 * loiter_score
        return {
            "behavior_score": round(float(behavior_score), 4),
            "speed_drop_knots": round(float(speed_delta), 1),
            "min_speed_knots": round(float(min_speed), 1),
            "max_speed_knots": round(float(max_speed), 1),
            "is_loitering": bool(is_loitering)
        }

    def compute_risk_score(self, vessel: Dict[str, Any]) -> Dict[str, Any]:
        vtype = vessel.get("vessel_type", "").lower()
        if any(w in vtype for w in ["crude", "vlcc", "suezmax", "aframax"]):
            type_score = 0.97
        elif any(w in vtype for w in ["chemical", "oil", "tanker", "product"]):
            type_score = 0.85
        elif any(w in vtype for w in ["bulk", "cargo", "ore"]):
            type_score = 0.52
        elif any(w in vtype for w in ["container", "reefer", "roro", "passenger", "lpg"]):
            type_score = 0.30
        else:
            type_score = 0.25
        flag_score = float(vessel.get("flag_risk_index", 0.3))
        age = float(vessel.get("vessel_age_years", vessel.get("vessel_age", 10)))
        age_score = min(1.0, age / 25.0)
        psc_defs = float(vessel.get("psc_deficiencies_past_3y", vessel.get("past_psc_deficiencies", 0)))
        psc_score = min(1.0, psc_defs / 5.0)
        risk_score = 0.40 * type_score + 0.30 * flag_score + 0.15 * age_score + 0.15 * psc_score
        return {
            "risk_score": round(float(risk_score), 4),
            "vessel_type_factor": type_score,
            "flag_risk_factor": flag_score,
            "age_factor": round(float(age_score), 2),
            "psc_factor": round(float(psc_score), 2)
        }
