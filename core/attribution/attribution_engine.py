"""
Project DRISHTI - Module 4: Multi-Factor Attribution Engine
Integrates 4 scoring dimensions with controlled dynamic weights:
S_culprit = gamma_evidence * [0.35 * S_spatial + 0.30 * S_dark + 0.20 * S_behavior + 0.15 * S_risk]
And performs dynamic normalization across candidate fleet.
"""
import numpy as np
from typing import List, Dict, Any

class MultiFactorAttributionEngine:
    def __init__(
        self,
        weight_spatial: float = 0.35,
        weight_dark: float = 0.30,
        weight_behavior: float = 0.20,
        weight_risk: float = 0.15
    ):
        self.w_spatial = weight_spatial
        self.w_dark = weight_dark
        self.w_behavior = weight_behavior
        self.w_risk = weight_risk

    def evaluate_fleet(
        self,
        vessels: List[Dict[str, Any]],
        origin_envelope: Dict[str, Any],
        st_gnn_predictions: Dict[str, Any],
        spatial_scorer,
        dark_scorer,
        behavior_risk_scorer
    ) -> List[Dict[str, Any]]:
        """Evaluates and ranks all candidate vessels with full explainable score breakdowns."""
        discharge_h = origin_envelope["discharge_time_hours_ago"]
        
        raw_evaluations = []
        matching_count = 0
        
        for vessel in vessels:
            vid = vessel["vessel_id"]
            gnn_res = st_gnn_predictions.get(vid, {})
            pos_at_discharge = gnn_res.get("pos_at_discharge", {"lat": 0, "lon": 0})
            
            # 1. Spatial Match Score (35%)
            spatial_eval = spatial_scorer.compute_spatial_score(pos_at_discharge, origin_envelope)
            s_spatial = spatial_eval["spatial_score"]
            dist_km = spatial_eval["distance_to_origin_km"]
            
            if s_spatial > 0.3:
                matching_count += 1
                
            # 2. AIS Dark Score (30%)
            dark_eval = dark_scorer.compute_dark_score(vessel, gnn_res, discharge_h, dist_km)
            s_dark = dark_eval["dark_score"]
            
            # 3. Behavioral Anomaly Score (20%)
            beh_eval = behavior_risk_scorer.compute_behavior_score(vessel)
            s_behavior = beh_eval["behavior_score"]
            
            # 4. Vessel Risk Profile Score (15%)
            risk_eval = behavior_risk_scorer.compute_risk_score(vessel)
            s_risk = risk_eval["risk_score"]
            
            # Raw composite
            raw_composite = (
                self.w_spatial * s_spatial +
                self.w_dark * s_dark +
                self.w_behavior * s_behavior +
                self.w_risk * s_risk
            )
            
            raw_evaluations.append({
                "vessel_id": vid,
                "name": vessel["name"],
                "mmsi": vessel["mmsi"],
                "imo": vessel["imo"],
                "vessel_type": vessel["vessel_type"],
                "flag_state": vessel["flag_state"],
                "dwt_tonnage": vessel["dwt_tonnage"],
                "has_dark_gap": vessel["has_dark_gap"],
                "raw_composite": raw_composite,
                "s_spatial": s_spatial,
                "s_dark": s_dark,
                "s_behavior": s_behavior,
                "s_risk": s_risk,
                "spatial_details": spatial_eval,
                "dark_details": dark_eval,
                "behavior_details": beh_eval,
                "risk_details": risk_eval,
                "pos_at_discharge": pos_at_discharge
            })
            
        # Evidence Quality Dampening (crowded lane penalty)
        # gamma = 1.0 / sqrt(max(1, matching_count))
        gamma_evidence = 1.0 / float(np.sqrt(max(1, matching_count)))
        
        # Apply gamma and normalize
        for item in raw_evaluations:
            item["dampened_composite"] = item["raw_composite"] * gamma_evidence
            
        total_dampened = sum(item["dampened_composite"] for item in raw_evaluations) + 1e-6
        for item in raw_evaluations:
            item["normalized_culprit_probability"] = round(float(item["dampened_composite"] / total_dampened), 4)
            item["culprit_percentage"] = round(item["normalized_culprit_probability"] * 100.0, 1)
            
        # Rank suspects in descending order
        ranked = sorted(raw_evaluations, key=lambda x: x["normalized_culprit_probability"], reverse=True)
        for idx, suspect in enumerate(ranked):
            suspect["rank"] = idx + 1
            
        return ranked
