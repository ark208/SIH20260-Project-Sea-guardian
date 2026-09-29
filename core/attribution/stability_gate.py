"""
Project DRISHTI - Module 5: Stability Gate, Sensitivity Check & NTRO Relay
Performs sensitivity verification (Stokes toggle, wind +/- 20%) and compiles forensic dossier.
"""
from typing import Dict, Any, List
import datetime

class StabilityGateAndRelay:
    def evaluate_stability(
        self,
        baseline_suspects: List[Dict[str, Any]],
        no_stokes_suspects: List[Dict[str, Any]],
        wind_plus_suspects: List[Dict[str, Any]],
        wind_minus_suspects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Tests whether the top-ranked suspect remains stable under oceanographic perturbations.
        """
        top_baseline = baseline_suspects[0]["vessel_id"]
        top_no_stokes = no_stokes_suspects[0]["vessel_id"]
        top_w_plus = wind_plus_suspects[0]["vessel_id"]
        top_w_minus = wind_minus_suspects[0]["vessel_id"]
        
        is_consistent = (
            top_baseline == top_no_stokes and
            top_baseline == top_w_plus and
            top_baseline == top_w_minus
        )
        
        # Calculate rank 1 probability variance across 4 perturbations
        probs = [
            baseline_suspects[0]["normalized_culprit_probability"],
            no_stokes_suspects[0]["normalized_culprit_probability"],
            wind_plus_suspects[0]["normalized_culprit_probability"],
            wind_minus_suspects[0]["normalized_culprit_probability"]
        ]
        prob_range = max(probs) - min(probs)
        
        confidence_level = "HIGH CONFIDENCE (DISPATCH READY)" if (is_consistent and prob_range < 0.15) else "MODERATE / REQUIRES ANALYST REVIEW"
        
        return {
            "is_stable": bool(is_consistent),
            "confidence_level": confidence_level,
            "top_suspect_id": top_baseline,
            "probability_range": round(float(prob_range), 4),
            "perturbation_scores": {
                "baseline": baseline_suspects[0]["normalized_culprit_probability"],
                "no_stokes_toggle": no_stokes_suspects[0]["normalized_culprit_probability"],
                "wind_plus_20": wind_plus_suspects[0]["normalized_culprit_probability"],
                "wind_minus_20": wind_minus_suspects[0]["normalized_culprit_probability"]
            }
        }

    def generate_ntRO_incident_dossier(
        self,
        incident_id: str,
        slick_detection: Dict[str, Any],
        origin_envelope: Dict[str, Any],
        ranked_suspects: List[Dict[str, Any]],
        stability_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compiles standard NTRO / Indian Navy / Coast Guard Forensic Incident Dossier."""
        prime_suspect = ranked_suspects[0]
        
        return {
            "incident_id": incident_id,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "classification": "RESTRICTED // MARITIME SURVEILLANCE INTELLIGENCE",
            "issuing_authority": "PROJECT DRISHTI - HIGH ASSURANCE MARITIME ATTRIBUTION ENGINE",
            "oil_slick_summary": {
                "slick_id": slick_detection["slick_id"],
                "area_km2": slick_detection["area_km2"],
                "detection_centroid": [slick_detection["centroid_lat"], slick_detection["centroid_lon"]],
                "estimated_discharge_age_hours": origin_envelope["discharge_time_hours_ago"],
                "inferred_origin_centroid": [origin_envelope["origin_lat"], origin_envelope["origin_lon"]],
                "origin_search_radius_m": origin_envelope["origin_radius_m"]
            },
            "stability_audit": stability_result,
            "prime_suspect_dossier": {
                "rank": 1,
                "vessel_name": prime_suspect["name"],
                "mmsi": prime_suspect["mmsi"],
                "imo": prime_suspect["imo"],
                "vessel_type": prime_suspect["vessel_type"],
                "flag_state": prime_suspect["flag_state"],
                "culprit_attribution_probability": prime_suspect["normalized_culprit_probability"],
                "culprit_percentage": f"{prime_suspect['culprit_percentage']}%",
                "breakdown": {
                    "spatial_overlap_score": prime_suspect["s_spatial"],
                    "distance_to_origin_km": prime_suspect["spatial_details"]["distance_to_origin_km"],
                    "ais_dark_blackout_score": prime_suspect["s_dark"],
                    "dark_duration_hours": prime_suspect["dark_details"]["dark_duration_hours"],
                    "behavioral_anomaly_score": prime_suspect["s_behavior"],
                    "speed_drop_knots": prime_suspect["behavior_details"]["speed_drop_knots"],
                    "vessel_risk_score": prime_suspect["s_risk"]
                }
            },
            "all_ranked_suspects": [
                {
                    "rank": s["rank"],
                    "vessel_name": s["name"],
                    "mmsi": s["mmsi"],
                    "type": s["vessel_type"],
                    "score": s["normalized_culprit_probability"]
                }
                for s in ranked_suspects
            ]
        }
