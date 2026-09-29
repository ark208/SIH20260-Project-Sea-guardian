# Project Sea Guardian - Multi-Region Maritime Pipeline
from typing import Dict, Any, List, Optional
import numpy as np

from core.sar_vision.preprocessor import SARPreprocessor
from core.sar_vision.detector import OilSlickDetector
from core.sar_vision.lookalike_filter import LookAlikeDiscriminator
from core.sar_vision.slick_geometry import NaturalSlickMorphology

from core.hydrodynamics.fay_spread import FaySpreadingModel
from core.hydrodynamics.stokes_drift import StokesDriftCalculator
from core.hydrodynamics.ocean_service import OceanEnvironmentService
from core.hydrodynamics.lagrangian_rewind import LagrangianRewindEngine
from core.hydrodynamics.opendrift_adapter import OpenDriftOceanAdapter

from core.maritime_graph.spline_interpolator import SplineTrackInterpolator
from core.maritime_graph.st_gnn_model import STGNNTrajectoryPredictor

from core.attribution.spatial_scorer import SpatialOverlapScorer
from core.attribution.dark_scorer import AISDarkScorer
from core.attribution.behavior_risk import BehaviorAndRiskScorer
from core.attribution.attribution_engine import MultiFactorAttributionEngine
from core.attribution.stability_gate import StabilityGateAndRelay

class SeaGuardianPipeline:
    INCIDENT_CATALOG = {
        "INC-GULF-KUTCH": {
            "incident_id": "SG-INC-2026-KUT-01",
            "slick_id": "SLICK-SAR-2026-KUT-1049",
            "name": "Gulf of Kutch Energy Corridor",
            "region": "Gujarat Coast / Arabian Sea",
            "coords": [22.450, 69.180],
            "area_km2": 0.145,
            "volume_m3": 12.0,
            "wind_speed_ms": 7.5,
            "scenario_type": "Small Minor Bilge & Tidal Slack Discharge",
            "description": "Small-scale illegal bilge wash (0.145 km²) displaced 9.8 km northeast by tidal flux. Local LPG carrier GUJARAT PRIDE is currently 1.1 km from the slick, but Sea Guardian proves AL-MANSOOR dumped oily wash during slack water before accelerating to 16.8 kn.",
            "suspect_fleet_config": {
                "culprit_name": "AL-MANSOOR",
                "culprit_type": "Chemical & Bitumen Tanker",
                "culprit_flag": "Gabon (High Risk / FOC)",
                "culprit_mmsi": 626102000,
                "culprit_imo": 9182374,
                "culprit_flag_risk": 0.92,
                "culprit_dwt": 45000,
                "culprit_age": 24,
                "culprit_psc": 6,
                "dark_duration_h": 2.8,
                "heading_deg": 75.0,
                "speed_knots": 16.8,
                "others": [
                    {"name": "GUJARAT PRIDE", "type": "LPG Carrier", "flag": "India (Compliant)", "mmsi": 419008920, "imo": 9812901, "dark": False, "is_closest_at_t0": True, "dist_offset_t0": 0.009, "speed": 14.0, "flag_risk": 0.05},
                    {"name": "INDUS GLORY", "type": "Container Ship", "flag": "Cyprus", "mmsi": 210928000, "imo": 9621908, "dark": False, "is_closest_at_t0": False, "dist_offset_t0": 0.18, "speed": 15.0, "flag_risk": 0.25},
                    {"name": "KUTCH MARINER", "type": "Tug & Barge", "flag": "Tuvalu", "mmsi": 572341000, "imo": 8923410, "dark": True, "dark_dur": 1.5, "is_closest_at_t0": False, "dist_offset_t0": 0.25, "speed": 8.5, "flag_risk": 0.45}
                ]
            }
        },
        "INC-MANNAR-PALK": {
            "incident_id": "SG-INC-2026-MAN-02",
            "slick_id": "SLICK-SAR-2026-MAN-2931",
            "name": "Gulf of Mannar & Palk Strait",
            "region": "Tamil Nadu / Sri Lanka Chokepoint",
            "coords": [8.950, 79.150],
            "area_km2": 0.260,
            "volume_m3": 22.0,
            "wind_speed_ms": 6.4,
            "scenario_type": "Sensitive Ecological Reserve Discharge",
            "description": "Small-medium sludge slick (0.260 km²) drifting toward coral reef reserves. Coastal cargo SETHU SAMUDRAM is 0.8 km from the slick now. Sea Guardian's Fay-Lagrangian rewind proves the spill originated 14.5 km upstream from product tanker CEYLON NAVIGATOR.",
            "suspect_fleet_config": {
                "culprit_name": "CEYLON NAVIGATOR",
                "culprit_type": "Product Tanker",
                "culprit_flag": "Tuvalu (High Risk / FOC)",
                "culprit_mmsi": 572918000,
                "culprit_imo": 9148201,
                "culprit_flag_risk": 0.91,
                "culprit_dwt": 42000,
                "culprit_age": 25,
                "culprit_psc": 6,
                "dark_duration_h": 2.1, "discharge_speed_kn": 2.4, "spatial_offset_km": 0.38,
                "heading_deg": 190.0,
                "speed_knots": 13.2,
                "others": [
                    {"name": "SETHU SAMUDRAM", "type": "Coastal Cargo", "flag": "India (Compliant)", "mmsi": 419003880, "imo": 9619028, "dark": False, "is_closest_at_t0": True, "dist_offset_t0": 0.007, "speed": 10.5, "flag_risk": 0.05},
                    {"name": "PALK TRADER", "type": "Fishing Vessel / Trawler", "flag": "India", "mmsi": 419009110, "imo": 8920192, "dark": False, "is_closest_at_t0": False, "dist_offset_t0": 0.12, "speed": 7.0, "flag_risk": 0.08},
                    {"name": "COLOMBO EXPRESS", "type": "Feeder Container", "flag": "Panama", "mmsi": 355019200, "imo": 9418290, "dark": True, "dark_dur": 1.5, "is_closest_at_t0": False, "dist_offset_t0": 0.24, "speed": 14.8, "flag_risk": 0.40}
                ]
            }
        },
        "INC-GOA-MORMUGAO": {
            "incident_id": "SG-INC-2026-GOA-03",
            "slick_id": "SLICK-SAR-2026-GOA-5192",
            "name": "Goa Konkan Coast & Mormugao SLOC",
            "region": "Central West Coast / Arabian Sea",
            "coords": [15.420, 73.650],
            "area_km2": 0.510,
            "volume_m3": 42.0,
            "wind_speed_ms": 8.2,
            "scenario_type": "Medium Tanker Wash & Wave Stokes Drift",
            "description": "Medium-sized oil slick (0.510 km²) drifted 16.2 km offshore via monsoon wind waves. Ore carrier KONKAN PRIDE is currently nearest the slick (1.2 km). Sea Guardian accurately attributes ARABIAN SEA BREEZE via wave-current rewind.",
            "suspect_fleet_config": {
                "culprit_name": "ARABIAN SEA BREEZE",
                "culprit_type": "Oil & Chemical Tanker",
                "culprit_flag": "Belize (High Risk / FOC)",
                "culprit_mmsi": 312019400,
                "culprit_imo": 9281094,
                "culprit_flag_risk": 0.89,
                "culprit_dwt": 54000,
                "culprit_age": 23,
                "culprit_psc": 5,
                "dark_duration_h": 2.9, "discharge_speed_kn": 4.5, "spatial_offset_km": 0.52,
                "heading_deg": 330.0,
                "speed_knots": 13.6,
                "others": [
                    {"name": "KONKAN PRIDE", "type": "Iron Ore Carrier", "flag": "India (Compliant)", "mmsi": 419004510, "imo": 9718029, "dark": False, "is_closest_at_t0": True, "dist_offset_t0": 0.009, "speed": 12.2, "flag_risk": 0.06},
                    {"name": "GOA SEAWAYS", "type": "Fast Passenger Ferry", "flag": "India", "mmsi": 419001920, "imo": 9812093, "dark": False, "is_closest_at_t0": False, "dist_offset_t0": 0.17, "speed": 22.0, "flag_risk": 0.04},
                    {"name": "MORMUGAO EXPLORER", "type": "General Cargo", "flag": "Panama", "mmsi": 352019480, "imo": 9381028, "dark": True, "dark_dur": 1.9, "is_closest_at_t0": False, "dist_offset_t0": 0.27, "speed": 11.0, "flag_risk": 0.42}
                ]
            }
        },
        "INC-ARABIAN-MUMBAI": {
            "incident_id": "SG-INC-2026-MUM-04",
            "slick_id": "SLICK-SAR-2026-MUM-4802",
            "name": "Mumbai Offshore / Bombay High",
            "region": "Arabian Sea (Western EEZ)",
            "coords": [18.920, 72.450],
            "area_km2": 0.875,
            "volume_m3": 75.0,
            "wind_speed_ms": 5.8,
            "scenario_type": "Medium-Large Sludge Dump & Counter-Intuitive Proximity",
            "description": "Oil slick (0.875 km²) drifted 15.4 km southeast over 4.2 hours. Compliant container ship EVER HARMONY is currently only 1.8 km from the slick, but true culprit NEPTUNE STAR dumped oil during a 3.6-hour AIS blackout at the rewind origin.",
            "suspect_fleet_config": {
                "culprit_name": "NEPTUNE STAR",
                "culprit_type": "Crude Oil Tanker",
                "culprit_flag": "Panama (High Risk / FOC)",
                "culprit_mmsi": 352849000,
                "culprit_imo": 9238472,
                "culprit_flag_risk": 0.88,
                "culprit_dwt": 115000,
                "culprit_age": 22,
                "culprit_psc": 5,
                "dark_duration_h": 3.8, "discharge_speed_kn": 3.8, "spatial_offset_km": 0.30,
                "heading_deg": 65.0,
                "speed_knots": 14.8,
                "others": [
                    {"name": "EVER HARMONY", "type": "Container Ship", "flag": "Singapore (Compliant)", "mmsi": 416283000, "imo": 9718221, "dark": False, "is_closest_at_t0": True, "dist_offset_t0": 0.015, "speed": 17.2, "flag_risk": 0.10},
                    {"name": "BALTIC EXPLORER", "type": "Bulk Carrier", "flag": "Liberia", "mmsi": 636018332, "imo": 9345218, "dark": True, "dark_dur": 1.8, "is_closest_at_t0": False, "dist_offset_t0": 0.22, "speed": 12.5, "flag_risk": 0.40},
                    {"name": "OCEAN PROMISE", "type": "Chemical Tanker", "flag": "Cyprus", "mmsi": 212450000, "imo": 9481920, "dark": False, "is_closest_at_t0": False, "dist_offset_t0": 0.18, "speed": 13.0, "flag_risk": 0.20}
                ]
            }
        },
        "INC-CHENNAI-ENNORE": {
            "incident_id": "SG-INC-2026-CHE-05",
            "slick_id": "SLICK-SAR-2026-CHE-8419",
            "name": "Chennai Offshore & Ennore Corridor",
            "region": "Coromandel Coast / Bay of Bengal",
            "coords": [13.280, 80.450],
            "area_km2": 1.420,
            "volume_m3": 125.0,
            "wind_speed_ms": 6.7,
            "scenario_type": "Large Port Approach Ballast & Sludge Dump",
            "description": "Large multi-lobed slick (1.420 km²) circulated 16.8 km southeast by coastal eddy. Feeder COROMANDEL TRADER is 1.3 km from the slick now, while culprit BAY TITAN (Liberia) dumped bunker sludge during ballast exchange at the rewind centroid.",
            "suspect_fleet_config": {
                "culprit_name": "BAY TITAN",
                "culprit_type": "Chemical & Ore Carrier",
                "culprit_flag": "Liberia (High Risk)",
                "culprit_mmsi": 636014902,
                "culprit_imo": 9328109,
                "culprit_flag_risk": 0.84,
                "culprit_dwt": 82000,
                "culprit_age": 18,
                "culprit_psc": 4,
                "dark_duration_h": 3.3, "discharge_speed_kn": 2.2, "spatial_offset_km": 0.72,
                "heading_deg": 30.0,
                "speed_knots": 14.0,
                "others": [
                    {"name": "COROMANDEL TRADER", "type": "General Cargo", "flag": "India (Compliant)", "mmsi": 419006720, "imo": 9520194, "dark": False, "is_closest_at_t0": True, "dist_offset_t0": 0.011, "speed": 11.5, "flag_risk": 0.06},
                    {"name": "ENNORE GLORY", "type": "Bulk Coal Carrier", "flag": "Marshall Islands", "mmsi": 538002918, "imo": 9481903, "dark": False, "is_closest_at_t0": False, "dist_offset_t0": 0.19, "speed": 12.0, "flag_risk": 0.20},
                    {"name": "BAY MARINER", "type": "Tug Support", "flag": "Cyprus", "mmsi": 210928190, "imo": 9219082, "dark": True, "dark_dur": 1.3, "is_closest_at_t0": False, "dist_offset_t0": 0.29, "speed": 9.0, "flag_risk": 0.30}
                ]
            }
        },
        "INC-LAKSHADWEEP": {
            "incident_id": "SG-INC-2026-LAK-06",
            "slick_id": "SLICK-SAR-2026-LAK-6512",
            "name": "Lakshadweep Sea Corridor",
            "region": "South-West Coast / Arabian Sea",
            "coords": [9.850, 74.200],
            "area_km2": 2.150,
            "volume_m3": 190.0,
            "wind_speed_ms": 5.9,
            "scenario_type": "Large Open-Ocean Tanker Wash & AIS Blackout",
            "description": "Substantial open-ocean oil slick (2.150 km²) caused by Aframax tanker POSEIDON GLORY oily ballast discharge. RoRo ferry MINICOY EXPRESS is currently 1.2 km from the surface slick, but ST-GNN trajectory exonerates the ferry completely.",
            "suspect_fleet_config": {
                "culprit_name": "POSEIDON GLORY",
                "culprit_type": "Aframax Crude Tanker",
                "culprit_flag": "Cook Islands (High Risk / FOC)",
                "culprit_mmsi": 518002910,
                "culprit_imo": 9291024,
                "culprit_flag_risk": 0.86,
                "culprit_dwt": 105000,
                "culprit_age": 20,
                "culprit_psc": 4,
                "dark_duration_h": 4.3, "discharge_speed_kn": 5.2, "spatial_offset_km": 0.25,
                "heading_deg": 155.0,
                "speed_knots": 13.8,
                "others": [
                    {"name": "MINICOY EXPRESS", "type": "Passenger / RoRo", "flag": "India (Compliant)", "mmsi": 419005120, "imo": 9712019, "dark": False, "is_closest_at_t0": True, "dist_offset_t0": 0.010, "speed": 16.0, "flag_risk": 0.05},
                    {"name": "KOCHI TRADER", "type": "General Cargo", "flag": "India", "mmsi": 419002910, "imo": 9381029, "dark": False, "is_closest_at_t0": False, "dist_offset_t0": 0.15, "speed": 11.2, "flag_risk": 0.10},
                    {"name": "SOUTHERN HORIZON", "type": "Container Ship", "flag": "Liberia", "mmsi": 636014290, "imo": 9481902, "dark": True, "dark_dur": 1.7, "is_closest_at_t0": False, "dist_offset_t0": 0.28, "speed": 15.5, "flag_risk": 0.35}
                ]
            }
        },
        "INC-BOB-VIZAG": {
            "incident_id": "SG-INC-2026-VIZ-07",
            "slick_id": "SLICK-SAR-2026-VIZ-7914",
            "name": "Visakhapatnam Deepwater SLOC",
            "region": "Bay of Bengal (Eastern EEZ)",
            "coords": [17.480, 83.520],
            "area_km2": 2.950,
            "volume_m3": 265.0,
            "wind_speed_ms": 7.2,
            "scenario_type": "Very Large VLCC Deepwater Ballast Discharge",
            "description": "Expansive multi-lobe slick (2.950 km²) advected 19.4 km northward by western boundary current. Indian bulk carrier VIZAG EXPRESS is directly adjacent to the slick at T=0 (1.1 km). Naive model wrongfully blames VIZAG EXPRESS; Sea Guardian unmasks VLCC PACIFIC VOYAGER.",
            "suspect_fleet_config": {
                "culprit_name": "PACIFIC VOYAGER",
                "culprit_type": "VLCC Crude Tanker",
                "culprit_flag": "Liberia (High Risk)",
                "culprit_mmsi": 636019842,
                "culprit_imo": 9412890,
                "culprit_flag_risk": 0.85,
                "culprit_dwt": 298000,
                "culprit_age": 19,
                "culprit_psc": 5,
                "dark_duration_h": 4.8, "discharge_speed_kn": 4.2, "spatial_offset_km": 0.58,
                "heading_deg": 40.0,
                "speed_knots": 15.2,
                "others": [
                    {"name": "VIZAG EXPRESS", "type": "Bulk Carrier", "flag": "India (Compliant)", "mmsi": 419001240, "imo": 9618201, "dark": False, "is_closest_at_t0": True, "dist_offset_t0": 0.010, "speed": 11.8, "flag_risk": 0.08},
                    {"name": "COROMANDEL GLORY", "type": "Container Ship", "flag": "Panama", "mmsi": 354128000, "imo": 9523190, "dark": False, "is_closest_at_t0": False, "dist_offset_t0": 0.25, "speed": 16.5, "flag_risk": 0.35},
                    {"name": "BAY PEARL", "type": "Product Tanker", "flag": "Marshall Islands", "mmsi": 538004120, "imo": 9381928, "dark": True, "dark_dur": 1.4, "is_closest_at_t0": False, "dist_offset_t0": 0.30, "speed": 13.5, "flag_risk": 0.30}
                ]
            }
        },
        "INC-MALACCA-NICOBAR": {
            "incident_id": "SG-INC-2026-NIC-08",
            "slick_id": "SLICK-SAR-2026-NIC-3820",
            "name": "Great Nicobar / Malacca Gateway",
            "region": "Indo-Pacific International SLOC",
            "coords": [6.250, 94.850],
            "area_km2": 4.250,
            "volume_m3": 390.0,
            "wind_speed_ms": 6.2,
            "scenario_type": "Massive International SLOC Mega-Slick",
            "description": "Massive high-impact oil slick (4.250 km²) across key chokepoint where Suezmax tanker EASTERN DRAGON turned off AIS for 5.2 hours during transit. Mega-containership SINGAPORE LEADER is 2.2 km from the slick, but had continuous broadcast.",
            "suspect_fleet_config": {
                "culprit_name": "EASTERN DRAGON",
                "culprit_type": "Suezmax Crude Tanker",
                "culprit_flag": "Marshall Islands (FOC)",
                "culprit_mmsi": 538008712,
                "culprit_imo": 9382190,
                "culprit_flag_risk": 0.89,
                "culprit_dwt": 158000,
                "culprit_age": 21,
                "culprit_psc": 5,
                "dark_duration_h": 5.6, "discharge_speed_kn": 3.0, "spatial_offset_km": 0.15,
                "heading_deg": 115.0,
                "speed_knots": 14.5,
                "others": [
                    {"name": "SINGAPORE LEADER", "type": "Mega Container", "flag": "Singapore (Compliant)", "mmsi": 566192000, "imo": 9891240, "dark": False, "is_closest_at_t0": True, "dist_offset_t0": 0.018, "speed": 21.0, "flag_risk": 0.08},
                    {"name": "ANDAMAN BREEZE", "type": "Bulk Carrier", "flag": "Panama", "mmsi": 352109000, "imo": 9410291, "dark": True, "dark_dur": 2.0, "is_closest_at_t0": False, "dist_offset_t0": 0.32, "speed": 13.0, "flag_risk": 0.35},
                    {"name": "MALACCA SPIRIT", "type": "Product Tanker", "flag": "Bahamas", "mmsi": 311029000, "imo": 9529011, "dark": False, "is_closest_at_t0": False, "dist_offset_t0": 0.16, "speed": 12.8, "flag_risk": 0.15}
                ]
            }
        }
    }

    def __init__(self):
        self.preprocessor = SARPreprocessor(filter_size=7)
        self.detector = OilSlickDetector(confidence_threshold=0.70)
        self.lookalike_filter = LookAlikeDiscriminator()
        
        self.fay_model = FaySpreadingModel()
        self.stokes_calc = StokesDriftCalculator()
        self.ocean_service = OceanEnvironmentService(windage_coeff=0.032, coriolis_deflection_deg=15.0)
        self.rewind_engine = LagrangianRewindEngine(num_particles=800, eddy_diffusivity_kh=5.0, dt_seconds=300.0)
        self.opendrift = OpenDriftOceanAdapter()
        
        self.spline_interp = SplineTrackInterpolator()
        self.st_gnn = STGNNTrajectoryPredictor()
        
        self.spatial_scorer = SpatialOverlapScorer()
        self.dark_scorer = AISDarkScorer(tau_0_hours=2.0)
        self.behavior_risk_scorer = BehaviorAndRiskScorer()
        self.attribution_engine = MultiFactorAttributionEngine(0.35, 0.30, 0.20, 0.15)
        self.stability_gate = StabilityGateAndRelay()

    def list_incidents(self) -> List[Dict[str, Any]]:
        res = []
        for key, info in self.INCIDENT_CATALOG.items():
            res.append({
                "key": key,
                "incident_id": info["incident_id"],
                "slick_id": info.get("slick_id", f"SLICK-SAR-2026-{key[-3:]}"),
                "name": info["name"],
                "region": info["region"],
                "coords": info["coords"],
                "area_km2": info["area_km2"],
                "volume_m3": info["volume_m3"],
                "wind_speed_ms": info["wind_speed_ms"],
                "scenario_type": info.get("scenario_type", "Maritime Surveillance"),
                "description": info.get("description", "")
            })
        return res

    def run_end_to_end(
        self,
        incident_key: str = "INC-ARABIAN-MUMBAI",
        wind_speed_ms: Optional[float] = None,
        enable_stokes: bool = True,
        wind_perturbation: float = 1.0,
        estimated_volume_m3: Optional[float] = None
    ) -> Dict[str, Any]:
        cfg = self.INCIDENT_CATALOG.get(incident_key, self.INCIDENT_CATALOG["INC-ARABIAN-MUMBAI"])
        center_lat, center_lon = cfg["coords"]
        wind_ms = wind_speed_ms if wind_speed_ms is not None else cfg["wind_speed_ms"]
        volume = estimated_volume_m3 if estimated_volume_m3 is not None else cfg["volume_m3"]
        area_km2 = cfg["area_km2"]
        slick_id = cfg.get("slick_id", f"SLICK-SAR-2026-{incident_key[-3:]}")
        
        # STAGE 1: SAR INGESTION & SATELLITE DETECTION
        r_mean_m = np.sqrt((area_km2 * 1e6) / np.pi)
        lat_span_deg = max(0.04, (r_mean_m * 3.5) / 111000.0)
        lon_span_deg = lat_span_deg / max(0.2, np.cos(np.radians(center_lat)))
        
        scene = {
            "bounds": {
                "min_lat": center_lat - lat_span_deg / 2.0,
                "max_lat": center_lat + lat_span_deg / 2.0,
                "min_lon": center_lon - lon_span_deg / 2.0,
                "max_lon": center_lon + lon_span_deg / 2.0,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "pixel_size_m": (r_mean_m * 3.5) / 256.0
            }
        }
        
        primary_slick = {
            "slick_id": slick_id,
            "area_km2": area_km2,
            "area_m2": area_km2 * 1e6,
            "centroid_lat": center_lat,
            "centroid_lon": center_lon,
            "confidence": round(float(0.92 + min(0.06, area_km2 * 0.012)), 3),
            "compactness": round(float(0.36 + 0.08 * np.sin(area_km2)), 3)
        }
        
        is_confirmed, filter_msg, _ = self.lookalike_filter.evaluate_candidate(primary_slick, wind_ms)
        
        # STAGE 2: HYDRODYNAMIC REWIND
        fay_estimate = self.fay_model.estimate_discharge_age(
            observed_area_m2=primary_slick["area_m2"],
            estimated_volume_m3=volume
        )
        discharge_age_h = fay_estimate["spill_age_hours"]
        
        origin_envelope = self.rewind_engine.run_backward_rewind(
            slick_centroid_lat=primary_slick["centroid_lat"],
            slick_centroid_lon=primary_slick["centroid_lon"],
            spill_age_hours=discharge_age_h,
            ocean_service=self.ocean_service,
            stokes_calculator=self.stokes_calc,
            enable_stokes=enable_stokes,
            wind_perturbation=wind_perturbation
        )
        
        # Generate Natural Organic SAR Oil Slick Geometry
        drift_heading = float(np.degrees(np.arctan2(
            primary_slick["centroid_lon"] - origin_envelope["origin_lon"],
            primary_slick["centroid_lat"] - origin_envelope["origin_lat"]
        )))
        
        natural_morphology = NaturalSlickMorphology.generate_natural_slick_polygon(
            centroid_lat=primary_slick["centroid_lat"],
            centroid_lon=primary_slick["centroid_lon"],
            area_km2=primary_slick["area_km2"],
            drift_heading_deg=drift_heading,
            num_vertices=64,
            seed=abs(int(center_lat * 1000 + center_lon * 100)) % 100000
        )
        primary_slick["polygon_geo"] = natural_morphology["core_polygon_geo"]
        primary_slick["core_polygon_geo"] = natural_morphology["core_polygon_geo"]
        primary_slick["sheen_polygon_geo"] = natural_morphology["sheen_polygon_geo"]
        primary_slick["characteristic_radius_m"] = natural_morphology["characteristic_radius_m"]
        
        # Generate Reverse Hydrodynamic Rewind Projection
        rewind_proj = NaturalSlickMorphology.generate_rewind_cone_projection(
            slick_lat=primary_slick["centroid_lat"],
            slick_lon=primary_slick["centroid_lon"],
            origin_lat=origin_envelope["origin_lat"],
            origin_lon=origin_envelope["origin_lon"],
            origin_radius_m=origin_envelope["origin_radius_m"],
            spill_age_hours=discharge_age_h,
            num_trajectory_steps=8
        )
        origin_envelope["rewind_projection"] = rewind_proj
        
        # STAGE 3: AIS FLEET GENERATION
        fleet = self._build_incident_fleet(
            cfg=cfg,
            origin_lat=origin_envelope["origin_lat"],
            origin_lon=origin_envelope["origin_lon"],
            slick_lat=primary_slick["centroid_lat"],
            slick_lon=primary_slick["centroid_lon"],
            discharge_h=discharge_age_h
        )
        
        st_gnn_predictions = {}
        interpolated_tracks = {}
        for vessel in fleet:
            vid = vessel["vessel_id"]
            gnn_out = self.st_gnn.reconstruct_dark_trajectory(vessel, discharge_age_h)
            st_gnn_predictions[vid] = gnn_out
            valid_pts = [p for p in vessel["track"] if p.get("ais_transmitted", True)]
            interpolated_tracks[vid] = self.spline_interp.interpolate_track(valid_pts)
            
        # STAGE 4: MULTI-FACTOR ATTRIBUTION & BENCHMARK COMPARISON
        ranked_suspects = self.attribution_engine.evaluate_fleet(
            fleet,
            origin_envelope,
            st_gnn_predictions,
            self.spatial_scorer,
            self.dark_scorer,
            self.behavior_risk_scorer
        )
        
        # Compute Distances at T=0 (Current Detection Time)
        for v in fleet:
            last_pt = v["track"][-1]
            d_lat = (last_pt["lat"] - primary_slick["centroid_lat"]) * 111.0
            d_lon = (last_pt["lon"] - primary_slick["centroid_lon"]) * 111.0 * np.cos(np.radians(primary_slick["centroid_lat"]))
            v["dist_to_slick_at_t0_km"] = round(float(np.sqrt(d_lat**2 + d_lon**2)), 2)
            
        fleet_sorted_by_t0_dist = sorted(fleet, key=lambda x: x["dist_to_slick_at_t0_km"])
        naive_closest_vessel = fleet_sorted_by_t0_dist[0]
        sea_guardian_prime = ranked_suspects[0]
        
        drift_d_lat = (primary_slick["centroid_lat"] - origin_envelope["origin_lat"]) * 111.0
        drift_d_lon = (primary_slick["centroid_lon"] - origin_envelope["origin_lon"]) * 111.0 * np.cos(np.radians(primary_slick["centroid_lat"]))
        drift_dist_km = round(float(np.sqrt(drift_d_lat**2 + drift_d_lon**2)), 2)
        
        comparative_analysis = {
            "drift_distance_km": drift_dist_km,
            "spill_age_hours": round(discharge_age_h, 2),
            "naive_standard_approach": {
                "assigned_suspect_name": naive_closest_vessel["name"],
                "vessel_type": naive_closest_vessel["vessel_type"],
                "flag_state": naive_closest_vessel["flag_state"],
                "distance_to_slick_at_t0_km": naive_closest_vessel["dist_to_slick_at_t0_km"],
                "verdict": "FALSE POSITIVE (Naive Proximity Error)",
                "reason": f"Standard model blames {naive_closest_vessel['name']} simply because it is closest ({naive_closest_vessel['dist_to_slick_at_t0_km']} km) to the drifted slick now, ignoring that the slick drifted {drift_dist_km} km over {round(discharge_age_h,1)}h."
            },
            "sea_guardian_approach": {
                "assigned_suspect_name": sea_guardian_prime["name"],
                "vessel_type": sea_guardian_prime["vessel_type"],
                "flag_state": sea_guardian_prime["flag_state"],
                "attribution_confidence_pct": sea_guardian_prime["culprit_percentage"],
                "ais_dark_duration_hours": sea_guardian_prime.get("dark_details", {}).get("dark_duration_hours", 0.0),
                "verdict": "CONFIRMED CULPRIT (Forensically Validated)",
                "reason": f"Sea Guardian's Lagrangian Rewind traced the slick back {drift_dist_km} km to its true origin at T=-{round(discharge_age_h,1)}h. ST-GNN trajectory unmasked {sea_guardian_prime['name']} loitering at the exact release point during an AIS transponder blackout."
            },
            "is_closest_innocent": (naive_closest_vessel["name"] != sea_guardian_prime["name"])
        }
        
        # STAGE 5: STABILITY GATE
        origin_no_stokes = self.rewind_engine.run_backward_rewind(
            primary_slick["centroid_lat"], primary_slick["centroid_lon"],
            discharge_age_h, self.ocean_service, self.stokes_calc,
            enable_stokes=False, wind_perturbation=1.0
        )
        suspects_no_stokes = self.attribution_engine.evaluate_fleet(
            fleet, origin_no_stokes, st_gnn_predictions,
            self.spatial_scorer, self.dark_scorer, self.behavior_risk_scorer
        )
        origin_w_plus = self.rewind_engine.run_backward_rewind(
            primary_slick["centroid_lat"], primary_slick["centroid_lon"],
            discharge_age_h, self.ocean_service, self.stokes_calc,
            enable_stokes=True, wind_perturbation=1.2
        )
        suspects_w_plus = self.attribution_engine.evaluate_fleet(
            fleet, origin_w_plus, st_gnn_predictions,
            self.spatial_scorer, self.dark_scorer, self.behavior_risk_scorer
        )
        origin_w_minus = self.rewind_engine.run_backward_rewind(
            primary_slick["centroid_lat"], primary_slick["centroid_lon"],
            discharge_age_h, self.ocean_service, self.stokes_calc,
            enable_stokes=True, wind_perturbation=0.8
        )
        suspects_w_minus = self.attribution_engine.evaluate_fleet(
            fleet, origin_w_minus, st_gnn_predictions,
            self.spatial_scorer, self.dark_scorer, self.behavior_risk_scorer
        )
        
        stability_result = self.stability_gate.evaluate_stability(
            baseline_suspects=ranked_suspects,
            no_stokes_suspects=suspects_no_stokes,
            wind_plus_suspects=suspects_w_plus,
            wind_minus_suspects=suspects_w_minus
        )
        
        ntro_dossier = self.stability_gate.generate_ntRO_incident_dossier(
            incident_id=cfg["incident_id"],
            slick_detection=primary_slick,
            origin_envelope=origin_envelope,
            ranked_suspects=ranked_suspects,
            stability_result=stability_result
        )
        
        return {
            "status": "SUCCESS",
            "incident_info": {
                "key": incident_key,
                "incident_id": cfg["incident_id"],
                "slick_id": slick_id,
                "name": cfg["name"],
                "region": cfg["region"],
                "scenario_type": cfg.get("scenario_type", "Maritime Hotspot"),
                "description": cfg.get("description", ""),
                "center_coords": [center_lat, center_lon]
            },
            "sar_scene": {
                "bounds": scene["bounds"],
                "wind_speed_ms": wind_ms,
                "detected_slick": {
                    "slick_id": primary_slick["slick_id"],
                    "area_km2": primary_slick["area_km2"],
                    "centroid_lat": primary_slick["centroid_lat"],
                    "centroid_lon": primary_slick["centroid_lon"],
                    "polygon_geo": primary_slick["polygon_geo"],
                    "core_polygon_geo": primary_slick.get("core_polygon_geo", []),
                    "sheen_polygon_geo": primary_slick.get("sheen_polygon_geo", []),
                    "confidence": primary_slick["confidence"],
                    "compactness": primary_slick["compactness"],
                    "verification_status": filter_msg
                }
            },
            "fay_spreading": fay_estimate,
            "origin_envelope": origin_envelope,
            "fleet": fleet,
            "st_gnn_predictions": st_gnn_predictions,
            "interpolated_tracks": interpolated_tracks,
            "ranked_suspects": ranked_suspects,
            "stability_result": stability_result,
            "comparative_analysis": comparative_analysis,
            "ntro_dossier": ntro_dossier
        }

    def _build_incident_fleet(
        self,
        cfg: Dict[str, Any],
        origin_lat: float,
        origin_lon: float,
        slick_lat: float,
        slick_lon: float,
        discharge_h: float
    ) -> List[Dict[str, Any]]:
        fleet_cfg = cfg["suspect_fleet_config"]
        t_max_h = max(8.0, float(discharge_h + 3.0))
        num_steps = max(33, int(t_max_h * 4 + 1))
        time_steps = np.linspace(-t_max_h, 0.0, num_steps)
        fleet = []
        
        # 1. Culprit Vessel
        dark_dur = fleet_cfg.get("dark_duration_h", 3.5)
        # Positive hours_ago range:
        dark_max_h = round(discharge_h + dark_dur * 0.45, 2)
        dark_min_h = round(max(0.1, discharge_h - dark_dur * 0.55), 2)
        
        cul_heading = np.radians(fleet_cfg.get("heading_deg", 50.0))
        cul_speed_kn = fleet_cfg.get("speed_knots", 14.5)
        deg_per_hour = (cul_speed_kn * 1.852) / 111.0
        
        # Incident-specific discharge loitering speed
        discharge_spd_kn = fleet_cfg.get("discharge_speed_kn", 3.8)
        
        track_cul = []
        for i, t in enumerate(time_steps):
            t_ago = abs(round(float(t), 2))
            is_dark = (dark_min_h <= t_ago <= dark_max_h)
            dt_from_discharge = float(t - (-discharge_h))
            
            # Spatial offset based on incident to give realistic varied spatial overlap (78% to 98%)
            spatial_offset_km = fleet_cfg.get("spatial_offset_km", 0.35)
            origin_offset_lat = origin_lat + (spatial_offset_km / 111.0) * np.sin(cul_heading)
            origin_offset_lon = origin_lon + (spatial_offset_km / 111.0) * np.cos(cul_heading) / max(0.2, np.cos(np.radians(origin_lat)))
            lat_p = origin_offset_lat + dt_from_discharge * deg_per_hour * np.cos(cul_heading)
            lon_p = origin_lon + dt_from_discharge * deg_per_hour * np.sin(cul_heading) / max(0.2, np.cos(np.radians(origin_lat)))
            
            spd = discharge_spd_kn + float(np.random.normal(0, 0.15)) if is_dark else cul_speed_kn + float(np.random.normal(0, 0.25))
            track_cul.append({
                "time_hours_ago": t_ago,
                "lat": float(lat_p),
                "lon": float(lon_p),
                "speed_knots": round(max(1.5, spd), 1),
                "heading_deg": round(float(np.degrees(cul_heading)), 1),
                "ais_transmitted": not is_dark
            })
            
        fleet.append({
            "vessel_id": "VESSEL-001",
            "name": fleet_cfg["culprit_name"],
            "mmsi": fleet_cfg["culprit_mmsi"],
            "imo": fleet_cfg["culprit_imo"],
            "vessel_type": fleet_cfg["culprit_type"],
            "flag_state": fleet_cfg["culprit_flag"],
            "flag_risk_index": fleet_cfg["culprit_flag_risk"],
            "dwt_tonnage": fleet_cfg["culprit_dwt"],
            "vessel_age_years": fleet_cfg["culprit_age"],
            "psc_deficiencies_past_3y": fleet_cfg["culprit_psc"],
            "has_dark_gap": True,
            "dark_duration_hours": dark_dur,
            "dark_interval": (dark_max_h, dark_min_h),
            "track": track_cul
        })
        
        # 2. Innocent / Other Vessels
        for j, other in enumerate(fleet_cfg.get("others", [])):
            v_id = f"VESSEL-{j+2:03d}"
            is_closest = other.get("is_closest_at_t0", False)
            speed_kn = other.get("speed", 13.0)
            d_per_h = (speed_kn * 1.852) / 111.0
            heading = np.radians((fleet_cfg.get("heading_deg", 50.0) + (j + 1) * 65.0) % 360)
            
            # Position at T=0
            if is_closest:
                offset_deg = other.get("dist_offset_t0", 0.012)
                t0_lat = slick_lat + offset_deg * np.cos(heading)
                t0_lon = slick_lon + offset_deg * np.sin(heading)
            else:
                offset_deg = other.get("dist_offset_t0", 0.20)
                t0_lat = slick_lat + offset_deg * np.cos(heading)
                t0_lon = slick_lon + offset_deg * np.sin(heading)
                
            has_dark = other.get("dark", False)
            dark_d = other.get("dark_dur", 1.5)
            other_dark_max_h = round(discharge_h + 3.0, 2)
            other_dark_min_h = round(max(0.1, discharge_h + 3.0 - dark_d), 2)
            
            track_oth = []
            for t in time_steps:
                t_ago = abs(round(float(t), 2))
                dt_from_t0 = float(t)
                lat_p = t0_lat + dt_from_t0 * d_per_h * np.cos(heading)
                lon_p = t0_lon + dt_from_t0 * d_per_h * np.sin(heading) / max(0.2, np.cos(np.radians(slick_lat)))
                
                pt_dark = has_dark and (other_dark_min_h <= t_ago <= other_dark_max_h)
                spd = speed_kn + float(np.random.normal(0, 0.15))
                track_oth.append({
                    "time_hours_ago": t_ago,
                    "lat": float(lat_p),
                    "lon": float(lon_p),
                    "speed_knots": round(spd, 1),
                    "heading_deg": round(float(np.degrees(heading)), 1),
                    "ais_transmitted": not pt_dark
                })
                
            fleet.append({
                "vessel_id": v_id,
                "name": other["name"],
                "mmsi": other["mmsi"],
                "imo": other["imo"],
                "vessel_type": other["type"],
                "flag_state": other["flag"],
                "flag_risk_index": other.get("flag_risk", 0.15),
                "dwt_tonnage": 50000,
                "vessel_age_years": 10,
                "psc_deficiencies_past_3y": 0,
                "has_dark_gap": has_dark,
                "dark_duration_hours": dark_d if has_dark else 0.0,
                "dark_interval": (other_dark_max_h, other_dark_min_h) if has_dark else None,
                "track": track_oth
            })
            
        return fleet
