"""
Project DRISHTI - Module 3: AIS Data Generator & Stream Handler
Simulates maritime traffic in Arabian Sea / Mumbai offshore corridor,
including compliant commercial vessels and illicit dark vessels (AIS transponder blackout).
"""
import numpy as np
from typing import List, Dict, Any

class AISDataGenerator:
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)

    def generate_fleet_scenario(
        self,
        origin_lat: float = 18.86,
        origin_lon: float = 72.38,
        discharge_hours_ago: float = 4.2
    ) -> List[Dict[str, Any]]:
        """
        Generates 4 candidate vessels in the vicinity of the spill:
        1. Dark Vessel A (Culprit): Crude Oil Tanker with 3.5h blackout precisely passing through origin zone, speed drop.
        2. Compliant Vessel B: Container Ship on standard sea lane, continuous AIS, 15 km away.
        3. Dark Vessel C (Innocent): Bulk Carrier with AIS gap elsewhere in the corridor (no spatial match).
        4. Compliant Vessel D: Chemical Tanker crossing outer perimeter.
        """
        vessels = []
        time_steps = np.linspace(-8.0, 0.0, 33) # 8 hours history to t_now (every 15 min)
        
        # 1. PRIMARY SUSPECT (Dark Tanker - "NEPTUNE STAR")
        # Starts SW, travels NE, goes dark at t = -5.5h to -2.0h, passes right through origin at t = -4.2h
        t_dark_start = -5.5
        t_dark_end = -2.0
        
        # Origin is at discharge_hours_ago (-4.2)
        # Trajectory equation parameterized
        lat_nep = origin_lat + (time_steps - (-discharge_hours_ago)) * 0.035
        lon_nep = origin_lon + (time_steps - (-discharge_hours_ago)) * 0.040
        
        # Speed drop during blackout (decelerates from 14.5 knots to 5.2 knots for bilge/tank washing)
        speeds_nep = []
        for t in time_steps:
            if t_dark_start <= t <= t_dark_end:
                speeds_nep.append(5.2 + np.random.normal(0, 0.3))
            else:
                speeds_nep.append(14.5 + np.random.normal(0, 0.4))
                
        # AIS transmissions: Missing between t_dark_start and t_dark_end
        points_nep = []
        for i, t in enumerate(time_steps):
            is_dark = (t_dark_start < t < t_dark_end)
            points_nep.append({
                "time_hours_ago": abs(round(float(t), 2)),
                "lat": float(lat_nep[i]),
                "lon": float(lon_nep[i]),
                "speed_knots": round(float(speeds_nep[i]), 1),
                "heading_deg": 48.0,
                "ais_transmitted": not is_dark
            })
            
        vessels.append({
            "vessel_id": "VESSEL-001",
            "name": "NEPTUNE STAR",
            "mmsi": 352849000,
            "imo": 9238472,
            "vessel_type": "Crude Oil Tanker",
            "flag_state": "Panama (High Risk / FOC)",
            "flag_risk_index": 0.85,
            "dwt_tonnage": 115000,
            "vessel_age": 22,
            "past_psc_deficiencies": 4,
            "has_dark_gap": True,
            "dark_duration_hours": abs(t_dark_end - t_dark_start),
            "dark_interval": [abs(t_dark_start), abs(t_dark_end)],
            "track": points_nep
        })
        
        # 2. VESSEL 002 ("EVER HARMONY" - Container Carrier, fully compliant)
        lat_con = origin_lat + 0.12 + (time_steps + 4.0) * 0.05
        lon_con = origin_lon - 0.10 + (time_steps + 4.0) * 0.03
        points_con = [
            {
                "time_hours_ago": abs(round(float(t), 2)),
                "lat": float(lat_con[i]),
                "lon": float(lon_con[i]),
                "speed_knots": 18.2 + float(np.random.normal(0, 0.2)),
                "heading_deg": 32.0,
                "ais_transmitted": True
            }
            for i, t in enumerate(time_steps)
        ]
        vessels.append({
            "vessel_id": "VESSEL-002",
            "name": "EVER HARMONY",
            "mmsi": 416283000,
            "imo": 9718221,
            "vessel_type": "Container Ship",
            "flag_state": "Singapore (Low Risk)",
            "flag_risk_index": 0.10,
            "dwt_tonnage": 68000,
            "vessel_age": 6,
            "past_psc_deficiencies": 0,
            "has_dark_gap": False,
            "dark_duration_hours": 0.0,
            "dark_interval": [0, 0],
            "track": points_con
        })
        
        # 3. VESSEL 003 ("BALTIC EXPLORER" - Bulk Carrier, AIS dark elsewhere)
        # Dark between -7.0h and -5.0h, but located 35 km away
        lat_blk = origin_lat - 0.25 + (time_steps + 3.0) * 0.02
        lon_blk = origin_lon + 0.30 - (time_steps + 3.0) * 0.04
        points_blk = []
        for i, t in enumerate(time_steps):
            is_dark = (-7.0 < t < -5.0)
            points_blk.append({
                "time_hours_ago": abs(round(float(t), 2)),
                "lat": float(lat_blk[i]),
                "lon": float(lon_blk[i]),
                "speed_knots": 11.5 + float(np.random.normal(0, 0.3)),
                "heading_deg": 120.0,
                "ais_transmitted": not is_dark
            })
        vessels.append({
            "vessel_id": "VESSEL-003",
            "name": "BALTIC EXPLORER",
            "mmsi": 636018332,
            "imo": 9345218,
            "vessel_type": "Bulk Carrier",
            "flag_state": "Liberia (Moderate Risk)",
            "flag_risk_index": 0.45,
            "dwt_tonnage": 82000,
            "vessel_age": 14,
            "past_psc_deficiencies": 1,
            "has_dark_gap": True,
            "dark_duration_hours": 2.0,
            "dark_interval": [7.0, 5.0],
            "track": points_blk
        })
        
        # 4. VESSEL 004 ("OCEAN PROMISE" - Chemical Tanker, compliant)
        lat_chem = origin_lat - 0.08 + (time_steps + 2.0) * 0.04
        lon_chem = origin_lon - 0.18 + (time_steps + 2.0) * 0.045
        points_chem = [
            {
                "time_hours_ago": abs(round(float(t), 2)),
                "lat": float(lat_chem[i]),
                "lon": float(lon_chem[i]),
                "speed_knots": 13.1 + float(np.random.normal(0, 0.2)),
                "heading_deg": 50.0,
                "ais_transmitted": True
            }
            for i, t in enumerate(time_steps)
        ]
        vessels.append({
            "vessel_id": "VESSEL-004",
            "name": "OCEAN PROMISE",
            "mmsi": 212450000,
            "imo": 9481920,
            "vessel_type": "Chemical Tanker",
            "flag_state": "Cyprus (Low Risk)",
            "flag_risk_index": 0.20,
            "dwt_tonnage": 37000,
            "vessel_age": 10,
            "past_psc_deficiencies": 0,
            "has_dark_gap": False,
            "dark_duration_hours": 0.0,
            "dark_interval": [0, 0],
            "track": points_chem
        })
        
        return vessels
