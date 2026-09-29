"""
Project Sea Guardian - Natural SAR Oil Slick Morphology & Reverse Envelope Generator
Generates realistic, fractal, multi-lobed oil slicks with wind shear, feathering tails,
and projects back the Lagrangian hydrodynamic rewind envelope to the discharge origin.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

class NaturalSlickMorphology:
    @staticmethod
    def generate_natural_slick_polygon(
        centroid_lat: float,
        centroid_lon: float,
        area_km2: float,
        drift_heading_deg: float = 45.0,
        num_vertices: int = 64,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Constructs an organic, non-uniform SAR oil slick with:
        - Asymmetric elongation along the wind/current shear vector
        - Multi-harmonic fractal boundary perturbations (turbulent ocean eddies)
        - Dense core emulsion + feathering sheen halo
        """
        rng = np.random.RandomState(int(abs(centroid_lat * 1000 + centroid_lon * 100) + seed) % 100000)
        
        # Characteristic radius in meters
        r_mean_m = np.sqrt((area_km2 * 1e6) / np.pi)
        
        # Scaling degrees
        m_per_deg_lat = 111000.0
        m_per_deg_lon = 111000.0 * np.cos(np.radians(centroid_lat))
        
        theta = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        drift_rad = np.radians(drift_heading_deg)
        
        # Multi-harmonic fractal boundary perturbation
        phi1 = rng.uniform(0, 2 * np.pi)
        phi2 = rng.uniform(0, 2 * np.pi)
        phi3 = rng.uniform(0, 2 * np.pi)
        
        r_profile = np.ones_like(theta)
        # Elongation along wind shear (elongation ratio ~2.4)
        r_profile += 0.55 * np.cos(2 * (theta - drift_rad))
        # Asymmetric tail (thinner and longer in upwind direction)
        r_profile += 0.35 * np.cos(theta - (drift_rad + np.pi))
        # Turbulent lobes
        r_profile += 0.22 * np.sin(3 * theta + phi1)
        # Fine eddy filaments
        r_profile += 0.14 * np.sin(5 * theta + phi2)
        r_profile += 0.08 * np.cos(8 * theta + phi3)
        # High frequency roughness
        r_profile += rng.normal(0, 0.04, size=num_vertices)
        
        r_profile = np.maximum(0.25, r_profile)
        # Scale to match physical area
        r_profile_m = r_profile * (r_mean_m / np.mean(r_profile))
        
        # Dense Core Emulsion Polygon + Outer Sheen
        core_lat_lon = []
        sheen_lat_lon = []
        
        for i in range(num_vertices):
            ang = theta[i]
            # Outer sheen radius
            r_sheen = r_profile_m[i] * 1.18
            dx_sheen = r_sheen * np.sin(ang)
            dy_sheen = r_sheen * np.cos(ang)
            
            plat_s = centroid_lat + (dy_sheen / m_per_deg_lat)
            plon_s = centroid_lon + (dx_sheen / m_per_deg_lon)
            sheen_lat_lon.append([round(float(plat_s), 6), round(float(plon_s), 6)])
            
            # Core emulsion radius
            r_core = r_profile_m[i] * 0.72
            dx_core = r_core * np.sin(ang)
            dy_core = r_core * np.cos(ang)
            
            plat_c = centroid_lat + (dy_core / m_per_deg_lat)
            plon_c = centroid_lon + (dx_core / m_per_deg_lon)
            core_lat_lon.append([round(float(plat_c), 6), round(float(plon_c), 6)])
            
        return {
            "core_polygon_geo": core_lat_lon,
            "sheen_polygon_geo": sheen_lat_lon,
            "characteristic_radius_m": round(float(r_mean_m), 1),
            "elongation_ratio": 2.4,
            "heading_deg": drift_heading_deg
        }

    @staticmethod
    def generate_rewind_cone_projection(
        slick_lat: float,
        slick_lon: float,
        origin_lat: float,
        origin_lon: float,
        origin_radius_m: float,
        spill_age_hours: float,
        num_trajectory_steps: int = 7
    ) -> Dict[str, Any]:
        """
        Constructs the backward Lagrangian trajectory streamline and expanding uncertainty cone
        projecting from the observed SAR slick back to the discharge origin radius.
        """
        m_per_deg_lat = 111000.0
        m_per_deg_lon = 111000.0 * np.cos(np.radians(slick_lat))
        
        # Trajectory streamline points (curved slightly by Coriolis)
        t_vals = np.linspace(0.0, 1.0, num_trajectory_steps)
        streamline_pts = []
        
        # Vector from slick to origin
        d_lat = origin_lat - slick_lat
        d_lon = origin_lon - slick_lon
        
        ortho_lat = -d_lon * 0.15
        ortho_lon = d_lat * 0.15
        
        for t in t_vals:
            curv = 4.0 * t * (1.0 - t)
            p_lat = slick_lat + t * d_lat + curv * ortho_lat
            p_lon = slick_lon + t * d_lon + curv * ortho_lon
            t_hours = round(float(t * spill_age_hours), 1)
            streamline_pts.append({
                "lat": round(float(p_lat), 6),
                "lon": round(float(p_lon), 6),
                "hours_ago": t_hours
            })
            
        # Conical envelope connecting slick bounds to origin circle
        angle_to_origin = np.arctan2(d_lon * m_per_deg_lon, d_lat * m_per_deg_lat)
        perp_angle = angle_to_origin + np.pi / 2.0
        
        # Origin perimeter tangent points
        dx_orig = (origin_radius_m * np.sin(perp_angle)) / m_per_deg_lon
        dy_orig = (origin_radius_m * np.cos(perp_angle)) / m_per_deg_lat
        
        # Slick base tangent points
        slick_base_r_m = 250.0
        dx_slick = (slick_base_r_m * np.sin(perp_angle)) / m_per_deg_lon
        dy_slick = (slick_base_r_m * np.cos(perp_angle)) / m_per_deg_lat
        
        cone_polygon = [
            [round(float(slick_lat + dy_slick), 6), round(float(slick_lon + dx_slick), 6)],
            [round(float(origin_lat + dy_orig), 6), round(float(origin_lon + dx_orig), 6)],
            [round(float(origin_lat - dy_orig), 6), round(float(origin_lon - dx_orig), 6)],
            [round(float(slick_lat - dy_slick), 6), round(float(slick_lon - dx_slick), 6)]
        ]
        
        # Stochastic Monte Carlo particles at origin
        rng = np.random.RandomState(1234)
        particle_cloud = []
        for _ in range(40):
            r_samp = rng.triangular(0.1, 0.6, 1.0) * origin_radius_m
            theta_samp = rng.uniform(0, 2 * np.pi)
            p_dx = (r_samp * np.sin(theta_samp)) / m_per_deg_lon
            p_dy = (r_samp * np.cos(theta_samp)) / m_per_deg_lat
            particle_cloud.append([round(float(origin_lat + p_dy), 6), round(float(origin_lon + p_dx), 6)])
            
        return {
            "streamline_points": streamline_pts,
            "cone_polygon_geo": cone_polygon,
            "particle_cloud": particle_cloud
        }
