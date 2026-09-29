"""
Project DRISHTI - Module 2: Backward Lagrangian Rewind Engine
Runs Monte Carlo backward particle advection with turbulent diffusion Kh
to construct probabilistic discharge origin envelope P_origin(x, y, t).
"""
import numpy as np
from typing import Dict, Any, List
from scipy.stats import gaussian_kde

class LagrangianRewindEngine:
    def __init__(
        self,
        num_particles: int = 1000,
        eddy_diffusivity_kh: float = 5.0, # m^2/s horizontal diffusion
        dt_seconds: float = 300.0         # 5 min time step
    ):
        self.num_particles = num_particles
        self.kh = eddy_diffusivity_kh
        self.dt = dt_seconds

    def run_backward_rewind(
        self,
        slick_centroid_lat: float,
        slick_centroid_lon: float,
        spill_age_hours: float,
        ocean_service,
        stokes_calculator,
        enable_stokes: bool = True,
        wind_perturbation: float = 1.0
    ) -> Dict[str, Any]:
        """
        Advects virtual particles backward in time from t_detect to t_detect - spill_age_hours.
        Returns particle trajectories, final origin centroid, and KDE distribution parameters.
        """
        # Conversion factors for Arabian Sea (~18.9 N)
        # 1 deg lat ~ 110,574 m; 1 deg lon ~ 111,320 * cos(lat) m
        m_per_deg_lat = 110574.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(slick_centroid_lat))
        
        total_steps = int((spill_age_hours * 3600.0) / self.dt)
        
        # Initial particle positions (Gaussian scatter around slick centroid)
        init_scatter_m = 150.0
        p_lat = slick_centroid_lat + np.random.normal(0, init_scatter_m / m_per_deg_lat, self.num_particles)
        p_lon = slick_centroid_lon + np.random.normal(0, init_scatter_m / m_per_deg_lon, self.num_particles)
        
        # Track history snapshots (sample every 10 steps)
        history_snapshots = []
        sample_interval = max(1, total_steps // 10)
        
        for step in range(total_steps):
            t_hours_ago = (step * self.dt) / 3600.0
            
            # Sample drift vector at ensemble mean location
            mean_lat = float(np.mean(p_lat))
            mean_lon = float(np.mean(p_lon))
            drift = ocean_service.compute_total_drift_vector(
                lat=mean_lat,
                lon=mean_lon,
                timestamp_hours_ago=t_hours_ago,
                stokes_calculator=stokes_calculator,
                enable_stokes=enable_stokes,
                wind_perturbation_factor=wind_perturbation
            )
            
            # Backward movement: minus drift * dt
            # Brownian random walk for diffusion: sqrt(2 * Kh * dt) * xi
            diff_std_m = np.sqrt(2.0 * self.kh * self.dt)
            diff_lat_m = np.random.normal(0, diff_std_m, self.num_particles)
            diff_lon_m = np.random.normal(0, diff_std_m, self.num_particles)
            
            d_lat = (-drift["v_total_ms"] * self.dt + diff_lat_m) / m_per_deg_lat
            d_lon = (-drift["u_total_ms"] * self.dt + diff_lon_m) / m_per_deg_lon
            
            p_lat += d_lat
            p_lon += d_lon
            
            if step % sample_interval == 0 or step == total_steps - 1:
                history_snapshots.append({
                    "hours_ago": round(t_hours_ago, 2),
                    "mean_lat": float(np.mean(p_lat)),
                    "mean_lon": float(np.mean(p_lon)),
                    "particles_sample": [
                        [round(float(p_lon[i]), 5), round(float(p_lat[i]), 5)]
                        for i in range(0, min(100, self.num_particles), 2)
                    ]
                })
                
        # Final discharge origin distribution
        origin_lat = float(np.mean(p_lat))
        origin_lon = float(np.mean(p_lon))
        std_lat_m = float(np.std(p_lat) * m_per_deg_lat)
        std_lon_m = float(np.std(p_lon) * m_per_deg_lon)
        origin_radius_m = float(np.hypot(std_lat_m, std_lon_m) * 2.0) # ~95% confidence radius
        
        return {
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "std_lat_m": std_lat_m,
            "std_lon_m": std_lon_m,
            "origin_radius_m": round(origin_radius_m, 1),
            "discharge_time_hours_ago": spill_age_hours,
            "snapshots": history_snapshots,
            "final_particles": [
                [round(float(p_lon[i]), 5), round(float(p_lat[i]), 5)]
                for i in range(0, min(150, self.num_particles))
            ]
        }
