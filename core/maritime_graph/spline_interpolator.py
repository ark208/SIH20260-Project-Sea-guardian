"""
Project DRISHTI - Module 3: Hermite Cubic Spline Track Interpolator
Provides smooth track interpolation for compliant AIS streams.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Dict, Any

class SplineTrackInterpolator:
    def interpolate_track(self, valid_points: List[Dict[str, Any]], num_interp: int = 50) -> List[Dict[str, float]]:
        """Interpolates lat, lon along time using cubic splines."""
        if len(valid_points) < 2:
            return valid_points
            
        times = [p["time_hours_ago"] for p in valid_points]
        lats = [p["lat"] for p in valid_points]
        lons = [p["lon"] for p in valid_points]
        
        # Sort by time ascending
        sorted_indices = np.argsort(times)
        t_arr = np.array(times)[sorted_indices]
        lat_arr = np.array(lats)[sorted_indices]
        lon_arr = np.array(lons)[sorted_indices]
        
        # Handle unique timestamps
        _, unique_idx = np.unique(t_arr, return_index=True)
        t_unique = t_arr[unique_idx]
        lat_unique = lat_arr[unique_idx]
        lon_unique = lon_arr[unique_idx]
        
        if len(t_unique) < 2:
            return valid_points
            
        cs_lat = CubicSpline(t_unique, lat_unique)
        cs_lon = CubicSpline(t_unique, lon_unique)
        
        t_dense = np.linspace(t_unique.min(), t_unique.max(), num_interp)
        lat_dense = cs_lat(t_dense)
        lon_dense = cs_lon(t_dense)
        
        return [
            {"time_hours_ago": round(float(t), 2), "lat": float(lat), "lon": float(lon)}
            for t, lat, lon in zip(t_dense, lat_dense, lon_dense)
        ]
