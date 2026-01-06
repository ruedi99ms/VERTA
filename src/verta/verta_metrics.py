# ------------------------------
# Metrics
# ------------------------------

from typing import Tuple, Optional
import numpy as np
import pandas as pd

try:
    from .verta_geometry import Circle, Rect
    from .verta_data_loader import Trajectory
except ImportError:
    from verta.verta_geometry import Circle, Rect
    from verta.verta_data_loader import Trajectory


def _safe_convert_time_data(t_data: np.ndarray) -> np.ndarray:
    """
    Safely convert time data to numeric format, handling string inputs and various formats.
    
    Args:
        t_data: Time data array (may be strings or numeric)
        
    Returns:
        Numeric time data array, or None if conversion fails
    """
    if t_data is None:
        return None
    
    try:
        # If already numeric, return as is
        if t_data.dtype.kind in ['i', 'f']:  # Integer or float
            return t_data
        
        # Try to convert string/object data to numeric
        if t_data.dtype.kind in ['U', 'S', 'O']:  # String or object type
            # First, try direct conversion
            t_numeric = pd.to_numeric(t_data, errors='coerce')
            
            # Check if conversion was successful
            if np.all(np.isnan(t_numeric)):
                # Try timestamp format (HH:MM:SS.mmm)
                t_numeric = _convert_timestamp_format(t_data)
                if t_numeric is not None:
                    return t_numeric
                
                # Try generic string cleaning
                t_numeric = _convert_generic_string_format(t_data)
                if t_numeric is not None:
                    return t_numeric
            
            return t_numeric
        
        return t_data
    except (ValueError, TypeError, AttributeError):
        return None


def _convert_timestamp_format(t_data: np.ndarray) -> Optional[np.ndarray]:
    """Convert timestamp format (HH:MM:SS.mmm) to seconds."""
    import re
    timestamp_pattern = re.compile(r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})')
    
    try:
        cleaned = []
        for val in t_data:
            if pd.isna(val) or val == '':
                cleaned.append(np.nan)
                continue
            
            val_str = str(val).strip()
            match = timestamp_pattern.match(val_str)
            if match:
                hours, minutes, seconds, milliseconds = match.groups()
                total_seconds = float(hours) * 3600 + float(minutes) * 60 + float(seconds) + float(milliseconds) / 1000
                cleaned.append(total_seconds)
            else:
                cleaned.append(np.nan)
        
        t_numeric = np.array(cleaned)
        return t_numeric if np.any(~np.isnan(t_numeric)) else None
    except (ValueError, TypeError):
        return None


def _convert_generic_string_format(t_data: np.ndarray) -> Optional[np.ndarray]:
    """Convert generic string format to numeric."""
    import re
    
    try:
        t_cleaned = []
        for val in t_data:
            if isinstance(val, str):
                # Remove common prefixes/suffixes and quotes
                cleaned = val.strip().strip('"\'')
                # Try to extract number from strings like "Time: 1.23" or "1.23s"
                match = re.search(r'(\d+\.?\d*)', cleaned)
                if match:
                    cleaned = match.group(1)
                t_cleaned.append(cleaned)
            else:
                t_cleaned.append(val)
        
        # Try conversion again
        t_numeric = pd.to_numeric(t_cleaned, errors='coerce')
        return t_numeric if np.any(~np.isnan(t_numeric)) else None
    except (ValueError, TypeError):
        return None


def time_to_distance_after_junction(tr: Trajectory,
                                    junction: Circle,
                                    path_length: float) -> float:
    """Time (seconds) from first entering `junction` until cumulative traveled distance along the path reaches `path_length`. Returns NaN if no time data or cannot be computed.
    """
    if tr.t is None:
        return float("nan")
    
    # Safely convert time data to numeric
    t_numeric = _safe_convert_time_data(tr.t)
    if t_numeric is None:
        return float("nan")
    
    dist = np.hypot(tr.x - junction.cx, tr.z - junction.cz)
    inside = dist <= junction.r
    if inside.any():
        start = int(np.argmax(inside))
    else:
        start = int(np.argmin(dist))
    dx = np.diff(tr.x[start:])
    dz = np.diff(tr.z[start:])
    seg = np.hypot(dx, dz)
    if len(seg) == 0:
        return float("nan")
    cum = np.cumsum(seg)
    mask = cum >= path_length
    if not mask.any():
        return float("nan")
    reach = int(np.argmax(mask))
    return float(t_numeric[start + reach] - t_numeric[start])

def time_from_junction_to_radial_exit(
    tr: Trajectory,
    junction: Circle,
    r_outer: float,
    window: int = 5,
    min_outward: float = 0.0,
) -> float:
    """
    Time (seconds) from first entering `junction` until the trajectory crosses r_outer with a non-negative (or >= min_outward) outward trend. NaN if no time or no crossing.
    """
    if tr.t is None:
        return float("nan")

    # Safely convert time data to numeric
    t_numeric = _safe_convert_time_data(tr.t)
    if t_numeric is None:
        return float("nan")

    # reuse your existing index finder but with a configurable outward threshold
    r = np.hypot(tr.x - junction.cx, tr.z - junction.cz)
    inside = r <= junction.r
    start = int(np.argmax(inside)) if inside.any() else int(np.argmin(r))

    i_cross = None
    for i in range(start + 1, len(r)):
        if r[i] >= r_outer:
            j0 = max(start + 1, i - window)
            seg = r[j0:i+1]
            if seg.size < 2 or float(np.nanmean(np.diff(seg))) >= float(min_outward):
                i_cross = i
                break

    if i_cross is None:
        return float("nan")
    return float(t_numeric[i_cross] - t_numeric[start])

def time_between_regions(tr: Trajectory,
                         A: Rect | Circle,
                         B: Rect | Circle) -> Tuple[float, float, float]:
    """Return (t_A, t_B, dt) where t_A is the first timestamp when trajectory is in region A, t_B for region B, and dt = t_B - t_A. Returns (nan, nan, nan) if timestamps missing or sequence invalid.
    """
    if tr.t is None:
        return (float("nan"), float("nan"), float("nan"))
    
    # Safely convert time data to numeric
    t_numeric = _safe_convert_time_data(tr.t)
    if t_numeric is None:
        return (float("nan"), float("nan"), float("nan"))
    
    inA = A.contains(tr.x, tr.z)
    inB = B.contains(tr.x, tr.z)
    iA = int(np.argmax(inA)) if inA.any() else None
    iB = int(np.argmax(inB)) if inB.any() else None
    if iA is not None and iB is not None and iB > iA:
        return float(t_numeric[iA]), float(t_numeric[iB]), float(t_numeric[iB] - t_numeric[iA])
    return (float("nan"), float("nan"), float("nan"))

def shannon_entropy(summary: pd.DataFrame) -> float:
    """Compute entropy (nats) from a branch summary with a 'percent' column."""
    if len(summary) == 0:
        return float("nan")
    p = summary["percent"].to_numpy(dtype=float) / 100.0
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))

def compute_basic_trajectory_metrics(tr: Trajectory) -> dict:
    """
    Compute basic trajectory metrics: total time, total distance, and average speed.
    
    Args:
        tr: Trajectory object
        
    Returns:
        dict with keys: total_time, total_distance, average_speed
    """
    metrics = {
        "total_time": 0.0,
        "total_distance": 0.0,
        "average_speed": 0.0
    }
    
    # Compute total distance
    if len(tr.x) > 1:
        # Filter out NaN values
        valid_mask = ~(np.isnan(tr.x) | np.isnan(tr.z))
        if not valid_mask.any():
            # All values are NaN
            metrics["total_distance"] = 0.0
        else:
            # Use only valid (non-NaN) points
            x_valid = tr.x[valid_mask]
            z_valid = tr.z[valid_mask]
            
            if len(x_valid) > 1:
                dx = np.diff(x_valid)
                dz = np.diff(z_valid)
                segments = np.hypot(dx, dz)
                total_distance = float(np.sum(segments))
                metrics["total_distance"] = total_distance
            else:
                metrics["total_distance"] = 0.0
    
    # Compute total time and average speed
    if tr.t is not None and len(tr.t) > 1:
        # Safely convert time data to numeric
        t_numeric = _safe_convert_time_data(tr.t)
        if t_numeric is not None:
            total_time = float(t_numeric[-1] - t_numeric[0])
            metrics["total_time"] = total_time
            
            if total_time > 0 and metrics["total_distance"] > 0:
                metrics["average_speed"] = metrics["total_distance"] / total_time
    
    return metrics


def speed_through_junction(tr: Trajectory,
                          junction: Circle,
                          decision_mode: str = "pathlen",
                          path_length: float = 100.0,
                          r_outer: Optional[float] = None,
                          window: int = 5,
                          min_outward: float = 0.0) -> Tuple[float, str]:
    """
    Calculate average speed through junction based on decision mode.
    
    Args:
        tr: Trajectory object
        junction: Junction circle
        decision_mode: "pathlen", "radial", or "hybrid"
        path_length: Path length for pathlen mode
        r_outer: Outer radius for radial mode
        window: Window size for radial mode
        min_outward: Minimum outward movement for radial mode
        
    Returns:
        Tuple of (average_speed, mode_used)
    """
    if tr.t is None:
        return (float("nan"), "no_time_data")
    
    # Safely convert time data to numeric
    t_numeric = _safe_convert_time_data(tr.t)
    if t_numeric is None:
        return (float("nan"), "no_time_data")
    
    # Find junction entry point
    dist = np.hypot(tr.x - junction.cx, tr.z - junction.cz)
    inside = dist <= junction.r
    if inside.any():
        start_idx = int(np.argmax(inside))
    else:
        start_idx = int(np.argmin(dist))
    
    # Determine end point based on decision mode
    end_idx = None
    mode_used = decision_mode
    
    if decision_mode == "pathlen":
        # Calculate cumulative distance from start
        dx = np.diff(tr.x[start_idx:])
        dz = np.diff(tr.z[start_idx:])
        seg = np.hypot(dx, dz)
        if len(seg) == 0:
            return (float("nan"), "no_movement")
        
        cum = np.cumsum(seg)
        mask = cum >= path_length
        if mask.any():
            end_idx = start_idx + int(np.argmax(mask)) + 1
        else:
            return (float("nan"), "pathlen_not_reached")
    
    elif decision_mode == "radial":
        # Find radial exit point
        rout = r_outer if (r_outer is not None and r_outer > junction.r) else (junction.r + 10.0)
        
        for i in range(start_idx + 1, len(dist)):
            if dist[i] >= rout:
                j0 = max(start_idx + 1, i - window)
                seg = dist[j0:i+1]
                if seg.size >= 2 and float(np.nanmean(np.diff(seg))) >= min_outward:
                    end_idx = i
                    break
        
        if end_idx is None:
            return (float("nan"), "radial_not_reached")
    
    elif decision_mode == "hybrid":
        # Try radial first, fall back to pathlen
        rout = r_outer if (r_outer is not None and r_outer > junction.r) else (junction.r + 10.0)
        
        # Try radial
        for i in range(start_idx + 1, len(dist)):
            if dist[i] >= rout:
                j0 = max(start_idx + 1, i - window)
                seg = dist[j0:i+1]
                if seg.size >= 2 and float(np.nanmean(np.diff(seg))) >= min_outward:
                    end_idx = i
                    mode_used = "radial"
                    break
        
        # Fall back to pathlen if radial failed
        if end_idx is None:
            dx = np.diff(tr.x[start_idx:])
            dz = np.diff(tr.z[start_idx:])
            seg = np.hypot(dx, dz)
            if len(seg) > 0:
                cum = np.cumsum(seg)
                mask = cum >= path_length
                if mask.any():
                    end_idx = start_idx + int(np.argmax(mask)) + 1
                    mode_used = "pathlen"
                else:
                    return (float("nan"), "hybrid_not_reached")
            else:
                return (float("nan"), "hybrid_no_movement")
    
    # Calculate speed if we have valid start and end points
    if end_idx is not None and end_idx > start_idx:
        # Calculate distance traveled
        x_segment = tr.x[start_idx:end_idx+1]
        z_segment = tr.z[start_idx:end_idx+1]
        
        if len(x_segment) > 1:
            dx = np.diff(x_segment)
            dz = np.diff(z_segment)
            segments = np.hypot(dx, dz)
            total_distance = float(np.sum(segments))
            
            # Calculate time elapsed
            total_time = float(t_numeric[end_idx] - t_numeric[start_idx])
            
            if total_time > 0:
                average_speed = total_distance / total_time
                return (average_speed, mode_used)
            else:
                return (float("nan"), f"{mode_used}_zero_time")
        else:
            return (float("nan"), f"{mode_used}_single_point")
    
    return (float("nan"), f"{mode_used}_invalid")


def junction_transit_speed(tr: Trajectory,
                          junction: Circle) -> Tuple[float, float, float]:
    """
    Calculate speed metrics for junction transit: entry speed, exit speed, and average speed.
    
    Args:
        tr: Trajectory object
        junction: Junction circle
        
    Returns:
        Tuple of (entry_speed, exit_speed, average_speed)
    """
    if tr.t is None or len(tr.x) < 3:
        return (float("nan"), float("nan"), float("nan"))
    
    # Safely convert time data to numeric
    t_numeric = _safe_convert_time_data(tr.t)
    if t_numeric is None:
        return (float("nan"), float("nan"), float("nan"))
    
    # Find junction entry and exit points
    dist = np.hypot(tr.x - junction.cx, tr.z - junction.cz)
    inside = dist <= junction.r
    
    if not inside.any():
        return (float("nan"), float("nan"), float("nan"))
    
    # Find entry point (first time inside)
    entry_idx = int(np.argmax(inside))
    
    # Find exit point (last time inside)
    exit_idx = len(inside) - 1 - int(np.argmax(inside[::-1]))
    
    if exit_idx <= entry_idx:
        return (float("nan"), float("nan"), float("nan"))
    
    # Calculate entry speed (average speed in 2-5 second window before entering)
    entry_speed = float("nan")
    if entry_idx > 0:
        # Find start of 2-5 second window before entry
        entry_time = t_numeric[entry_idx]
        window_start_time = entry_time - 5.0  # 5 seconds before
        window_end_time = entry_time - 2.0   # 2 seconds before
        
        # Find indices for the time window
        window_mask = (t_numeric >= window_start_time) & (t_numeric <= window_end_time)
        window_indices = np.where(window_mask)[0]
        
        if len(window_indices) > 1:
            # Calculate average speed in this window
            x_window = tr.x[window_indices]
            z_window = tr.z[window_indices]
            t_window = t_numeric[window_indices]
            
            if len(x_window) > 1:
                dx = np.diff(x_window)
                dz = np.diff(z_window)
                segments = np.hypot(dx, dz)
                total_distance = float(np.sum(segments))
                total_time = float(t_window[-1] - t_window[0])
                
                if total_time > 0:
                    entry_speed = total_distance / total_time
    
    # Calculate exit speed (average speed in 2-5 second window after exiting)
    exit_speed = float("nan")
    if exit_idx < len(tr.x) - 1:
        # Find start of 2-5 second window after exit
        exit_time = t_numeric[exit_idx]
        window_start_time = exit_time + 2.0   # 2 seconds after
        window_end_time = exit_time + 5.0     # 5 seconds after
        
        # Find indices for the time window
        window_mask = (t_numeric >= window_start_time) & (t_numeric <= window_end_time)
        window_indices = np.where(window_mask)[0]
        
        if len(window_indices) > 1:
            # Calculate average speed in this window
            x_window = tr.x[window_indices]
            z_window = tr.z[window_indices]
            t_window = t_numeric[window_indices]
            
            if len(x_window) > 1:
                dx = np.diff(x_window)
                dz = np.diff(z_window)
                segments = np.hypot(dx, dz)
                total_distance = float(np.sum(segments))
                total_time = float(t_window[-1] - t_window[0])
                
                if total_time > 0:
                    exit_speed = total_distance / total_time
    
    # Calculate average speed through junction
    average_speed = float("nan")
    if exit_idx > entry_idx:
        # Calculate total distance through junction
        x_segment = tr.x[entry_idx:exit_idx+1]
        z_segment = tr.z[entry_idx:exit_idx+1]
        
        if len(x_segment) > 1:
            dx = np.diff(x_segment)
            dz = np.diff(z_segment)
            segments = np.hypot(dx, dz)
            total_distance = float(np.sum(segments))
            
            # Calculate total time through junction
            total_time = t_numeric[exit_idx] - t_numeric[entry_idx]
            
            if total_time > 0:
                average_speed = total_distance / total_time
    
    return (entry_speed, exit_speed, average_speed)


def _timing_for_traj(
    tr: Trajectory,
    junction: Circle,
    decision_mode: str,
    distance: float,
    r_outer: float | None,
    trend_window: int,
    min_outward: float,
) -> tuple[float, str]:
    """
    Returns (time_value_seconds, mode_used).
    - pathlen: junction entry -> reach `distance` of walked path
    - radial:  junction entry -> first intercept of r_outer with outward trend
    - hybrid:  try radial first, fall back to pathlen
    """
    if decision_mode == "pathlen":
        return (
            time_to_distance_after_junction(tr, junction, path_length=float(distance)),
            "pathlen",
        )

    # radial (or hybrid attempt): need a sensible outer radius
    rout = None
    if r_outer is not None and float(r_outer) > float(junction.r):
        rout = float(r_outer)
    else:
        # soft default if user forgot to pass r_outer
        rout = float(junction.r) + 10.0

    t_rad = time_from_junction_to_radial_exit(
        tr,
        junction,
        r_outer=rout,
        window=int(trend_window),
        min_outward=float(min_outward),
    )
    if decision_mode == "radial":
        return (t_rad, "radial")

    # hybrid: radial if it worked, else pathlen
    if not (t_rad is None or (isinstance(t_rad, float) and (t_rad != t_rad))):  # not NaN
        return (t_rad, "radial")
    return (
        time_to_distance_after_junction(tr, junction, path_length=float(distance)),
        "pathlen",
    )