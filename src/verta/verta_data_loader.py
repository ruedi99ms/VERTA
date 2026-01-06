# ------------------------------
# Unified Data Loading System
# ------------------------------

from dataclasses import dataclass
from typing import Dict, List, Optional, Type, TypeVar, Union
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from verta.verta_logging import get_logger

T = TypeVar('T', bound='Trajectory')


@dataclass
class ColumnMapping:
    """Centralized column mapping configuration"""
    x: str = "x"
    z: str = "z"
    t: str = "t"
    # VR-specific columns
    headset_x: str = "Headset.Head.Position.X"
    headset_y: str = "Headset.Head.Position.Y"
    headset_z: str = "Headset.Head.Position.Z"
    headset_rot_x: str = "Headset.Head.Rotation.X"
    headset_rot_y: str = "Headset.Head.Rotation.Y"
    headset_rot_z: str = "Headset.Head.Rotation.Z"
    headset_rot_w: str = "Headset.Head.Rotation.W"
    controller_left_x: str = "Controller.Left.Position.X"
    controller_left_y: str = "Controller.Left.Position.Y"
    controller_left_z: str = "Controller.Left.Position.Z"
    controller_right_x: str = "Controller.Right.Position.X"
    controller_right_y: str = "Controller.Right.Position.Y"
    controller_right_z: str = "Controller.Right.Position.Z"
    time: str = "Time"
    # VR headset gaze/physio columns
    head_forward_x: str = "Headset.Head.Forward.X"
    head_forward_y: str = "Headset.Head.Forward.Y"
    head_forward_z: str = "Headset.Head.Forward.Z"
    head_up_x: str = "Headset.Head.Up.X"
    head_up_y: str = "Headset.Head.Up.Y"
    head_up_z: str = "Headset.Head.Up.Z"
    gaze_x: str = "Headset.Gaze.X"
    gaze_y: str = "Headset.Gaze.Y"
    pupil_l: str = "Headset.PupilDilation.L"
    pupil_r: str = "Headset.PupilDilation.R"
    heart_rate: str = "Headset.HeartRate"

    @classmethod
    def vr_defaults(cls) -> 'ColumnMapping':
        """Create VR default column mapping"""
        return cls()

    @classmethod
    def from_dict(cls, columns: Dict[str, str]) -> 'ColumnMapping':
        """Create column mapping from dictionary"""
        return cls(**columns)


@dataclass
class Trajectory:
    """Unified trajectory with optional gaze/physio fields."""
    tid: str
    x: np.ndarray
    z: np.ndarray
    t: Optional[np.ndarray] = None
    # Optional gaze/physio fields (present only when available)
    head_forward_x: Optional[np.ndarray] = None
    head_forward_y: Optional[np.ndarray] = None
    head_forward_z: Optional[np.ndarray] = None
    head_up_x: Optional[np.ndarray] = None
    head_up_y: Optional[np.ndarray] = None
    head_up_z: Optional[np.ndarray] = None
    gaze_x: Optional[np.ndarray] = None
    gaze_y: Optional[np.ndarray] = None
    heart_rate: Optional[np.ndarray] = None
    pupil_l: Optional[np.ndarray] = None
    pupil_r: Optional[np.ndarray] = None

# Backward-compatibility: treat GazeTrajectory as Trajectory
# This alias is deprecated and will be removed in future versions
GazeTrajectory = Trajectory


def has_gaze_data(trajectory: Trajectory) -> bool:
    """Check if trajectory has gaze tracking data."""
    return (trajectory.head_forward_x is not None and
            trajectory.head_forward_z is not None and
            trajectory.gaze_x is not None and
            trajectory.gaze_y is not None)


def has_physio_data(trajectory: Trajectory) -> bool:
    """Check if trajectory has physiological data."""
    return (trajectory.pupil_l is not None and
            trajectory.pupil_r is not None and
            trajectory.heart_rate is not None)


def has_vr_headset_data(trajectory: Trajectory) -> bool:
    """Check if trajectory has VR headset data (head tracking + gaze + physio)."""
    return has_gaze_data(trajectory) and has_physio_data(trajectory)


class TrajectoryLoader:
    """Unified trajectory loading system"""

    def __init__(self, column_mapping: ColumnMapping):
        self.columns = column_mapping
        self.logger = get_logger()

    def _read_table(self, path: str) -> pd.DataFrame:
        """Read table from various formats"""
        ext = os.path.splitext(path)[1].lower()
        if ext in {".csv", ".tsv"}:
            sep = "," if ext == ".csv" else "\t"
            return pd.read_csv(path, sep=sep)
        if ext in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        # Fallback: try CSV
        return pd.read_csv(path)

    def _to_seconds(self, series: pd.Series) -> np.ndarray:
        """Convert time series to seconds"""
        if pd.api.types.is_numeric_dtype(series):
            return series.to_numpy(dtype=float)
        try:
            td = pd.to_timedelta(series)
            return td.dt.total_seconds().to_numpy(dtype=float)
        except Exception:
            return np.full(len(series), np.nan)

    def _extract_coordinates(self, df: pd.DataFrame, mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
        """Extract and scale x,z coordinates"""
        x = df.loc[mask, self.columns.x].to_numpy(dtype=float) * scale
        z = df.loc[mask, self.columns.z].to_numpy(dtype=float) * scale
        return x, z

    def _extract_time(self, df: pd.DataFrame, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract time data if available"""
        if self.columns.t in df.columns:
            return self._to_seconds(df.loc[mask, self.columns.t])
        return None

    def _extract_gaze_data(self, df: pd.DataFrame, mask: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """Extract gaze/head tracking data if available"""
        gaze_data = {}

        gaze_fields = [
            'head_forward_x', 'head_forward_y', 'head_forward_z',
            'head_up_x', 'head_up_y', 'head_up_z',
            'gaze_x', 'gaze_y', 'heart_rate', 'pupil_l', 'pupil_r'
        ]

        # Debug: Print available columns and mappings
        self.logger.debug(f"Available columns: {list(df.columns)}")
        self.logger.debug(f"Column mapping: {self.columns}")

        for field in gaze_fields:
            col_name = getattr(self.columns, field)
            self.logger.debug(f"Field {field} -> Column {col_name}")
            if col_name and col_name in df.columns:
                gaze_data[field] = df.loc[mask, col_name].to_numpy(dtype=float)
                self.logger.debug(f"✅ Extracted {field} from {col_name}: {len(gaze_data[field])} values")
            else:
                gaze_data[field] = None
                self.logger.debug(f"❌ Field {field} not found (column: {col_name})")

        return gaze_data

    def _trim_static_segment(self, x: np.ndarray, z: np.ndarray, t: Optional[np.ndarray],
                           motion_threshold: float) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Trim initial static segment and zero time"""
        if len(x) > 1:
            dd = np.hypot(np.diff(x), np.diff(z))
            idx0 = int(np.argmax(dd > motion_threshold))
            x, z = x[idx0:], z[idx0:]
            if t is not None:
                t = t[idx0:] - t[idx0]
        return x, z, t

    def _trim_gaze_data(self, gaze_data: Dict[str, Optional[np.ndarray]], idx0: int) -> Dict[str, Optional[np.ndarray]]:
        """Trim gaze data arrays to match trimmed trajectory"""
        trimmed = {}
        for field, arr in gaze_data.items():
            if arr is not None:
                trimmed[field] = arr[idx0:]
            else:
                trimmed[field] = None
        return trimmed

    def load_folder(self,
                   folder: str,
                   pattern: str = "*.csv",
                   trajectory_class: Type[T] = Trajectory,
                   require_time: bool = False,
                   scale: float = 1.0,
                   motion_threshold: float = 0.001,
                   progress_callback: Optional[callable] = None) -> List[T]:
        """
        Unified loading logic for any trajectory type

        Args:
            folder: Folder to search
            pattern: Glob pattern for files
            trajectory_class: Class to instantiate trajectories
            require_time: Whether time data is required
            scale: Coordinate scaling factor
            motion_threshold: Motion detection threshold
            progress_callback: Optional callback function for progress updates (current, total, message)

        Returns:
            List of trajectory objects
        """
        paths = sorted(glob.glob(os.path.join(folder, pattern)))
        out: List[T] = []

        self.logger.info(f"Found {len(paths)} files matching '{pattern}'")

        # Use progress callback if provided, otherwise use tqdm
        if progress_callback:
            for i, p in enumerate(paths):
                try:
                    # Update progress
                    progress_callback(i, len(paths), f"Loading file {i+1}/{len(paths)}: {os.path.basename(p)}")

                    df = self._read_table(p)

                    # Check for required coordinate columns
                    coord_cols = [self.columns.x, self.columns.z]
                    missing_cols = [col for col in coord_cols if col not in df.columns]
                    if missing_cols:
                        raise KeyError(f"Missing columns: {missing_cols}")

                    # Create mask for valid coordinates
                    mask = df[coord_cols].notnull().all(axis=1)

                    # Extract coordinates
                    x, z = self._extract_coordinates(df, mask, scale)

                    # Extract time
                    t = self._extract_time(df, mask)

                    # Check time requirements
                    if require_time:
                        if t is None or np.all(np.isnan(t)):
                            continue

                    # Trim static segment
                    x, z, t = self._trim_static_segment(x, z, t, motion_threshold)

                    # Extract gaze/physio data (optional, always attempted)
                    gaze_data = self._extract_gaze_data(df, mask)
                    # Trim gaze data to match trajectory length if we trimmed static segment
                    if len(x) < len(df[mask]):
                        idx0 = len(df[mask]) - len(x)
                        gaze_data = self._trim_gaze_data(gaze_data, idx0)

                    # Create unified Trajectory with optional fields
                    tid = os.path.splitext(os.path.basename(p))[0]
                    trajectory = Trajectory(
                        tid=tid, x=x, z=z, t=t, **gaze_data
                    )

                    out.append(trajectory)

                except Exception as e:
                    self.logger.warning(f"Skip {p}: {e}")
        else:
            # Fallback to tqdm for console usage
            for p in tqdm(paths, desc="Loading trajectories", unit="file"):
                try:
                    df = self._read_table(p)

                    # Check for required coordinate columns
                    coord_cols = [self.columns.x, self.columns.z]
                    missing_cols = [col for col in coord_cols if col not in df.columns]
                    if missing_cols:
                        raise KeyError(f"Missing columns: {missing_cols}")

                    # Create mask for valid coordinates
                    mask = df[coord_cols].notnull().all(axis=1)

                    # Extract coordinates
                    x, z = self._extract_coordinates(df, mask, scale)

                    # Extract time
                    t = self._extract_time(df, mask)

                    # Check time requirements
                    if require_time:
                        if t is None or np.all(np.isnan(t)):
                            continue

                    # Trim static segment
                    x, z, t = self._trim_static_segment(x, z, t, motion_threshold)

                    # Extract gaze/physio data (optional, always attempted)
                    gaze_data = self._extract_gaze_data(df, mask)
                    # Trim gaze data to match trajectory length if we trimmed static segment
                    if len(x) < len(df[mask]):
                        idx0 = len(df[mask]) - len(x)
                        gaze_data = self._trim_gaze_data(gaze_data, idx0)

                    # Create unified Trajectory with optional fields
                    tid = os.path.splitext(os.path.basename(p))[0]
                    trajectory = Trajectory(
                        tid=tid, x=x, z=z, t=t, **gaze_data
                    )

                    out.append(trajectory)

                except Exception as e:
                    self.logger.warning(f"Skip {p}: {e}")

        self.logger.info(f"Loaded {len(out)} trajectories")
        return out


def load_folder(folder: str,
                pattern: str = "*.csv",
                columns: Optional[Dict[str, str]] = None,
                require_time: bool = False,
                require_gaze: bool = False,
                scale: float = 1.0,
                motion_threshold: float = 0.001,
                progress_callback: Optional[callable] = None) -> List[Trajectory]:
    """
    Load trajectories with optional gaze data support.

    Args:
        folder: Folder to search
        pattern: Glob pattern for files
        columns: Column mapping dictionary
        require_time: Whether time data is required
        require_gaze: Whether gaze data is required (legacy parameter for compatibility)
        scale: Coordinate scaling factor
        motion_threshold: Motion detection threshold

    Returns:
        List of Trajectory objects
    """
    if columns is None:
        # Use VR defaults for better compatibility with VR data files
        column_mapping = ColumnMapping.vr_defaults()
    else:
        column_mapping = ColumnMapping.from_dict(columns)

    loader = TrajectoryLoader(column_mapping)
    return loader.load_folder(folder, pattern, Trajectory, require_time, scale, motion_threshold, progress_callback)


def load_folder_with_gaze(folder: str,
                         pattern: str = "*.csv",
                         columns: Optional[Dict[str, str]] = None,
                         require_time: bool = False,
                         scale: float = 1.0,
                         motion_threshold: float = 0.001,
                         progress_callback: Optional[callable] = None) -> List[Trajectory]:
    """
    Load gaze trajectories (backward compatibility).

    DEPRECATED: Use load_folder() with require_gaze=True instead.

    Args:
        folder: Folder to search
        pattern: Glob pattern for files
        columns: Column mapping dictionary
        require_time: Whether time data is required
        scale: Coordinate scaling factor
        motion_threshold: Motion detection threshold

    Returns:
        List of Trajectory objects
    """
    # Delegate to the unified load_folder function
    return load_folder(folder, pattern, columns, require_time, require_gaze=True, scale=scale, motion_threshold=motion_threshold, progress_callback=progress_callback)


# I/O helper functions (moved from ra_io.py for backward compatibility)
def save_centers(centers: np.ndarray, path: str) -> None:
    """Save branch centers to numpy file"""
    np.save(path, centers)


def save_centers_json(centers: np.ndarray, path: str) -> None:
    """Save branch centers to JSON file"""
    import json
    with open(path, "w") as f:
        json.dump(centers.tolist(), f)


def save_assignments(assignments: pd.DataFrame, path: str) -> None:
    """Save branch assignments to CSV"""
    assignments.to_csv(path, index=False)


def save_summary(summary: pd.DataFrame, path: str, with_entropy: bool = True) -> None:
    """Save branch summary with optional entropy"""
    import math
    if with_entropy:
        from .verta_metrics import shannon_entropy

        ent = shannon_entropy(summary)
        summary = summary.copy()
        summary.loc[len(summary)] = {
            "branch": "entropy",
            "count": math.nan,
            "percent": ent,
        }
    summary.to_csv(path, index=False)
