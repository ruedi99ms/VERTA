from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

try:
    from .verta_geometry import Circle
    from .verta_data_loader import Trajectory
except ImportError:  # fallback for direct runs
    from verta.verta_geometry import Circle
    from verta.verta_data_loader import Trajectory


def _build_tid_map(trajectories: Sequence[Trajectory], assignments_df: Optional[pd.DataFrame] = None) -> Dict[str, int]:
    """Map stringified trajectory ids to their sequential indices.

    We normalize everything to integer indices 0..N-1 used by in-memory data,
    while tolerating original string ids in CSVs.
    
    If assignments_df is provided, use the trajectory IDs from the assignments DataFrame
    to create the mapping, ensuring compatibility with the data being processed.
    """
    if assignments_df is not None and "trajectory" in assignments_df.columns:
        # Use trajectory IDs from assignments DataFrame to ensure compatibility
        assignment_ids = assignments_df["trajectory"].unique()
        # Handle mixed numeric and non-numeric trajectory IDs
        assignment_ids_numeric = []
        for x in assignment_ids:
            try:
                assignment_ids_numeric.append(int(x))
            except (ValueError, TypeError):
                # Skip non-numeric IDs like 'outlier_test'
                print(f"[consistency_debug] Skipping non-numeric trajectory ID: {x}")
                continue
        return {str(tid): int(i) for i, tid in enumerate(sorted(assignment_ids_numeric))}
    else:
        # Fallback to original behavior
        return {str(getattr(t, "tid", i)): int(i) for i, t in enumerate(trajectories)}


def normalize_assignments(
    assignments_df: pd.DataFrame,
    *,
    trajectories: Sequence[Trajectory],
    junctions: Sequence[Circle],
    current_junction_idx: Optional[int] = None,
    decisions_df: Optional[pd.DataFrame] = None,
    prefer_decisions: bool = True,
    include_outliers: bool = False,
    strict: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Normalize assignment dataframe for downstream analysis.

    - Harmonizes trajectory ids to in-memory integer indices
    - Ensures presence of branch_j{i} columns
    - Optionally merges decision point columns for the target junction
    - Optionally filters out negative branches (outliers)
    Returns (normalized_df, report)
    """
    df = assignments_df.copy()

    # 1) Map trajectory ids
    tid_map = _build_tid_map(trajectories, assignments_df)
    before_rows = len(df)
    df["trajectory"] = df["trajectory"].astype(str).map(tid_map)
    df = df.dropna(subset=["trajectory"]).copy()
    df["trajectory"] = df["trajectory"].astype(int)
    after_rows = len(df)

    # 2) Ensure branch_j{i} columns exist for each junction
    if len(junctions) == 1:
        # For single junction analysis, use the current_junction_idx if provided, otherwise default to 0
        junction_idx = current_junction_idx if current_junction_idx is not None else 0
        branch_col = f"branch_j{junction_idx}"
        if branch_col not in df.columns:
            if "branch" in df.columns:
                df[branch_col] = df["branch"].astype(float)
            else:
                # create empty if not present; downstream will filter
                df[branch_col] = np.nan
    else:
        for i in range(len(junctions)):
            col = f"branch_j{i}"
            if col not in df.columns and "branch" in df.columns and len(junctions) == 1:
                df[col] = df["branch"].astype(float)

    # 3) Optionally merge decisions for current junction
    if prefer_decisions and decisions_df is not None:
        dec = decisions_df.copy()
        # Accept either consolidated 'junction_index' or assume single junction (0)
        if current_junction_idx is not None and "junction_index" in dec.columns:
            # Ensure both sides of comparison are the same type
            dec["junction_index"] = pd.to_numeric(dec["junction_index"], errors='coerce')
            dec = dec[dec["junction_index"] == int(current_junction_idx)]
        elif "junction_index" in dec.columns and dec["junction_index"].nunique() == 1:
            pass
        # For multi-junction analysis (current_junction_idx=None), keep all decision points
        # Map trajectory ids then merge
        dec = dec.copy()
        dec["trajectory"] = dec["trajectory"].astype(str).map(tid_map)
        dec = dec.dropna(subset=["trajectory"]).copy()
        dec["trajectory"] = dec["trajectory"].astype(int)
        keep_cols = [c for c in ["trajectory", "decision_idx", "intercept_x", "intercept_z"] if c in dec.columns]
        if keep_cols:
            # Ensure both DataFrames have the same trajectory ID data type before merging
            print(f"[consistency_debug] Before merge - df trajectory types: {[type(x) for x in df['trajectory'].unique()[:3]]}")
            print(f"[consistency_debug] Before merge - dec trajectory types: {[type(x) for x in dec['trajectory'].unique()[:3]]}")
            df["trajectory"] = df["trajectory"].astype(int)
            dec["trajectory"] = dec["trajectory"].astype(int)
            print(f"[consistency_debug] After conversion - df trajectory types: {[type(x) for x in df['trajectory'].unique()[:3]]}")
            print(f"[consistency_debug] After conversion - dec trajectory types: {[type(x) for x in dec['trajectory'].unique()[:3]]}")
            df = df.merge(dec[keep_cols], on="trajectory", how="left")

    # 4) Filter out outliers by default for consumers (can be kept if requested)
    if not include_outliers:
        for i in range(len(junctions)):
            col = f"branch_j{i}"
            if col in df.columns:
                df = df[(df[col].notna()) & (df[col] >= 0)]

    # Prepare report
    dropped = before_rows - after_rows
    report = {
        "input_rows": float(before_rows),
        "kept_after_tid_map": float(after_rows),
        "dropped_unmapped_ids": float(dropped),
        "has_decisions": float(1 if ("decision_idx" in df.columns) else 0),
    }

    if strict:
        coverage = after_rows / max(1, before_rows)
        if coverage < 0.95:
            raise ValueError(f"Low ID coverage after mapping: {coverage:.1%}")

    return df, report


def validate_consistency(assignments_df: pd.DataFrame, trajectories: Sequence[Trajectory], junctions: Sequence[Circle]) -> None:
    """Emit assertions/warnings about typical mismatches."""
    # Unique trajectory ids
    tids = [str(getattr(t, "tid", i)) for i, t in enumerate(trajectories)]
    if len(set(tids)) != len(tids):
        print("[consistency] WARNING: Non-unique trajectory IDs in memory; downstream mapping may be ambiguous")

    # Expected columns
    for i in range(len(junctions)):
        col = f"branch_j{i}"
        if col not in assignments_df.columns and not (len(junctions) == 1 and "branch" in assignments_df.columns):
            print(f"[consistency] WARNING: Expected column '{col}' not found in assignments")


def color_index_mapping(gaze_df: pd.DataFrame, junction: Circle) -> Dict[int, int]:
    """Map original branch ids to compact 0..N-1 order by angular direction.

    Keeps physical branch directions visually stable across plots.
    """
    if gaze_df is None or len(gaze_df) == 0 or "branch" not in gaze_df.columns:
        return {}

    # Compute circular mean angle per branch using intercept positions
    unique_branches = sorted(int(b) for b in gaze_df["branch"].dropna().unique())
    branch_angles: Dict[int, float] = {}
    for b in unique_branches:
        data_b = gaze_df[gaze_df["branch"] == b]
        if len(data_b) == 0:
            branch_angles[b] = 0.0
            continue
        vx = (data_b["intercept_x"] - junction.cx).to_numpy()
        vz = (data_b["intercept_z"] - junction.cz).to_numpy()
        ang = np.arctan2(vx, vz)
        mean_sin = float(np.mean(np.sin(ang)))
        mean_cos = float(np.mean(np.cos(ang)))
        branch_angles[b] = float(np.arctan2(mean_sin, mean_cos))

    angle_sorted = [b for b, _ in sorted(branch_angles.items(), key=lambda kv: kv[1])]
    return {b: i for i, b in enumerate(angle_sorted)}


def validate_trajectories_unique(trajectories: Sequence[Trajectory]) -> None:
    tids = [str(getattr(t, "tid", i)) for i, t in enumerate(trajectories)]
    if len(set(tids)) != len(tids):
        print("[consistency] WARNING: Duplicate trajectory IDs detected.")


