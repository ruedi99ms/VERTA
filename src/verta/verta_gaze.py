# Gaze and Head Tracking Analysis

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .verta_geometry import Circle
    from .verta_data_loader import Trajectory
    from .verta_decisions import get_decision_index
except ImportError:
    from verta.verta_geometry import Circle
    from verta.verta_data_loader import Trajectory
    from verta.verta_decisions import get_decision_index



def compute_head_yaw_at_decisions(
    trajectories: Sequence[Trajectory],
    junctions: Sequence[Circle],
    assignments_df: pd.DataFrame,
    decision_mode: str = "hybrid",
    r_outer_list: Optional[Sequence[float]] = None,
    path_length: float = 100.0,
    epsilon: float = 0.05,
    linger_delta: float = 0.0,
    base_index: int = 0,
    decisions_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute head yaw angles at decision points for each trajectory."""
    from tqdm import tqdm
    
    if r_outer_list is None:
        r_outer_list = [None] * len(junctions)
    
    results = []
    
    def _nearest_valid_index(
        tr: Trajectory,
        start_idx: Optional[int],
        max_window: int = 200,
    ) -> Optional[int]:
        """Return nearest index to start_idx with valid x,z and head_forward data.

        Searches outward from start_idx up to max_window samples in both directions.
        """
        if start_idx is None:
            return None
        length = min(len(tr.x), len(tr.z))
        hf_len = min(
            len(tr.head_forward_x) if tr.head_forward_x is not None else 0,
            len(tr.head_forward_z) if tr.head_forward_z is not None else 0,
        )
        if length == 0 or hf_len == 0:
            return None
        # Clamp start index
        start_idx = max(0, min(start_idx, length - 1, hf_len - 1))

        def is_valid(i: int) -> bool:
            if i < 0 or i >= length or i >= hf_len:
                return False
            x_ok = not np.isnan(tr.x[i])
            z_ok = not np.isnan(tr.z[i])
            hf_x_ok = tr.head_forward_x is not None and not np.isnan(tr.head_forward_x[i])
            hf_z_ok = tr.head_forward_z is not None and not np.isnan(tr.head_forward_z[i])
            return x_ok and z_ok and hf_x_ok and hf_z_ok

        if is_valid(start_idx):
            return start_idx

        for delta in range(1, max_window + 1):
            left = start_idx - delta
            right = start_idx + delta
            if is_valid(left):
                return left
            if is_valid(right):
                return right
        return None
    
    print(f"[gaze] Computing head directions for {len(trajectories)} trajectories...")
    
    for tr in tqdm(trajectories, desc="Computing head directions", unit="traj"):
        # Get trajectory assignments for all junctions
        traj_assignments = assignments_df[assignments_df["trajectory"] == tr.tid]
        
        if traj_assignments.empty:
            continue
            
        for i, junc in enumerate(junctions):
            # CRITICAL FIX: Use the actual junction index (base_index + i) for branch column lookup
            # This ensures we look in the correct branch_j{X} column for each junction
            label_idx = base_index + i
            label_str = f"Junction {label_idx} ({junc.cx}, {junc.cz}, r={junc.r})"
            
            # Use the actual junction index for branch column to ensure correct mapping
            branch_col = f"branch_j{label_idx}"
            branch = traj_assignments[branch_col].iloc[0] if branch_col in traj_assignments.columns else None
            
            # Debug: Show branch assignment details
            print(f"[gaze_debug] Junction {label_idx}: branch_col={branch_col}, branch={branch}")
            
            # If branch_col not found, try the generic 'branch' column (for single-junction analysis)
            if branch is None and "branch" in traj_assignments.columns:
                branch = traj_assignments["branch"].iloc[0]
                print(f"[gaze_debug] Junction {label_idx}: Using generic 'branch' column, branch={branch}")
            
            if branch is None or (isinstance(branch, float) and np.isnan(branch)) or int(branch) < 0:
                continue
                
            # Prefer precomputed decisions if provided via assignments_df.
            # Look for decision points specific to this trajectory and junction
            pre_idx = None
            pre_x = np.nan
            pre_z = np.nan
            
            # First try junction-specific columns (e.g., decision_idx_j0, intercept_x_j0, etc.)
            junction_cols = {
                'decision_idx': f"decision_idx_j{label_idx}",
                'intercept_x': f"intercept_x_j{label_idx}",
                'intercept_z': f"intercept_z_j{label_idx}"
            }
            
            # Debug: Check what junction-specific columns are available
            available_junction_cols = [col for col in traj_assignments.columns if col.startswith('decision_idx_j')]
            if tr.tid < 5:  # Only debug first 5 trajectories to avoid spam
                print(f"[gaze_debug] Trajectory {tr.tid} at junction {label_idx}: Available junction columns: {available_junction_cols}")
                print(f"[gaze_debug] Trajectory {tr.tid} at junction {label_idx}: Looking for: {list(junction_cols.values())}")
            
            if all(col in traj_assignments.columns for col in junction_cols.values()):
                # Use junction-specific columns
                traj_assignments_for_traj = traj_assignments[traj_assignments["trajectory"] == tr.tid]
                if not traj_assignments_for_traj.empty:
                    decision_idx_val = traj_assignments_for_traj[junction_cols['decision_idx']].iloc[0]
                    intercept_x_val = traj_assignments_for_traj[junction_cols['intercept_x']].iloc[0]
                    intercept_z_val = traj_assignments_for_traj[junction_cols['intercept_z']].iloc[0]
                    
                    if not (isinstance(decision_idx_val, float) and np.isnan(decision_idx_val)):
                        pre_idx = int(decision_idx_val)
                    if not (isinstance(intercept_x_val, float) and np.isnan(intercept_x_val)):
                        pre_x = float(intercept_x_val)
                    if not (isinstance(intercept_z_val, float) and np.isnan(intercept_z_val)):
                        pre_z = float(intercept_z_val)
                        
                if pre_idx is not None and not np.isnan(pre_x) and not np.isnan(pre_z):
                    print(f"[gaze_debug] Using precomputed decision point for trajectory {tr.tid} at junction {label_idx}: idx={pre_idx}, x={pre_x}, z={pre_z}")
            elif "decision_idx" in traj_assignments.columns:
                # Filter to this specific trajectory
                print(f"[gaze_debug] Looking for trajectory {tr.tid} in assignments with trajectory IDs: {traj_assignments['trajectory'].unique()[:10]}")
                print(f"[gaze_debug] Assignments DataFrame shape: {traj_assignments.shape}")
                print(f"[gaze_debug] Assignments DataFrame columns: {list(traj_assignments.columns)}")
                print(f"[gaze_debug] Sample assignments data: {traj_assignments[['trajectory', 'decision_idx', 'intercept_x', 'intercept_z']].head(3).to_dict('records')}")
                
                traj_assignments_for_traj = traj_assignments[traj_assignments["trajectory"] == tr.tid]
                print(f"[gaze_debug] Found {len(traj_assignments_for_traj)} assignments for trajectory {tr.tid}")
                if not traj_assignments_for_traj.empty:
                    # Get the decision point for this trajectory
                    if "decision_idx" in traj_assignments_for_traj.columns:
                        decision_idx_val = traj_assignments_for_traj["decision_idx"].iloc[0]
                        if not (isinstance(decision_idx_val, float) and np.isnan(decision_idx_val)):
                            pre_idx = int(decision_idx_val)
                    if "intercept_x" in traj_assignments_for_traj.columns:
                        intercept_x_val = traj_assignments_for_traj["intercept_x"].iloc[0]
                        if not (isinstance(intercept_x_val, float) and np.isnan(intercept_x_val)):
                            pre_x = float(intercept_x_val)
                    if "intercept_z" in traj_assignments_for_traj.columns:
                        intercept_z_val = traj_assignments_for_traj["intercept_z"].iloc[0]
                        if not (isinstance(intercept_z_val, float) and np.isnan(intercept_z_val)):
                            pre_z = float(intercept_z_val)

            if pre_idx is not None and not np.isnan(pre_x) and not np.isnan(pre_z):
                idx = pre_idx
                method_used = "precomputed"
                print(f"[gaze_debug] Using precomputed decision point for trajectory {tr.tid}: idx={pre_idx}, x={pre_x}, z={pre_z}")
            else:
                print(f"[gaze_debug] No precomputed decision point for trajectory {tr.tid}: pre_idx={pre_idx}, pre_x={pre_x}, pre_z={pre_z}")
                # Find decision intercept using the same logic as discover analysis
                # This ensures arrows are plotted at the exact same points used for branch assignment
                r_out = r_outer_list[i]
                
                # Use the same decision point calculation with identical params
                if decision_mode == "radial" or (decision_mode == "hybrid" and r_out is not None and float(r_out) > float(junc.r)):
                    idx = get_decision_index(
                        tr.x, tr.z, junc, 
                        decision_mode="radial",
                        r_outer=r_out if r_out is not None else (junc.r + 10.0),
                        window=5
                    )
                    method_used = "radial"
                else:
                    idx = get_decision_index(
                        tr.x, tr.z, junc,
                        decision_mode="pathlen", 
                        path_length=path_length,
                        epsilon=epsilon,
                        linger_delta=linger_delta
                    )
                    method_used = "pathlen"
            
            # Debug: Track decision point calculation success
            if idx is None:
                print(f"[decision_debug] FAILED to find decision point for trajectory {tr.tid} at {label_str}")
                print(f"[decision_debug] Method attempted: {method_used}")
                print(f"[decision_debug] Junction: cx={junc.cx}, cz={junc.cz}, r={junc.r}")
                print(f"[decision_debug] R_outer: {r_out}, path_length: {path_length}")
                print(f"[decision_debug] Trajectory bounds: x=[{np.min(tr.x):.1f}, {np.max(tr.x):.1f}], z=[{np.min(tr.z):.1f}, {np.max(tr.z):.1f}]")
                
                # Don't use fallback for trajectories without proper decision points
                # These trajectories should not be assigned to any branch
                print(f"[decision_debug] NO FALLBACK: Trajectory {tr.tid} has no proper decision point - skipping")
                continue
            else:
                print(f"[decision_debug] SUCCESS: Found decision point for trajectory {tr.tid} at {label_str} using {method_used}")
                print(f"[decision_debug] Decision point: x={tr.x[idx]:.1f}, z={tr.z[idx]:.1f}")
                print(f"[decision_debug] Distance from junction center: {np.hypot(tr.x[idx] - junc.cx, tr.z[idx] - junc.cz):.1f}")
            
            # Debug: Check decision index calculation
            if tr.tid in ['2', '3', '4']:  # Debug first few trajectories
                print(f"[head_yaw_debug] Trajectory {tr.tid} at {label_str}: idx={idx}, decision_mode={decision_mode}")
                print(f"[head_yaw_debug] Junction: cx={junc.cx}, cz={junc.cz}, r={junc.r}")
                print(f"[head_yaw_debug] R_outer: {r_out}, path_length: {path_length}")
                print(f"[head_yaw_debug] Trajectory length: x={len(tr.x)}, z={len(tr.z)}")
                print(f"[head_yaw_debug] Head forward length: x={len(tr.head_forward_x) if tr.head_forward_x is not None else 'None'}, z={len(tr.head_forward_z) if tr.head_forward_z is not None else 'None'}")
                if idx is not None:
                    print(f"[head_yaw_debug] Index bounds check: idx < len(x) = {idx < len(tr.x)}, idx < len(head_forward_x) = {idx < len(tr.head_forward_x) if tr.head_forward_x is not None else 'N/A'}")
                    if idx < len(tr.x):
                        print(f"[head_yaw_debug] Decision point: x={tr.x[idx]}, z={tr.z[idx]}")
                    else:
                        print(f"[head_yaw_debug] Decision index {idx} is out of bounds for trajectory length {len(tr.x)}!")
                    print(f"[head_yaw_debug] Head forward data available: {tr.head_forward_x is not None and tr.head_forward_z is not None}")
                    if tr.head_forward_x is not None and tr.head_forward_z is not None and idx < len(tr.head_forward_x):
                        print(f"[head_yaw_debug] Head forward at idx: x={tr.head_forward_x[idx]}, z={tr.head_forward_z[idx]}")
                    else:
                        print(f"[head_yaw_debug] Head forward index {idx} is out of bounds for head_forward length {len(tr.head_forward_x) if tr.head_forward_x is not None else 'None'} at {label_str}!")
                else:
                    print(f"[head_yaw_debug] No decision index found at {label_str}!")
            
            # Choose nearest valid index around decision point for head-forward data
            # Keep the intercept position anchored at the geometric decision index
            # (or its boundary fallback), even if head-forward is taken from nearby.
            valid_idx = _nearest_valid_index(tr, idx)
            intercept_idx = idx
            
            # Debug: Show valid index adjustment
            if tr.tid in ['2', '3', '4']:  # Debug first few trajectories
                print(f"[head_yaw_debug] Valid index adjustment: idx={idx} -> valid_idx={valid_idx}")
                if valid_idx is not None:
                    print(f"[head_yaw_debug] Valid decision point: x={tr.x[valid_idx]}, z={tr.z[valid_idx]}")
                    print(f"[head_yaw_debug] Valid head forward: x={tr.head_forward_x[valid_idx]}, z={tr.head_forward_z[valid_idx]}")
            
            # Compute head yaw at decision point (VR convention: Z forward, X right)
            if (
                tr.head_forward_x is not None
                and tr.head_forward_z is not None
                and valid_idx is not None
                and valid_idx < len(tr.head_forward_x)
                and valid_idx < len(tr.head_forward_z)
            ):
                head_forward_x = tr.head_forward_x[valid_idx]
                head_forward_z = tr.head_forward_z[valid_idx]
                if not (np.isnan(head_forward_x) or np.isnan(head_forward_z)):
                    head_yaw = np.degrees(np.arctan2(head_forward_x, head_forward_z))
                    # Debug: Show calculated head_yaw
                    if tr.tid in ['2', '3', '4']:  # Debug first few trajectories
                        print(f"[head_yaw_debug] Calculated head_yaw: {head_yaw:.2f}° for trajectory {tr.tid}, branch {int(branch)}")
                else:
                    head_yaw = np.nan
                    if tr.tid in ['2', '3', '4']:  # Debug first few trajectories
                        print(f"[head_yaw_debug] Head forward data is NaN for trajectory {tr.tid}, branch {int(branch)}")
            else:
                head_yaw = np.nan
                if tr.tid in ['2', '3', '4']:  # Debug first few trajectories
                    print(f"[head_yaw_debug] Cannot calculate head_yaw for trajectory {tr.tid}, branch {int(branch)} - missing data or invalid index")
            
            # Movement direction at decision point
            if (
                valid_idx is not None
                and valid_idx > 0
                and valid_idx < len(tr.x) - 1
                and valid_idx < len(tr.z) - 1
            ):
                # Use velocity direction
                dx = tr.x[valid_idx + 1] - tr.x[valid_idx - 1]
                dz = tr.z[valid_idx + 1] - tr.z[valid_idx - 1]
                # Check for NaN values in trajectory coordinates
                if not (np.isnan(dx) or np.isnan(dz)) and np.hypot(dx, dz) > 1e-6:
                    movement_yaw = np.degrees(np.arctan2(dx, dz))
                else:
                    movement_yaw = np.nan
            else:
                movement_yaw = np.nan
            
            # Gaze-movement alignment
            if not np.isnan(movement_yaw) and not np.isnan(head_yaw):
                yaw_diff = ((head_yaw - movement_yaw + 180) % 360) - 180  # [-180, 180]
            else:
                yaw_diff = np.nan
                
            # Intercept position (use precomputed if available; else use decision index)
            if pre_idx is not None and not np.isnan(pre_x) and not np.isnan(pre_z):
                intercept_x = float(pre_x)
                intercept_z = float(pre_z)
            elif (
                intercept_idx is not None
                and intercept_idx < len(tr.x)
                and intercept_idx < len(tr.z)
                and not np.isnan(tr.x[intercept_idx])
                and not np.isnan(tr.z[intercept_idx])
            ):
                intercept_x = tr.x[intercept_idx]
                intercept_z = tr.z[intercept_idx]
            elif (
                valid_idx is not None
                and valid_idx < len(tr.x)
                and valid_idx < len(tr.z)
                and not np.isnan(tr.x[valid_idx])
                and not np.isnan(tr.z[valid_idx])
            ):
                # Fallback to nearest valid sample if the decision index is unusable
                intercept_x = tr.x[valid_idx]
                intercept_z = tr.z[valid_idx]
            else:
                intercept_x = np.nan
                intercept_z = np.nan
                
            results.append({
                "trajectory": tr.tid,
                "junction": label_idx,  # Use label_idx instead of i to match the actual junction index
                "branch": int(branch),
                "head_yaw": head_yaw,
                "movement_yaw": movement_yaw,
                "yaw_difference": yaw_diff,
                "intercept_x": intercept_x,
                "intercept_z": intercept_z,
                "gaze_x": tr.gaze_x[valid_idx] if tr.gaze_x is not None and valid_idx is not None and valid_idx < len(tr.gaze_x) else np.nan,
                "gaze_y": tr.gaze_y[valid_idx] if tr.gaze_y is not None and valid_idx is not None and valid_idx < len(tr.gaze_y) else np.nan,
                "heart_rate": tr.heart_rate[valid_idx] if tr.heart_rate is not None and valid_idx is not None and valid_idx < len(tr.heart_rate) else np.nan,
                "pupil_avg": (tr.pupil_l[valid_idx] + tr.pupil_r[valid_idx]) / 2 if (tr.pupil_l is not None and tr.pupil_r is not None and valid_idx is not None and valid_idx < len(tr.pupil_l)) else np.nan,
            })
    
    return pd.DataFrame(results)


def analyze_physiological_at_junctions(
    trajectories: Sequence[Trajectory],
    junctions: Sequence[Circle],
    assignments_df: pd.DataFrame,
    decision_mode: str = "hybrid",
    r_outer_list: Optional[Sequence[float]] = None,
    path_length: float = 100.0,
    epsilon: float = 0.05,
    linger_delta: float = 0.0,
    physio_window: float = 3.0,
    base_index: int = 0,
) -> pd.DataFrame:
    """
    Analyze physiological data (heart rate, pupil dilation) at decision points.
    
    Calculation method:
    - Baseline: Average during 2-5 seconds BEFORE entering junction radius (normal navigation)
    - Decision: Average during junction approach period (from entry to exit) (decision-making context)
    """
    from tqdm import tqdm
    
    if r_outer_list is None:
        r_outer_list = [None] * len(junctions)
    
    results = []
    
    print(f"[physio] Analyzing physiological data for {len(trajectories)} trajectories...")
    
    for tr in tqdm(trajectories, desc="Analyzing physiological data", unit="traj"):
        # Get trajectory assignments for all junctions
        traj_assignments = assignments_df[assignments_df["trajectory"] == tr.tid]
        
        if traj_assignments.empty:
            continue
            
        for i, junc in enumerate(junctions):
            # Use the actual junction index (base_index + i) for branch column lookup
            label_idx = base_index + i
            branch_col = f"branch_j{label_idx}"
            branch = traj_assignments[branch_col].iloc[0] if branch_col in traj_assignments.columns else None
            if branch is None or (isinstance(branch, float) and np.isnan(branch)) or int(branch) < 0:
                continue
                
            # Prefer precomputed decisions if provided via assignments_df.
            # Look for decision points specific to this trajectory and junction
            pre_idx = None
            pre_x = np.nan
            pre_z = np.nan
            
            # First try junction-specific columns (e.g., decision_idx_j0, intercept_x_j0, etc.)
            junction_cols = {
                'decision_idx': f"decision_idx_j{i}",
                'intercept_x': f"intercept_x_j{i}",
                'intercept_z': f"intercept_z_j{i}"
            }
            
            if all(col in traj_assignments.columns for col in junction_cols.values()):
                # Use junction-specific columns
                traj_assignments_for_traj = traj_assignments[traj_assignments["trajectory"] == tr.tid]
                if not traj_assignments_for_traj.empty:
                    decision_idx_val = traj_assignments_for_traj[junction_cols['decision_idx']].iloc[0]
                    intercept_x_val = traj_assignments_for_traj[junction_cols['intercept_x']].iloc[0]
                    intercept_z_val = traj_assignments_for_traj[junction_cols['intercept_z']].iloc[0]
                    
                    if not (isinstance(decision_idx_val, float) and np.isnan(decision_idx_val)):
                        pre_idx = int(decision_idx_val)
                    if not (isinstance(intercept_x_val, float) and np.isnan(intercept_x_val)):
                        pre_x = float(intercept_x_val)
                    if not (isinstance(intercept_z_val, float) and np.isnan(intercept_z_val)):
                        pre_z = float(intercept_z_val)
                        
                if pre_idx is not None and not np.isnan(pre_x) and not np.isnan(pre_z):
                    print(f"[physio_debug] Trajectory {tr.tid} at junction {i}: Using precomputed decision point idx={pre_idx}, x={pre_x}, z={pre_z}")
            elif "decision_idx" in traj_assignments.columns:
                # Filter to this specific trajectory
                traj_assignments_for_traj = traj_assignments[traj_assignments["trajectory"] == tr.tid]
                if not traj_assignments_for_traj.empty:
                    # Get the decision point for this trajectory
                    if "decision_idx" in traj_assignments_for_traj.columns:
                        decision_idx_val = traj_assignments_for_traj["decision_idx"].iloc[0]
                        if not (isinstance(decision_idx_val, float) and np.isnan(decision_idx_val)):
                            pre_idx = int(decision_idx_val)
                    if "intercept_x" in traj_assignments_for_traj.columns:
                        intercept_x_val = traj_assignments_for_traj["intercept_x"].iloc[0]
                        if not (isinstance(intercept_x_val, float) and np.isnan(intercept_x_val)):
                            pre_x = float(intercept_x_val)
                    if "intercept_z" in traj_assignments_for_traj.columns:
                        intercept_z_val = traj_assignments_for_traj["intercept_z"].iloc[0]
                        if not (isinstance(intercept_z_val, float) and np.isnan(intercept_z_val)):
                            pre_z = float(intercept_z_val)

            if pre_idx is not None and not np.isnan(pre_x) and not np.isnan(pre_z):
                decision_idx = pre_idx
                decision_time = tr.t[decision_idx] if decision_idx < len(tr.t) else None
                method_used = "precomputed"
                print(f"[physio_debug] Trajectory {tr.tid} at junction {i}: Using precomputed decision point idx={pre_idx}, time={decision_time}")
            else:
                print(f"[physio_debug] Trajectory {tr.tid} at junction {i}: No precomputed decision point, calculating from scratch")
                # Find junction entry point (first sample inside junction radius)
                rx = tr.x - junc.cx
                rz = tr.z - junc.cz
                r = np.hypot(rx, rz)
                inside = r <= junc.r
                entry_idx = int(np.argmax(inside)) if inside.any() else None
                
                if entry_idx is None:
                    continue
                    
                entry_time = tr.t[entry_idx]
                
                # Find decision point (exit point) using the same logic as branch discovery
                decision_time = None
                decision_idx = None
                
                start = entry_idx
                
                # Find decision point based on mode
                if decision_mode == "radial":
                    rout = r_outer_list[i] if (r_outer_list[i] is not None and r_outer_list[i] > junc.r) else (junc.r + 10.0)
                    i_cross = None
                    for i_idx in range(start + 1, len(r)):
                        if r[i_idx] >= rout:
                            j0 = max(start + 1, i_idx - 5)
                            seg = r[j0:i_idx+1]
                            outward = float(np.nanmean(np.diff(seg))) >= 0.0 if seg.size >= 2 else True
                            if outward:
                                i_cross = i_idx
                                break
                    decision_idx = int(i_cross) if i_cross is not None else None
                elif decision_mode == "pathlen":
                    dx = np.diff(tr.x[start:])
                    dz = np.diff(tr.z[start:])
                    seg = np.hypot(dx, dz)
                    cum = np.cumsum(seg)
                    reach_idx = int(np.argmax(cum >= float(path_length))) if (cum >= float(path_length)).any() else None
                    decision_idx = int(start + reach_idx + 1) if reach_idx is not None else None
                else:  # hybrid
                    # Try radial first, fall back to pathlen
                    rout = r_outer_list[i] if (r_outer_list[i] is not None and r_outer_list[i] > junc.r) else (junc.r + 10.0)
                    i_cross = None
                    for i_idx in range(start + 1, len(r)):
                        if r[i_idx] >= rout:
                            j0 = max(start + 1, i_idx - 5)
                            seg = r[j0:i_idx+1]
                            outward = float(np.nanmean(np.diff(seg))) >= 0.0 if seg.size >= 2 else True
                            if outward:
                                i_cross = i_idx
                                break
                    
                    if i_cross is not None:
                        decision_idx = int(i_cross)
                    else:
                        # Fall back to pathlen
                        dx = np.diff(tr.x[start:])
                        dz = np.diff(tr.z[start:])
                        seg = np.hypot(dx, dz)
                        cum = np.cumsum(seg)
                        reach_idx = int(np.argmax(cum >= float(path_length))) if (cum >= float(path_length)).any() else None
                        decision_idx = int(start + reach_idx + 1) if reach_idx is not None else None
                
                if decision_idx is None or decision_idx >= len(tr.t):
                    continue
                    
                decision_time = tr.t[decision_idx]
                method_used = "calculated"
            
            # CRITICAL FIX: Ensure we have a valid decision point before proceeding
            if decision_idx is None or decision_idx >= len(tr.t):
                print(f"[physio_debug] Trajectory {tr.tid} at junction {i}: No valid decision point found, skipping")
                continue
            
            # Find junction entry point for baseline calculation
            rx = tr.x - junc.cx
            rz = tr.z - junc.cz
            r = np.hypot(rx, rz)
            inside = r <= junc.r
            entry_idx = int(np.argmax(inside)) if inside.any() else None
            
            if entry_idx is None:
                continue
                
            entry_time = tr.t[entry_idx]
            
            # OPTION A: Pre-entry baseline vs junction approach period
            # Baseline: 2-5 seconds before junction entry (normal navigation)
            baseline_start_time = entry_time - 5.0  # 5 seconds before entry
            baseline_end_time = entry_time - 2.0    # 2 seconds before entry
            baseline_mask = (tr.t >= baseline_start_time) & (tr.t <= baseline_end_time)
            
            # Decision period: From junction entry to exit (decision-making context)
            decision_mask = (tr.t >= entry_time) & (tr.t <= decision_time)
            
            hr_baseline = hr_decision = np.nan
            pupil_baseline = pupil_decision = np.nan
            
            # Heart rate analysis
            if tr.heart_rate is not None:
                # Baseline period (normal navigation: 2-5 seconds before junction entry)
                if np.any(baseline_mask):
                    hr_baseline_data = tr.heart_rate[baseline_mask]
                    hr_baseline_data = hr_baseline_data[~np.isnan(hr_baseline_data)]
                    if len(hr_baseline_data) > 0:
                        hr_baseline = np.mean(hr_baseline_data)
                
                # Decision period (decision-making context: from junction entry to exit)
                if np.any(decision_mask):
                    hr_decision_data = tr.heart_rate[decision_mask]
                    hr_decision_data = hr_decision_data[~np.isnan(hr_decision_data)]
                    if len(hr_decision_data) > 0:
                        hr_decision = np.mean(hr_decision_data)
            
            # Pupil dilation analysis
            if tr.pupil_l is not None and tr.pupil_r is not None:
                pupil_avg = (tr.pupil_l + tr.pupil_r) / 2
                
                # Baseline period (normal navigation: 2-5 seconds before junction entry)
                if np.any(baseline_mask):
                    pupil_baseline_data = pupil_avg[baseline_mask]
                    pupil_baseline_data = pupil_baseline_data[~np.isnan(pupil_baseline_data)]
                    if len(pupil_baseline_data) > 0:
                        pupil_baseline = np.mean(pupil_baseline_data)
                
                # Decision period (decision-making context: from junction entry to exit)
                if np.any(decision_mask):
                    pupil_decision_data = pupil_avg[decision_mask]
                    pupil_decision_data = pupil_decision_data[~np.isnan(pupil_decision_data)]
                    if len(pupil_decision_data) > 0:
                        pupil_decision = np.mean(pupil_decision_data)
            
            # Debug: Check if valid physiological data is available
            if tr.tid in [0, 1, 2, 3, 4]:  # Debug first few trajectories
                print(f"[physio_debug] Trajectory {tr.tid} at junction {i}: hr_baseline={hr_baseline}, hr_decision={hr_decision}, pupil_baseline={pupil_baseline}, pupil_decision={pupil_decision}")
                print(f"[physio_debug] Decision time: {decision_time}, Entry time: {entry_time}")
                print(f"[physio_debug] Baseline mask: {np.sum(baseline_mask)} samples, Decision mask: {np.sum(decision_mask)} samples")
            
            results.append({
                "trajectory": tr.tid,
                "junction": label_idx,  # Use label_idx instead of i to match the actual junction index
                "branch": int(branch),
                "heart_rate_baseline": hr_baseline,
                "heart_rate_decision": hr_decision,
                "heart_rate_change": hr_decision - hr_baseline if not (np.isnan(hr_baseline) or np.isnan(hr_decision)) else np.nan,
                "pupil_baseline": pupil_baseline,
                "pupil_decision": pupil_decision,
                "pupil_change": pupil_decision - pupil_baseline if not (np.isnan(pupil_baseline) or np.isnan(pupil_decision)) else np.nan,
            })
    
    return pd.DataFrame(results)


def analyze_pupil_dilation_trajectory(
    trajectories: Sequence[Trajectory],
    junctions: Sequence[Circle],
    assignments_df: pd.DataFrame,
    decision_mode: str = "hybrid",
    r_outer_list: Optional[Sequence[float]] = None,
    path_length: float = 100.0,
    epsilon: float = 0.05,
    linger_delta: float = 0.0,
    physio_window: float = 3.0,
    base_index: int = 0,
) -> pd.DataFrame:
    """Analyze pupil dilation trajectory from junction entry to decision point."""
    from tqdm import tqdm
    
    if r_outer_list is None:
        r_outer_list = [None] * len(junctions)
    
    results = []
    
    print(f"[pupil] Analyzing pupil dilation trajectories for {len(trajectories)} trajectories...")
    
    for traj in tqdm(trajectories, desc="Analyzing pupil trajectories", unit="traj"):
        # Get trajectory assignments for all junctions
        traj_assignments = assignments_df[assignments_df["trajectory"] == traj.tid]
        
        if traj_assignments.empty:
            continue
            
        for i, junc in enumerate(junctions):
            # Use the actual junction index (base_index + i) for branch column lookup
            label_idx = base_index + i
            branch_col = f"branch_j{label_idx}"
            branch = traj_assignments[branch_col].iloc[0] if branch_col in traj_assignments.columns else None
            if branch is None or (isinstance(branch, float) and np.isnan(branch)) or int(branch) < 0:
                continue
                
            # Find decision intercept using precomputed decision_idx when available
            r_out = r_outer_list[i]
            
            # First try junction-specific columns (e.g., decision_idx_j0, intercept_x_j0, etc.)
            junction_cols = {
                'decision_idx': f"decision_idx_j{i}",
                'intercept_x': f"intercept_x_j{i}",
                'intercept_z': f"intercept_z_j{i}"
            }
            
            idx = None
            if all(col in traj_assignments.columns for col in junction_cols.values()):
                # Use junction-specific columns
                decision_idx_val = traj_assignments[junction_cols['decision_idx']].iloc[0]
                if not (isinstance(decision_idx_val, float) and np.isnan(decision_idx_val)):
                    idx = int(decision_idx_val)
                    print(f"[pupil_debug] Trajectory {traj.tid} at junction {i}: Using precomputed decision point idx={idx}")
            elif "decision_idx" in traj_assignments.columns:
                val = traj_assignments["decision_idx"].iloc[0]
                if not (isinstance(val, float) and np.isnan(val)):
                    idx = int(val)
                else:
                    if decision_mode == "radial" or (decision_mode == "hybrid" and r_out is not None and float(r_out) > float(junc.r)):
                        idx = get_decision_index(
                            traj.x, traj.z, junc,
                            decision_mode="radial",
                            r_outer=r_out if r_out is not None else (junc.r + 10.0),
                            window=5
                        )
                    else:
                        idx = get_decision_index(
                            traj.x, traj.z, junc,
                            decision_mode="pathlen",
                            path_length=path_length,
                            epsilon=epsilon,
                            linger_delta=linger_delta
                        )
            else:
                if decision_mode == "radial" or (decision_mode == "hybrid" and r_out is not None and float(r_out) > float(junc.r)):
                    idx = get_decision_index(
                        traj.x, traj.z, junc,
                        decision_mode="radial",
                        r_outer=r_out if r_out is not None else (junc.r + 10.0),
                        window=5
                    )
                else:
                    idx = get_decision_index(
                        traj.x, traj.z, junc,
                        decision_mode="pathlen",
                        path_length=path_length,
                        epsilon=epsilon,
                        linger_delta=linger_delta
                    )
            if idx is None:
                # Don't use fallback for trajectories without proper decision points
                # These trajectories should not be assigned to any branch
                continue
            
            if traj.t is None or idx >= len(traj.t):
                continue
            
            # Extract pupil dilation data in time window
            mask = (traj.t >= traj.t[idx] - physio_window) & (traj.t <= traj.t[idx] + physio_window)
            
            pupil_baseline = pupil_decision = np.nan
            
            if traj.pupil_l is not None and traj.pupil_r is not None and np.any(mask):
                pupil_avg = (traj.pupil_l + traj.pupil_r) / 2
                pupil_window = pupil_avg[mask]
                pupil_window = pupil_window[~np.isnan(pupil_window)]
                if len(pupil_window) > 0:
                    pupil_decision = np.mean(pupil_window)
                    
                    # Baseline
                    baseline_mask = traj.t < traj.t[idx] - physio_window
                    if np.any(baseline_mask):
                        pupil_baseline_data = pupil_avg[baseline_mask]
                        pupil_baseline_data = pupil_baseline_data[~np.isnan(pupil_baseline_data)]
                        if len(pupil_baseline_data) > 0:
                            pupil_baseline = np.mean(pupil_baseline_data[-10:])
            
            results.append({
                "trajectory": traj.tid,
                "junction": label_idx,  # Use label_idx instead of i to match the actual junction index
                "branch": int(branch),
                "pupil_baseline": pupil_baseline,
                "pupil_decision": pupil_decision,
                "pupil_change": pupil_decision - pupil_baseline if not (np.isnan(pupil_baseline) or np.isnan(pupil_decision)) else np.nan,
            })
    
    return pd.DataFrame(results)


def plot_gaze_directions_at_junctions(
    trajectories: List[Trajectory],
    junctions: List[Circle],
    gaze_df: pd.DataFrame,
    out_path: str = "Gaze_Directions.png",
    r_outer_list: Optional[List[float]] = None,
    junction_labels: Optional[List[str]] = None,
    centers_list: Optional[List[np.ndarray]] = None,
) -> None:
    """Plot head directions at decision points with improved readability and minimap."""
    
    # Create layout with minimap if multiple junctions
    if len(junctions) > 1:
        fig = plt.figure(figsize=(5*len(junctions), 7))
        gs = fig.add_gridspec(2, len(junctions), height_ratios=[4, 1], hspace=0.3)
        main_axes = [fig.add_subplot(gs[0, i]) for i in range(len(junctions))]
        mini_ax = fig.add_subplot(gs[1, :])
    else:
        fig = plt.figure(figsize=(8, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.3)
        main_axes = [fig.add_subplot(gs[0])]
        mini_ax = fig.add_subplot(gs[1])
    
    axes = main_axes
    
    # Plot each junction separately to reduce overlap
    for j_idx, (ax, junction) in enumerate(zip(axes, junctions)):
        # Background trajectories
        for traj in trajectories:
            ax.plot(traj.x, traj.z, color="lightgray", alpha=0.3, linewidth=0.5, zorder=1)
        
        # Junction circle (inner)
        circle = plt.Circle((junction.cx, junction.cz), junction.r, 
                          fill=False, color="black", linewidth=2, zorder=2)
        ax.add_patch(circle)
        
        # Outer circle (if provided)
        if r_outer_list and j_idx < len(r_outer_list):
            r_outer = r_outer_list[j_idx]
            outer_circle = plt.Circle((junction.cx, junction.cz), r_outer, 
                                    fill=False, color="red", linewidth=2, linestyle="--", 
                                    zorder=2)
            ax.add_patch(outer_circle)
        
        # Junction center point
        ax.scatter([junction.cx], [junction.cz], color="black", s=30, zorder=3)
        
        # Filter gaze data for this junction
        print(f"[gaze_plot_debug] Junction {j_idx}: Looking for junction index {j_idx} in gaze_df")
        print(f"[gaze_plot_debug] Junction {j_idx}: Available junction indices in gaze_df: {sorted(gaze_df['junction'].unique())}")
        print(f"[gaze_plot_debug] Junction {j_idx}: Total gaze_df rows: {len(gaze_df)}")
        
        # Handle individual junction data: if we only have data for one junction, use all the data regardless of junction index
        available_junction_indices = sorted(gaze_df['junction'].unique())
        if len(available_junction_indices) == 1:
            # Single junction data - use all data
            junction_gaze = gaze_df.copy()
            print(f"[gaze_plot_debug] Junction {j_idx}: Single junction data detected, using all {len(junction_gaze)} rows")
        else:
            # Multi-junction data - filter by junction index
            junction_gaze = gaze_df[gaze_df["junction"] == j_idx]
            print(f"[gaze_plot_debug] Junction {j_idx}: Found {len(junction_gaze)} rows after filtering")
        
        # Drop unassigned/outlier branches to avoid confusing colors
        if "branch" in junction_gaze.columns:
            junction_gaze = junction_gaze.copy()
            junction_gaze["branch"] = pd.to_numeric(junction_gaze["branch"], errors="coerce")
            junction_gaze = junction_gaze.dropna(subset=["branch"]).copy()
            junction_gaze["branch"] = junction_gaze["branch"].astype(int)
            junction_gaze = junction_gaze[junction_gaze["branch"] >= 0]
            print(f"[gaze_plot_debug] Junction {j_idx}: After branch filtering: {len(junction_gaze)} rows")
        
        # Debug: Check what gaze data we have
        # Prefer user-provided label; otherwise use coordinates for clarity
        label = None
        try:
            # junction_labels is passed from GUI; keep local if available
            pass
        except Exception:
            pass
        label_str = f"Junction {j_idx}"
        if junction_labels is not None and j_idx < len(junction_labels):
            label_str = f"Junction {junction_labels[j_idx]}"
        else:
            label_str = f"Junction ({junction.cx}, {junction.cz}, r={junction.r})"
        print(f"[gaze_plot] {label_str}: {len(junction_gaze)} gaze records")
        if len(junction_gaze) > 0:
            print(f"[gaze_plot] Branches found: {sorted(junction_gaze['branch'].unique())} for {label_str}")
            print(f"[gaze_plot] Sample head_yaw values: {junction_gaze['head_yaw'].head().tolist()} at {label_str}")
            print(f"[gaze_plot] Sample intercept positions: x={junction_gaze['intercept_x'].head().tolist()}, z={junction_gaze['intercept_z'].head().tolist()} at {label_str}")
            
            # Debug: Show head_yaw values by branch
            for branch in sorted(junction_gaze['branch'].unique()):
                branch_data = junction_gaze[junction_gaze['branch'] == branch]
                valid_head_yaw = branch_data[~np.isnan(branch_data['head_yaw'])]
                print(f"[gaze_plot] Branch {int(branch)}: {len(valid_head_yaw)} valid head_yaw values out of {len(branch_data)} total at {label_str}")
                if len(valid_head_yaw) > 0:
                    print(f"[gaze_plot] Branch {int(branch)} head_yaw range: {valid_head_yaw['head_yaw'].min():.1f}° to {valid_head_yaw['head_yaw'].max():.1f}° at {label_str}")
        
        # Gaze arrows with scaling based on r_outer
        cmap = plt.get_cmap("tab10")

        # Use the original discover branch IDs directly for colors so they match the intercept plot
        unique_branches = sorted(junction_gaze["branch"].unique()) if len(junction_gaze) else []
        branch_remap = {int(b): int(b) for b in unique_branches}
        
        # Scale arrow size and plot limits based on r_outer
        if r_outer_list and j_idx < len(r_outer_list):
            r_outer = r_outer_list[j_idx]
            arrow_scale = max(10, r_outer * 0.3)  # Scale arrows with r_outer
            margin = max(50, r_outer * 1.5 + arrow_scale)  # Scale margin with r_outer + arrow length
        else:
            arrow_scale = 15
            margin = 50
        
        print(f"[gaze_plot] Arrow scale: {arrow_scale}, margin: {margin} at {label_str}")
        
        arrows_drawn = 0
        from collections import defaultdict
        arrows_by_branch = defaultdict(int)  # Count arrows by branch (after remap)
        
        # Optional consistency check vs centers (if provided)
        mismatch_count = 0
        checked = 0

        for _, row in junction_gaze.iterrows():
            if np.isnan(row["head_yaw"]):
                print(f"[gaze_plot] Skipping NaN head_yaw for branch {row['branch']} at {label_str}")
                continue
            
            # Use the actual intercept positions from the gaze DataFrame
            # These should now be in the correct coordinate system
            x = row["intercept_x"]
            z = row["intercept_z"]
            yaw = np.radians(row["head_yaw"])
            dx = arrow_scale * np.sin(yaw)
            dz = arrow_scale * np.cos(yaw)
            
            original_branch = int(row["branch"])
            branch = branch_remap.get(original_branch, original_branch)
            color = cmap(branch % 10)
            
            # Debug: Show arrow details for first few arrows of each branch
            if arrows_by_branch[branch] < 3:  # Show first 3 arrows of each branch
                print(f"[gaze_plot] Branch {original_branch}→{branch} arrow {arrows_by_branch[branch]+1}: pos=({x:.1f}, {z:.1f}), yaw={row['head_yaw']:.1f}°, dx={dx:.1f}, dz={dz:.1f}, color={color}")
            
            ax.arrow(x, z, dx, dz, head_width=arrow_scale*0.15, head_length=arrow_scale*0.2, 
                    fc=color, ec=color, alpha=0.7, zorder=4, linewidth=1)
            arrows_drawn += 1
            arrows_by_branch[branch] += 1
            
            # Compare assignment to nearest center (if centers available)
            if centers_list is not None and j_idx < len(centers_list) and centers_list[j_idx] is not None:
                c = centers_list[j_idx]
                if c.size > 0 and not (np.isnan(x) or np.isnan(z)):
                    vx = x - junction.cx; vz = z - junction.cz
                    vv = np.array([vx, vz], dtype=float)
                    n = np.linalg.norm(vv)
                    if n > 1e-9:
                        vv = vv / n
                        dots = c @ vv
                        nearest = int(np.argmax(dots))
                        checked += 1
                        if nearest != original_branch:
                            mismatch_count += 1
        
            print(f"[gaze_plot] Drew {arrows_drawn} arrows for {label_str}")
            print(f"[gaze_plot] Arrows by branch: {arrows_by_branch} at {label_str}")
        
        if checked > 0 and mismatch_count > 0:
            rate = mismatch_count / checked * 100.0
            print(f"[gaze_plot] Branch-vs-center mismatch at {label_str}: {mismatch_count}/{checked} ({rate:.1f}%)")

        # Set equal aspect and labels
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        if junction_labels and j_idx < len(junction_labels):
            this_label = junction_labels[j_idx]
        else:
            this_label = f"J{j_idx}"
        ax.set_title(f"{this_label}: Head Directions")
        
        # Set axis limits around junction with proper scaling
        ax.set_xlim(junction.cx - margin, junction.cx + margin)
        ax.set_ylim(junction.cz - margin, junction.cz + margin)
        
        # Add legend for branches (show original branch ids) in numeric order
        branch_handles = []
        legend_order = sorted([int(b) for b in unique_branches])
        for original_branch in legend_order:
            color = cmap(int(branch_remap[int(original_branch)]) % 10)
            branch_handles.append(plt.Line2D([0], [0], color=color, linewidth=3, 
                                           label=f"Branch {int(original_branch)}"))
        if branch_handles:
            ax.legend(handles=branch_handles, loc="upper right")
    
    # Add minimap showing overall view
    if len(junctions) > 0:
        # Plot all trajectories in mini-map
        for traj in trajectories:
            mini_ax.plot(traj.x, traj.z, color="0.7", linewidth=0.3, alpha=0.2)
        
        # Plot all junctions
        for i, junc in enumerate(junctions):
            circle = plt.Circle((junc.cx, junc.cz), junc.r, fill=False, 
                              color='red', linewidth=2, alpha=0.8)
            mini_ax.add_patch(circle)
            mini_ax.scatter([junc.cx], [junc.cz], c='red', s=30, marker='o', zorder=5)
            
            # Add junction labels
            label_txt = junction_labels[i] if junction_labels and i < len(junction_labels) else f'J{i}'
            mini_ax.text(junc.cx, junc.cz + junc.r + 10, label_txt, 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add rectangles showing the areas shown in main plots
        for i, ax in enumerate(axes):
            main_xlim = ax.get_xlim()
            main_ylim = ax.get_ylim()
            rect = plt.Rectangle((main_xlim[0], main_ylim[0]), 
                               main_xlim[1] - main_xlim[0], 
                               main_ylim[1] - main_ylim[0],
                               fill=False, color='blue', linewidth=1, linestyle='--', alpha=0.6)
            mini_ax.add_patch(rect)
        
        mini_ax.set_aspect('equal')
        mini_ax.set_xlabel('X (Overall View)')
        mini_ax.set_ylabel('Z (Overall View)')
        mini_ax.set_title('Mini-map: All Junctions and Trajectories')
        mini_ax.grid(True, alpha=0.3)
        
        # Set reasonable limits for mini-map
        all_x = np.concatenate([traj.x for traj in trajectories])
        all_z = np.concatenate([traj.z for traj in trajectories])
        
        # Handle NaN values in coordinates
        valid_x = all_x[~np.isnan(all_x)]
        valid_z = all_z[~np.isnan(all_z)]
        
        if len(valid_x) > 0 and len(valid_z) > 0:
            x_margin = (np.max(valid_x) - np.min(valid_x)) * 0.1
            z_margin = (np.max(valid_z) - np.min(valid_z)) * 0.1
            mini_ax.set_xlim(np.min(valid_x) - x_margin, np.max(valid_x) + x_margin)
            mini_ax.set_ylim(np.min(valid_z) - z_margin, np.max(valid_z) + z_margin)
        else:
            # Fallback: set reasonable default limits
            mini_ax.set_xlim(-100, 100)
            mini_ax.set_ylim(-100, 100)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_physiological_by_branch(
    physio_df: pd.DataFrame,
    out_path: str = "Physiological_Analysis.png"
) -> None:
    """Plot heart rate and pupil dilation changes by branch choice."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heart rate changes by branch
    hr_data = physio_df.dropna(subset=["heart_rate_change"])
    if len(hr_data) > 0:
        branches = sorted(hr_data["branch"].unique())
        hr_by_branch = [hr_data[hr_data["branch"] == b]["heart_rate_change"].values for b in branches]
        
        ax1.boxplot(hr_by_branch, labels=[f"Branch {b}" for b in branches])
        ax1.set_ylabel("Heart Rate Change (bpm)")
        ax1.set_title("Heart Rate Change at Decision Points\n(Baseline: Normal navigation 2-5s before junction entry)")
        ax1.grid(True, alpha=0.3)
    
    # Pupil dilation changes by branch
    pupil_data = physio_df.dropna(subset=["pupil_change"])
    if len(pupil_data) > 0:
        branches = sorted(pupil_data["branch"].unique())
        pupil_by_branch = [pupil_data[pupil_data["branch"] == b]["pupil_change"].values for b in branches]
        
        ax2.boxplot(pupil_by_branch, labels=[f"Branch {b}" for b in branches])
        ax2.set_ylabel("Pupil Dilation Change (mm)")
        ax2.set_title("Pupil Dilation Change at Decision Points\n(Baseline: Normal navigation 2-5s before junction entry)")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pupil_trajectory_analysis(
    pupil_df: pd.DataFrame,
    out_path: str = "Pupil_Trajectory_Analysis.png"
) -> None:
    """Plot pupil dilation trajectory analysis by junction."""
    
    junctions = sorted(pupil_df["junction"].unique())
    n_junctions = len(junctions)
    
    fig, axes = plt.subplots(1, n_junctions, figsize=(5*n_junctions, 5))
    if n_junctions == 1:
        axes = [axes]
    
    for j_idx, ax in enumerate(axes):
        junction_data = pupil_df[pupil_df["junction"] == j_idx]
        
        if len(junction_data) == 0:
            ax.set_title(f"Junction {j_idx}: No Data")
            continue
        
        # Box plot of pupil changes by branch
        branches = sorted(junction_data["branch"].unique())
        pupil_by_branch = [junction_data[junction_data["branch"] == b]["pupil_change"].values for b in branches]
        
        ax.boxplot(pupil_by_branch, labels=[f"Branch {b}" for b in branches])
        ax.set_ylabel("Pupil Dilation Change (mm)")
        ax.set_title(f"Junction {j_idx}: Pupil Dilation Changes\n(Baseline: Normal navigation 2-5s before junction entry)")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def gaze_movement_consistency_report(gaze_df: pd.DataFrame) -> Dict[str, float]:
    """Generate summary statistics for gaze-movement alignment."""
    
    # Check if DataFrame is empty or missing required columns
    if len(gaze_df) == 0:
        return {"error": "Empty gaze DataFrame", "total_decisions": 0}
    
    if "yaw_difference" not in gaze_df.columns:
        return {
            "error": "Missing yaw_difference column", 
            "total_decisions": len(gaze_df),
            "available_columns": list(gaze_df.columns)
        }
    
    valid_data = gaze_df.dropna(subset=["yaw_difference"])
    
    if len(valid_data) == 0:
        return {"error": "No valid gaze-movement data", "total_decisions": len(gaze_df)}
    
    # Alignment metrics
    mean_abs_diff = np.mean(np.abs(valid_data["yaw_difference"]))
    aligned_threshold = 30  # degrees
    aligned_pct = np.mean(np.abs(valid_data["yaw_difference"]) < aligned_threshold) * 100
    
    # By branch analysis
    branch_consistency = {}
    for branch in sorted(valid_data["branch"].unique()):
        branch_data = valid_data[valid_data["branch"] == branch]
        branch_consistency[f"branch_{branch}_alignment"] = np.mean(np.abs(branch_data["yaw_difference"]))
    
    return {
        "mean_absolute_yaw_difference": mean_abs_diff,
        "aligned_percentage": aligned_pct,
        "total_decisions": len(valid_data),
        **branch_consistency
    }


def create_pupil_dilation_heatmap(
    trajectories: Sequence[Trajectory],
    junctions: Optional[Sequence[Circle]] = None,
    grid_size: Optional[int] = None,
    cell_size: Optional[float] = None,
    normalization: str = "relative",
    aggregation: str = "mean",
    x_bounds: Optional[Tuple[float, float]] = None,
    y_bounds: Optional[Tuple[float, float]] = None
) -> Dict[str, any]:
    """
    Create spatial heatmap of pupil dilation changes.
    
    Args:
        trajectories: List of trajectory objects with pupil data
        junctions: Optional list of junctions to overlay
        grid_size: Number of grid cells per dimension (1-100)
        normalization: "relative" (% change from baseline) or "zscore" (standard deviations)
        aggregation: "mean" or "max" for combining values in bins
        x_bounds: Optional (min, max) for x-axis, auto-calculated if None
        y_bounds: Optional (min, max) for z-axis, auto-calculated if None
    
    Returns:
        Dictionary containing:
        - heatmap: 2D numpy array of pupil values (masked where no data)
        - x_edges: Grid x boundaries
        - z_edges: Grid z boundaries
        - sample_counts: 2D array of sample counts per bin
        - normalization_used: str
        - aggregation_used: str
    """
    print(f"[heatmap] Creating pupil dilation heatmap with grid_size={grid_size}, normalization={normalization}")
    
    # Debug: Check trajectory data availability
    print(f"[heatmap_debug] Processing {len(trajectories)} trajectories")
    pupil_data_count = 0
    for i, traj in enumerate(trajectories[:5]):  # Check first 5 trajectories for better sampling
        has_pupil_l = traj.pupil_l is not None
        has_pupil_r = traj.pupil_r is not None
        print(f"[heatmap_debug] Trajectory {i}: pupil_l={has_pupil_l}, pupil_r={has_pupil_r}")
        if has_pupil_l and has_pupil_r:
            pupil_data_count += 1
            print(f"[heatmap_debug] Trajectory {i}: pupil_l length={len(traj.pupil_l)}, pupil_r length={len(traj.pupil_r)}")
            if len(traj.pupil_l) > 0:
                print(f"[heatmap_debug] Trajectory {i}: pupil_l sample values={traj.pupil_l[:5]}")
    
    print(f"[heatmap_debug] {pupil_data_count}/{min(5, len(trajectories))} sampled trajectories have pupil data")
    print(f"[heatmap_debug] Note: This is a sample check - all {len(trajectories)} trajectories will be processed")
    
    # Collect all pupil data with spatial coordinates
    all_x = []
    all_z = []
    all_pupil_changes = []
    
    valid_traj_count = 0
    
    for traj in trajectories:
        # Check if trajectory has pupil data
        if traj.pupil_l is None or traj.pupil_r is None:
            continue
        
        # Calculate average pupil size
        pupil_avg = (traj.pupil_l + traj.pupil_r) / 2.0
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(pupil_avg) | np.isnan(traj.x) | np.isnan(traj.z))
        if not np.any(valid_mask):
            continue
        
        pupil_valid = pupil_avg[valid_mask]
        x_valid = traj.x[valid_mask]
        z_valid = traj.z[valid_mask]
        
        if len(pupil_valid) < 2:
            continue
        
        # Calculate baseline (trajectory mean)
        baseline = np.nanmean(pupil_valid)
        
        if baseline == 0 or np.isnan(baseline):
            continue
        
        # Calculate pupil changes based on normalization method
        if normalization == "relative":
            # Relative change as percentage
            pupil_changes = ((pupil_valid - baseline) / baseline) * 100.0
        elif normalization == "zscore":
            # Z-score normalization
            std = np.nanstd(pupil_valid)
            if std == 0 or np.isnan(std):
                continue
            pupil_changes = (pupil_valid - baseline) / std
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        # Add to collection
        all_x.extend(x_valid)
        all_z.extend(z_valid)
        all_pupil_changes.extend(pupil_changes)
        valid_traj_count += 1
    
    print(f"[heatmap] Processed {valid_traj_count}/{len(trajectories)} trajectories with valid pupil data")
    print(f"[heatmap] Total data points: {len(all_x)}")
    
    if len(all_x) == 0:
        print("[heatmap] WARNING: No valid pupil data found")
        return {
            "heatmap": None,
            "x_edges": None,
            "z_edges": None,
            "sample_counts": None,
            "normalization_used": normalization,
            "aggregation_used": aggregation,
            "error": "No valid pupil data"
        }
    
    # Convert to numpy arrays
    all_x = np.array(all_x)
    all_z = np.array(all_z)
    all_pupil_changes = np.array(all_pupil_changes)
    
    # Determine bounds
    if x_bounds is None:
        x_min, x_max = np.min(all_x), np.max(all_x)
        # Add 5% padding
        x_padding = (x_max - x_min) * 0.05
        x_bounds = (x_min - x_padding, x_max + x_padding)
        print(f"[heatmap] Calculated x_bounds from data: ({x_bounds[0]:.1f}, {x_bounds[1]:.1f})")
    else:
        print(f"[heatmap] Using provided x_bounds: ({x_bounds[0]:.1f}, {x_bounds[1]:.1f})")
    
    if y_bounds is None:
        z_min, z_max = np.min(all_z), np.max(all_z)
        # Add 5% padding
        z_padding = (z_max - z_min) * 0.05
        y_bounds = (z_min - z_padding, z_max + z_padding)
        print(f"[heatmap] Calculated y_bounds from data: ({y_bounds[0]:.1f}, {y_bounds[1]:.1f})")
    else:
        print(f"[heatmap] Using provided y_bounds: ({y_bounds[0]:.1f}, {y_bounds[1]:.1f})")
    
    # Calculate grid_size from cell_size if provided
    if cell_size is not None:
        x_range = x_bounds[1] - x_bounds[0]
        z_range = y_bounds[1] - y_bounds[0]
        # Ensure minimum grid size for visibility
        grid_size = max(5, int(max(x_range, z_range) / cell_size))
        print(f"[heatmap] Calculated grid_size={grid_size} from cell_size={cell_size} (x_range={x_range:.1f}, z_range={z_range:.1f})")
        print(f"[heatmap] Using bounds: x=({x_bounds[0]:.1f}, {x_bounds[1]:.1f}), z=({y_bounds[0]:.1f}, {y_bounds[1]:.1f})")
    elif grid_size is None:
        grid_size = 10  # Default fallback
        print(f"[heatmap] Using default grid_size={grid_size}")
    
    # Create grid edges
    x_edges = np.linspace(x_bounds[0], x_bounds[1], grid_size + 1)
    z_edges = np.linspace(y_bounds[0], y_bounds[1], grid_size + 1)
    
    # Bin the data - first get counts
    sample_counts, _, _ = np.histogram2d(
        all_x, all_z,
        bins=[x_edges, z_edges]
    )
    
    # Create heatmap based on aggregation method
    if aggregation == "mean":
        # Calculate sum for each bin, then divide by count
        pupil_sum, _, _ = np.histogram2d(
            all_x, all_z,
            bins=[x_edges, z_edges],
            weights=all_pupil_changes
        )
        
        # Avoid division by zero with better handling
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = np.where(sample_counts > 0, pupil_sum / sample_counts, np.nan)
        
    elif aggregation == "max":
        # Use maximum absolute change in each bin
        heatmap = np.full((grid_size, grid_size), np.nan)
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Find points in this bin
                mask = (
                    (all_x >= x_edges[i]) & (all_x < x_edges[i+1]) &
                    (all_z >= z_edges[j]) & (all_z < z_edges[j+1])
                )
                
                if np.any(mask):
                    bin_values = all_pupil_changes[mask]
                    # Take value with maximum absolute magnitude
                    abs_max_idx = np.argmax(np.abs(bin_values))
                    heatmap[i, j] = bin_values[abs_max_idx]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Transpose to match imshow convention (z is first axis)
    heatmap = heatmap.T
    sample_counts = sample_counts.T
    
    print(f"[heatmap] Heatmap created: {grid_size}x{grid_size} grid")
    print(f"[heatmap] Valid bins: {np.sum(~np.isnan(heatmap))}/{grid_size*grid_size}")
    print(f"[heatmap] Value range: {np.nanmin(heatmap):.2f} to {np.nanmax(heatmap):.2f}")
    
    return {
        "heatmap": heatmap,
        "x_edges": x_edges,
        "z_edges": z_edges,
        "sample_counts": sample_counts,
        "normalization_used": normalization,
        "aggregation_used": aggregation,
        "valid_trajectories": valid_traj_count,
        "total_points": len(all_x)
    }


def plot_pupil_dilation_heatmap(
    heatmap_data: Dict[str, any],
    junctions: Optional[Sequence[Circle]] = None,
    trajectories: Optional[Sequence[Trajectory]] = None,
    all_trajectories: Optional[Sequence[Trajectory]] = None,
    title: str = "Pupil Dilation Heatmap",
    out_path: Optional[str] = None,
    show_sample_counts: bool = False,
    show_minimap: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> plt.Figure:
    """
    Plot pupil dilation heatmap with overlays.
    
    Args:
        heatmap_data: Dictionary from create_pupil_dilation_heatmap()
        junctions: Optional list of junctions to overlay
        trajectories: Optional list of trajectories to show as background
        title: Plot title
        out_path: Optional path to save figure
        show_sample_counts: If True, annotate bins with sample counts
        show_minimap: If True, show minimap (default True, disable for per-junction plots)
    
    Returns:
        Matplotlib figure
    """
    # Check for errors
    if heatmap_data.get("error"):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"Error: {heatmap_data['error']}", 
                ha='center', va='center', fontsize=14)
        ax.set_title(title)
        return fig
    
    heatmap = heatmap_data["heatmap"]
    x_edges = heatmap_data["x_edges"]
    z_edges = heatmap_data["z_edges"]
    sample_counts = heatmap_data["sample_counts"]
    normalization = heatmap_data["normalization_used"]
    
    # Create figure with main plot and optional minimap
    if show_minimap:
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 5, width_ratios=[4, 0.2, 1, 0.2, 0.3], wspace=0.4)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_minimap = fig.add_subplot(gs[0, 2])
        ax_colorbar = fig.add_subplot(gs[0, 4])
    else:
        fig = plt.figure(figsize=(12, 9))
        gs = fig.add_gridspec(1, 2, width_ratios=[5, 0.3], wspace=0.15)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_colorbar = fig.add_subplot(gs[0, 1])
        ax_minimap = None
    
    # Determine colormap scale
    if vmin is not None and vmax is not None:
        # Use provided consistent scaling
        pass
    else:
        # Use dynamic scaling (fallback)
        # Ensure heatmap is a numpy array
        if not isinstance(heatmap, np.ndarray):
            heatmap = np.array(heatmap)
        valid_values = heatmap[~np.isnan(heatmap)]
        if len(valid_values) == 0:
            vmin, vmax = -10, 10
        else:
            # For relative changes, center at 0
            if normalization == "relative":
                abs_max = np.max(np.abs(valid_values))
                # Ensure minimum range for visibility
                min_range = 1.0  # Minimum 1% range
                abs_max = max(abs_max, min_range)
                vmin, vmax = -abs_max, abs_max
                print(f"[plot_debug] Dynamic scaling: abs_max={abs_max:.2f}, range=[{vmin:.2f}, {vmax:.2f}]")
            else:  # zscore
                vmin, vmax = np.min(valid_values), np.max(valid_values)
    
    # Plot trajectories as background (before heatmap for context)
    if trajectories:
        for traj in trajectories:
            if hasattr(traj, 'x') and hasattr(traj, 'z'):
                # Plot with moderate alpha for visible background
                ax_main.plot(traj.x, traj.z, color='gray', alpha=0.35, linewidth=0.8, zorder=1)
    
    # Debug: Check heatmap data
    # Ensure heatmap is a numpy array
    if not isinstance(heatmap, np.ndarray):
        heatmap = np.array(heatmap)
    print(f"[plot_debug] Heatmap shape: {heatmap.shape}")
    print(f"[plot_debug] Heatmap bounds: x=({x_edges[0]:.1f}, {x_edges[-1]:.1f}), z=({z_edges[0]:.1f}, {z_edges[-1]:.1f})")
    print(f"[plot_debug] Valid heatmap values: {np.sum(~np.isnan(heatmap))}/{heatmap.size}")
    if np.sum(~np.isnan(heatmap)) > 0:
        print(f"[plot_debug] Heatmap value range: {np.nanmin(heatmap):.2f} to {np.nanmax(heatmap):.2f}")
        print(f"[plot_debug] Color scale: vmin={vmin}, vmax={vmax}")
    else:
        print(f"[plot_debug] WARNING: No valid heatmap data found!")
    
    # Plot main heatmap using pcolormesh for discrete grid data
    im = ax_main.pcolormesh(
        x_edges, z_edges, heatmap,
        cmap='RdBu_r',  # Red for dilation, blue for constriction
        vmin=vmin,
        vmax=vmax,
        shading='flat',  # No interpolation between cells
        alpha=0.85,  # Slightly transparent to show trajectories
        zorder=2
    )
    
    # CRITICAL FIX: Set axis limits to match heatmap bounds to ensure proper zoom
    ax_main.set_xlim(x_edges[0], x_edges[-1])
    ax_main.set_ylim(z_edges[0], z_edges[-1])
    
    # Add explicit grid lines to make discrete cells more visible
    ax_main.grid(True, linestyle=':', alpha=0.3, color='black', linewidth=0.5)
    
    # Add sample count annotations if requested
    if show_sample_counts:
        grid_size = heatmap.shape[0]
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        
        for i in range(grid_size):
            for j in range(grid_size):
                count = int(sample_counts[i, j])
                if count > 0:
                    # Only show if count is significant
                    if count >= 5:
                        ax_main.text(x_centers[j], z_centers[i], str(count),
                                   ha='center', va='center', fontsize=6,
                                   color='black' if abs(heatmap[i, j]) < (vmax-vmin)/2 else 'white')
    
    # Overlay junctions (on top of everything)
    if junctions:
        for idx, junc in enumerate(junctions):
            circle = plt.Circle((junc.cx, junc.cz), junc.r, 
                              fill=False, edgecolor='green', linewidth=2, linestyle='--', zorder=5)
            ax_main.add_patch(circle)
            
            # Use the correct junction index from heatmap data if available
            junction_label_idx = idx
            if 'junction_idx' in heatmap_data:
                junction_label_idx = heatmap_data['junction_idx']
                print(f"[plot_debug] Using junction index from heatmap data: {junction_label_idx}")
            else:
                print(f"[plot_debug] Using local junction index: {idx}")
            
            ax_main.text(junc.cx, junc.cz, f'J{junction_label_idx}', 
                        ha='center', va='center', fontsize=10, 
                        color='green', weight='bold', zorder=6,
                        bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', edgecolor='green'))
            
            # Add junction radius display
            radius_text = f'r={junc.r:.1f}'
            ax_main.text(junc.cx, junc.cz + junc.r + 5, radius_text,
                        ha='center', va='bottom', fontsize=8,
                        color='green', weight='bold', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='green', alpha=0.8))
    
    ax_main.set_xlabel('X Position', fontsize=12)
    ax_main.set_ylabel('Z Position', fontsize=12)
    ax_main.set_title(title, fontsize=14, weight='bold')
    ax_main.grid(True, alpha=0.3, linestyle=':')
    
    # Add colorbar
    if show_minimap:
        cbar = plt.colorbar(im, cax=ax_colorbar, orientation='vertical')
    else:
        cbar = plt.colorbar(im, cax=ax_colorbar, orientation='vertical')
    
    if normalization == "relative":
        cbar.set_label('Pupil Dilation Change (%)', fontsize=11)
        
        # Create consistent colorbar labels: 0, 20, 40, >60 (and negative equivalents)
        print(f"[colorbar_debug] vmin={vmin}, vmax={vmax}")
        
        # Use consistent scale for all heatmaps to improve comparability
        # Normal pupil dilation range: ±60%, outliers get same color
        ticks = [-60, -40, -20, 0, 20, 40, 60]
        tick_labels = ['<-60', '-40', '-20', '0', '20', '40', '>60']
        
        print(f"[colorbar_debug] Using ticks: {ticks}")
        print(f"[colorbar_debug] Using labels: {tick_labels}")
        
        # Set consistent color scale limits to ensure same colors across all plots
        im.set_clim(-60, 60)
        
        # Set ticks and labels
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
    else:
        cbar.set_label('Pupil Dilation (Z-score)', fontsize=11)
    
    # Create minimap (overview of all data) - only if requested
    if show_minimap and ax_minimap is not None:
        # Plot trajectories in minimap (use all_trajectories if available for full map view)
        minimap_trajectories = all_trajectories if all_trajectories is not None else trajectories
        if minimap_trajectories:
            for traj in minimap_trajectories:
                if hasattr(traj, 'x') and hasattr(traj, 'z'):
                    ax_minimap.plot(traj.x, traj.z, color='gray', alpha=0.4, linewidth=0.5)
        
        # Overlay junctions on minimap (no heatmap for global view)
        if junctions:
            for idx, junc in enumerate(junctions):
                circle = plt.Circle((junc.cx, junc.cz), junc.r, 
                                  fill=False, edgecolor='red', linewidth=1, zorder=10)
                ax_minimap.add_patch(circle)
                ax_minimap.plot(junc.cx, junc.cz, 'ro', markersize=4, zorder=10)
        
        ax_minimap.set_title('Overview', fontsize=10)
        ax_minimap.set_xticks([])
        ax_minimap.set_yticks([])
        ax_minimap.set_aspect('equal')
        
        # CRITICAL FIX: Set minimap limits to show full data extent, not zoomed view
        # For junction heatmaps, we need to show the full map extent
        if minimap_trajectories:
            all_x_coords = []
            all_z_coords = []
            for traj in minimap_trajectories:
                if hasattr(traj, 'x') and hasattr(traj, 'z'):
                    all_x_coords.extend(traj.x)
                    all_z_coords.extend(traj.z)
            
            if all_x_coords and all_z_coords:
                x_margin = (max(all_x_coords) - min(all_x_coords)) * 0.1
                z_margin = (max(all_z_coords) - min(all_z_coords)) * 0.1
                ax_minimap.set_xlim(min(all_x_coords) - x_margin, max(all_x_coords) + x_margin)
                ax_minimap.set_ylim(min(all_z_coords) - z_margin, max(all_z_coords) + z_margin)
        
        # Add statistics text below minimap
        stats_text = f"Valid bins: {np.sum(~np.isnan(heatmap))}/{heatmap.size}\n"
        stats_text += f"Trajectories: {heatmap_data['valid_trajectories']}\n"
        stats_text += f"Data points: {heatmap_data['total_points']}"
        
        ax_minimap.text(0.5, -0.15, stats_text, fontsize=8, 
                       ha='center', va='top', transform=ax_minimap.transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"[heatmap] Saved plot to {out_path}")
    
    return fig


def get_consistent_pupil_scaling(heatmap_data_list, normalization="relative"):
    """
    Get consistent color scaling for pupil dilation heatmaps across all plots.
    
    Args:
        heatmap_data_list: List of heatmap data dictionaries
        normalization: "relative" or "zscore"
    
    Returns:
        tuple: (vmin, vmax) for consistent scaling
    """
    all_values = []
    
    # Collect all valid values from all heatmaps
    for heatmap_data in heatmap_data_list:
        if heatmap_data and 'heatmap' in heatmap_data and heatmap_data['heatmap'] is not None:
            heatmap = heatmap_data['heatmap']
            # Ensure heatmap is a numpy array
            if not isinstance(heatmap, np.ndarray):
                heatmap = np.array(heatmap)
            valid_values = heatmap[~np.isnan(heatmap)]
            all_values.extend(valid_values)
    
    if len(all_values) == 0:
        return -10, 10  # Default fallback
    
    all_values = np.array(all_values)
    
    if normalization == "relative":
        # For relative changes, use consistent percentage-based scaling
        # Cap extreme values but allow strong colors for outliers
        abs_max = np.max(np.abs(all_values))
        
        # Use realistic limits for pupil dilation changes based on medical research
        # Normal pupil dilation changes are typically 5-25% in most situations
        if abs_max <= 5:
            vmin, vmax = -5, 5
        elif abs_max <= 10:
            vmin, vmax = -10, 10
        elif abs_max <= 25:
            vmin, vmax = -25, 25
        elif abs_max <= 50:
            vmin, vmax = -50, 50
        else:
            # For extreme values (>50%), cap at 60% to accommodate clipping labels
            vmin, vmax = -60, 60
            print(f"[scaling_debug] Extreme values detected (abs_max={abs_max:.1f}%), capping at ±60%")
    else:  # zscore
        # For z-score, use standard deviation-based scaling
        mean_val = np.mean(all_values)
        std_val = np.std(all_values)
        vmin = mean_val - 3 * std_val
        vmax = mean_val + 3 * std_val
    
    print(f"[scaling_debug] Final scaling: vmin={vmin}, vmax={vmax}")
    return vmin, vmax


def create_per_junction_pupil_heatmap(
    trajectories: Sequence[Trajectory],
    junctions: Sequence[Circle],
    r_outer_list: Optional[Sequence[float]] = None,
    grid_size: Optional[int] = None,
    cell_size: Optional[float] = None,
    normalization: str = "relative",
    base_index: int = 0,
) -> Dict[int, Dict[str, any]]:
    """
    Create focused heatmaps for each junction separately.
    
    Args:
        trajectories: List of trajectory objects with pupil data
        junctions: List of junction circles
        r_outer_list: List of r_outer values (analysis radius) for each junction
        grid_size: Number of grid cells per dimension
        normalization: "relative" or "zscore"
    
    Returns:
        Dictionary mapping junction index to heatmap data
    """
    if r_outer_list is None:
        r_outer_list = [100.0] * len(junctions)
    
    junction_heatmaps = {}
    
    for idx, (junction, r_outer) in enumerate(zip(junctions, r_outer_list)):
        global_idx = base_index + idx
        label_str = f"Junction {global_idx} ({junction.cx}, {junction.cz}, r={junction.r})"
        print(f"[heatmap] Creating per-junction heatmap for {label_str}")
        
        # Filter trajectory points within junction area (including inlet)
        filtered_trajs = []
        
        for traj in trajectories:
            if traj.pupil_l is None or traj.pupil_r is None:
                continue
            
            # Calculate distances from junction center
            rx = traj.x - junction.cx
            rz = traj.z - junction.cz
            r = np.hypot(rx, rz)
            
            # Keep points within r_outer radius
            inlet_mask = r <= r_outer
            
            if not np.any(inlet_mask):
                continue
            
            # Create filtered trajectory with only junction-relevant points
            # Create a simple object to hold the filtered data
            class FilteredTraj:
                def __init__(self, tid, x, z, pupil_l, pupil_r):
                    self.tid = tid
                    self.x = x
                    self.z = z
                    self.pupil_l = pupil_l
                    self.pupil_r = pupil_r
            
            filtered_traj = FilteredTraj(
                tid=traj.tid,
                x=traj.x[inlet_mask],
                z=traj.z[inlet_mask],
                pupil_l=traj.pupil_l[inlet_mask] if traj.pupil_l is not None else None,
                pupil_r=traj.pupil_r[inlet_mask] if traj.pupil_r is not None else None
            )
            
            filtered_trajs.append(filtered_traj)
        
        print(f"[heatmap] {label_str}: {len(filtered_trajs)} trajectories pass through")
        
        if len(filtered_trajs) == 0:
            junction_heatmaps[idx] = {
                "error": "No trajectories pass through this junction",
                "junction": junction,
                "r_outer": r_outer
            }
            continue
        
        # Calculate bounds centered on junction
        # Use r_outer + buffer to properly encompass the junction area
        buffer = r_outer * 0.2  # 20% buffer beyond r_outer
        plot_radius = r_outer + buffer
        x_bounds = (junction.cx - plot_radius, junction.cx + plot_radius)
        y_bounds = (junction.cz - plot_radius, junction.cz + plot_radius)
        
        print(f"[heatmap] {label_str}: plot_radius={plot_radius:.1f}, buffer={buffer:.1f}")
        print(f"[heatmap] {label_str}: bounds x=({x_bounds[0]:.1f}, {x_bounds[1]:.1f}), z=({y_bounds[0]:.1f}, {y_bounds[1]:.1f})")
        
        # Create heatmap for this junction
        heatmap_data = create_pupil_dilation_heatmap(
            trajectories=filtered_trajs,
            junctions=[junction],
            grid_size=grid_size,
            cell_size=cell_size,
            normalization=normalization,
            x_bounds=x_bounds,
            y_bounds=y_bounds
        )
        
        # Add junction-specific metadata
        heatmap_data["junction"] = junction
        heatmap_data["junction_idx"] = global_idx
        heatmap_data["r_outer"] = r_outer
        
        # Debug: Print junction index information
        print(f"[heatmap_debug] Junction {global_idx}: base_index={base_index}, idx={idx}, global_idx={global_idx}")
        print(f"[heatmap_debug] Junction coordinates: ({junction.cx}, {junction.cz}), radius={junction.r}")
        
        junction_heatmaps[idx] = heatmap_data
    
    return junction_heatmaps