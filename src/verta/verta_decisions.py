# Route-decision extraction

from collections import Counter
import math
import os
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from verta.verta_clustering import best_k_by_silhouette, cluster_angles_dbscan, kmeans_2d, merge_close_centers
from verta.verta_geometry import Circle, entered_junction_idx
from verta.verta_data_loader import Trajectory
from verta.verta_plotting import plot_decision_intercepts
from verta.verta_logging import get_logger

logger = get_logger()

def get_decision_index(
    x: np.ndarray,
    z: np.ndarray,
    junction: Circle,
    decision_mode: str,
    path_length: float = 100.0,
    r_outer: Optional[float] = None,
    epsilon: float = 0.05,
    linger_delta: float = 0.0,
    window: int = 5
) -> Optional[int]:
    """
    Get decision index using the specified decision mode.

    Args:
        x, z: Trajectory coordinates
        junction: Junction circle
        decision_mode: "pathlen", "radial", or "hybrid"
        path_length: Path length for pathlen mode
        r_outer: Outer radius for radial mode
        epsilon: Minimum step size
        linger_delta: Linger distance beyond junction
        window: Window size for radial mode

    Returns:
        Decision index or None if not found
    """
    if len(x) < 2:
        return None

    # Find junction entry point
    rx = x - junction.cx
    rz = z - junction.cz
    r = np.hypot(rx, rz)
    inside = r <= junction.r
    start = int(np.argmax(inside)) if inside.any() else int(np.argmin(r))

    if decision_mode == "radial":
        rout = r_outer if (r_outer is not None and r_outer > junction.r) else (junction.r + 10.0)
        return _get_radial_decision_index(x, z, junction, rout, start, window)
    elif decision_mode == "pathlen":
        return _get_pathlen_decision_index(x, z, junction, path_length, epsilon, linger_delta, start)
    elif decision_mode == "hybrid":
        # Try radial first, fall back to pathlen
        rout = r_outer if (r_outer is not None and r_outer > junction.r) else (junction.r + 10.0)
        idx = _get_radial_decision_index(x, z, junction, rout, start, window)
        if idx is not None:
            return idx
        return _get_pathlen_decision_index(x, z, junction, path_length, epsilon, linger_delta, start)

    return None


def _get_radial_decision_index(
    x: np.ndarray,
    z: np.ndarray,
    junction: Circle,
    r_outer: float,
    start: int,
    window: int
) -> Optional[int]:
    """Get decision index using radial mode."""
    rx = x - junction.cx
    rz = z - junction.cz
    r = np.hypot(rx, rz)

    # Find the first index crossing r_outer with outward trend
    for i in range(start + 1, len(r)):
        if r[i] >= r_outer:
            j0 = max(start + 1, i - window)
            seg = r[j0:i+1]
            if seg.size >= 2:
                outward = float(np.nanmean(np.diff(seg))) >= 0.0
            else:
                outward = True
            if outward:
                return i
    return None


def _get_pathlen_decision_index(
    x: np.ndarray,
    z: np.ndarray,
    junction: Circle,
    path_length: float,
    epsilon: float,
    linger_delta: float,
    start: int
) -> Optional[int]:
    """Get decision index using path length mode."""
    dx = np.diff(x[start:])
    dz = np.diff(z[start:])
    seg = np.hypot(dx, dz)
    if len(seg) == 0:
        return None

    cum = np.cumsum(seg)
    reach_idx = int(np.argmax(cum >= path_length)) if (cum >= path_length).any() else None

    if reach_idx is not None:
        return start + reach_idx + 1
    return None


def first_unit_vector_after_distance(
    x: np.ndarray,
    z: np.ndarray,
    origin_region: Circle,
    path_length: float = 100.0,
    epsilon: float = 0.05,
    fallback_window: int = 10,
    linger_delta: float = 0.0
) -> Optional[np.ndarray]:
    """
    Returns a single unit direction vector capturing initial route choice.
    Fallback order:
      T1. First step >= epsilon after reaching `path_length`.
      T2. Largest single step anywhere >= epsilon.
      T3. Net displacement over last `fallback_window` steps (ignores epsilon).
    Returns None only if there is no motion at all.
    """
    if len(x) < 2 or len(z) < 2:
        return None

    min_radial = origin_region.r + max(5.0, linger_delta)  # Require at least 5 units from junction center

    # Start: first time inside the circle; else closest approach
    dist = np.hypot(x - origin_region.cx, z - origin_region.cz)
    inside = dist <= origin_region.r
    start = int(np.argmax(inside)) if inside.any() else int(np.argmin(dist))

    dx = np.diff(x[start:])
    dz = np.diff(z[start:])
    seg = np.hypot(dx, dz)
    if len(seg) == 0:
        return None

    cum = np.cumsum(seg)
    reach_idx = int(np.argmax(cum >= path_length)) if (cum >= path_length).any() else None

    # T1: first step >= epsilon after we reached the requested path length
    if reach_idx is not None:
        for j in range(reach_idx, len(dx)):
            if seg[j] >= epsilon:
                i = j  # step index after 'start'
                rad_now = float(np.hypot(x[start + i] - origin_region.cx,
                                        z[start + i] - origin_region.cz))
                if rad_now < min_radial:
                    continue  # keep scanning for a later, farther step
                v = np.array([dx[j], dz[j]]) / seg[j]
                return v

    # T2: largest single step anywhere
    jmax = int(np.argmax(seg))
    if seg[jmax] > 0:
        rad_at_jmax = float(np.hypot(x[start + jmax] - origin_region.cx,
                                    z[start + jmax] - origin_region.cz))
        if rad_at_jmax >= min_radial:  # NEW
            v = np.array([dx[jmax], dz[jmax]]) / seg[jmax]
            return v

    # T3: windowed net displacement (ignores epsilon threshold)
    w = min(fallback_window, len(dx))
    if w > 0:
        end_i = start + len(dx) - 1
        rad_now = float(np.hypot(x[end_i] - origin_region.cx, z[end_i] - origin_region.cz))
        if rad_now >= min_radial:  # NEW
            ddx = float(np.sum(dx[-w:]))
            ddz = float(np.sum(dz[-w:]))
            n = float(np.hypot(ddx, ddz))
            if n > 0:
                return np.array([ddx / n, ddz / n])

    return None

def first_unit_vector_after_radial_exit(
    x: np.ndarray,
    z: np.ndarray,
    junction: Circle,
    r_outer: float,
    epsilon: float = 0.05,
    window: int = 5,          # default smoothing window
) -> Optional[np.ndarray]:
    """
    Direction when the path *exits* an outer radius around the junction.
    Start at first time inside junction.r (else nearest approach).
    Trigger when r >= r_outer (with non-negative outward trend).
    Direction is the unit vector of the summed step vectors over a short window.
    """
    if len(x) < 2:
        return None

    rx = x - junction.cx
    rz = z - junction.cz
    r  = np.hypot(rx, rz)

    inside = r <= junction.r
    start = int(np.argmax(inside)) if inside.any() else int(np.argmin(r))

    # find the first index crossing r_outer with outward trend
    i_cross = None
    for i in range(start + 1, len(r)):
        if r[i] >= r_outer:
            j0 = max(start + 1, i - window)
            seg = r[j0:i+1]
            # Robust outward-trend test: if we don't have at least 2 samples, accept (avoids "mean of empty slice" warning)
            if seg.size >= 2:
                outward = float(np.nanmean(np.diff(seg))) >= 0.0
            else:
                outward = True
            if outward:
                i_cross = i
                break
    if i_cross is None:
        return None

    # Smooth direction over the last `window` steps ending at i_cross
    j0 = max(start, i_cross - window)
    dx = np.diff(x[j0:i_cross+1])
    dz = np.diff(z[j0:i_cross+1])
    step = np.hypot(dx, dz)

    # Use meaningful steps if available; otherwise fall back to max step in window
    mask = step >= epsilon
    if np.any(mask):
        vx = dx[mask].sum()
        vz = dz[mask].sum()
    else:
        if step.size == 0 or float(np.nanmax(step)) <= 0:
            return None
        k = int(np.nanargmax(step))
        vx, vz = dx[k], dz[k]

    n = float(np.hypot(vx, vz))
    if n == 0:
        return None
    return np.array([vx / n, vz / n])

def _pick_vector_and_source(
    tr: Trajectory,
    junction: Circle,
    decision_mode: str,
    path_length: float,
    r_outer: Optional[float],
    epsilon: float,
    linger_delta: float = 0.0
) -> tuple[Optional[np.ndarray], str]:
    """Return (v, 'radial'|'pathlen') without changing existing APIs."""
    if decision_mode in ("radial", "hybrid"):
        rout = r_outer if (r_outer is not None and r_outer > junction.r) else (junction.r + 10.0)
        v_rad = first_unit_vector_after_radial_exit(tr.x, tr.z, junction, rout, epsilon=epsilon)
        if decision_mode == "radial":
            return v_rad, "radial"
        if v_rad is not None:
            return v_rad, "radial"
        # fallback
        v_pl = first_unit_vector_after_distance(tr.x, tr.z, junction, path_length=path_length, epsilon=epsilon, linger_delta=linger_delta)
        return v_pl, "pathlen"
    # pathlen only
    v_pl = first_unit_vector_after_distance(tr.x, tr.z, junction, path_length=path_length, epsilon=epsilon, linger_delta=linger_delta)
    return v_pl, "pathlen"

def discover_branches(trajectories: Sequence[Trajectory],
                      junction: Circle,
                      k: int = 3,
                      path_length: float = 100.0,
                      epsilon: float = 0.05,
                      seed: int = 0,
                      decision_mode="hybrid",
                      r_outer=None,
                      linger_delta: float = 0.0,
                      out_dir = None,
                      cluster_method: str = "kmeans",
                      k_min: int = 2,
                      k_max: int = 6,
                      min_sep_deg: float = 12.0,
                      angle_eps: float = 15.0,
                      min_samples: int = 5,
                      junction_number: int = 0,
                      all_junctions: Sequence[Circle] = None
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Compute initial movement vectors and cluster them into k branches.

    Returns
    -------
    assignments : DataFrame with columns ["trajectory", "branch"]
    summary     : DataFrame with columns ["branch", "count", "percent"]
    centers     : (k,2) ndarray of unit vectors (sorted by angle)
    """
    from tqdm import tqdm

    # --- Extract one direction vector per trajectory ---
    vecs, ids, diags, mode_log = [], [], [], []
    decisions_rows = []  # persist decision index/coords per trajectory
    assign_all_rows = []  # holds rows for −2 (no entry)

    logger.info(f"Processing {len(trajectories)} trajectories...")

    for tr in tqdm(trajectories, desc="Analyzing trajectories", unit="traj"):
        # 0) Hard "no entry" → branch -2 (kept only in *all* table for plotting/stats)
        entered, _ = entered_junction_idx(tr.x, tr.z, junction)
        if not entered:
            assign_all_rows.append({"trajectory": tr.tid, "branch": -2})
            diags.append({"trajectory": tr.tid, "used": False, "reason": "no_junction_entry"})
            continue

        # 1) Get initial direction according to mode (hybrid tries radial, falls back to pathlen)
        v, mode_used = _pick_vector_and_source(
            tr=tr,
            junction=junction,
            decision_mode=str(decision_mode),
            path_length=float(path_length),
            r_outer=r_outer,
            epsilon=float(epsilon),
            linger_delta=float(linger_delta),
        )
        mode_log.append({"trajectory": tr.tid, "mode_used": mode_used})

        # 2) If we still didn’t get a vector, mark as “entered but off-center” (branch -1 in *all*)
        if v is None:
            assign_all_rows.append({"trajectory": tr.tid, "branch": -1})
            diags.append({"trajectory": tr.tid, "used": False, "reason": "no_vector"})
            continue

        # 3) Normalize and keep
        v = v / max(1e-12, float(np.linalg.norm(v)))
        vecs.append(v)
        ids.append(tr.tid)
        diags.append({"trajectory": tr.tid, "used": True, "reason": mode_used})

        # Persist approximate decision index consistent with mode_used
        try:
            rx = tr.x - junction.cx
            rz = tr.z - junction.cz
            r = np.hypot(rx, rz)
            inside = r <= junction.r
            start = int(np.argmax(inside)) if inside.any() else int(np.argmin(r))

            decision_idx = None
            if mode_used == "radial":
                rout = r_outer if (r_outer is not None and r_outer > junction.r) else (junction.r + 10.0)
                i_cross = None
                for i in range(start + 1, len(r)):
                    if r[i] >= rout:
                        j0 = max(start + 1, i - 5)
                        seg = r[j0:i+1]
                        outward = float(np.nanmean(np.diff(seg))) >= 0.0 if seg.size >= 2 else True
                        if outward:
                            i_cross = i
                            break
                decision_idx = int(i_cross) if i_cross is not None else None
            else:
                dx = np.diff(tr.x[start:])
                dz = np.diff(tr.z[start:])
                seg = np.hypot(dx, dz)
                cum = np.cumsum(seg)
                reach_idx = int(np.argmax(cum >= float(path_length))) if (cum >= float(path_length)).any() else None
                decision_idx = int(start + reach_idx + 1) if reach_idx is not None else None

            if decision_idx is None:
                # Don't assign trajectories without proper decision points
                # Mark as *entered but no decision* (branch -1 in *all*)
                assign_all_rows.append({"trajectory": tr.tid, "branch": -1})
                diags.append({"trajectory": tr.tid, "used": False, "reason": "no_decision_point"})
                continue

            if 0 <= decision_idx < len(tr.x) and 0 <= decision_idx < len(tr.z):
                ix = float(tr.x[decision_idx])
                iz = float(tr.z[decision_idx])

                # CRITICAL VALIDATION: Ensure decision point is not outside r_outer
                decision_distance = np.hypot(ix - junction.cx, iz - junction.cz)
                rout = r_outer if (r_outer is not None and r_outer > junction.r) else (junction.r + 10.0)

                # Calculate adaptive tolerance based on trajectory resolution
                # Use the typical step size in the trajectory as tolerance
                if len(tr.x) > 1:
                    step_sizes = np.hypot(np.diff(tr.x), np.diff(tr.z))
                    typical_step = np.median(step_sizes) if len(step_sizes) > 0 else 1.0
                    tolerance = max(typical_step * 2.0, 1.0)  # At least 2 steps or 1 unit
                else:
                    tolerance = 1.0

                if decision_distance > rout + tolerance:
                    logger.debug(f"REJECTED: Decision point at distance {decision_distance:.1f} is outside r_outer {rout:.1f} (tolerance: {tolerance:.1f}) for trajectory {tr.tid}")
                    continue
            else:
                # Don't use fallback if decision_idx is out of bounds
                # This indicates an invalid decision point
                continue

            decisions_rows.append({
                "trajectory": tr.tid,
                "junction_index": 0,
                "decision_idx": int(decision_idx),
                "intercept_x": ix,
                "intercept_z": iz,
                "mode_used": str(mode_used),
            })
        except Exception:
            pass





    diag_df = pd.DataFrame(diags)
    diag_df.to_csv(os.path.join(out_dir, "discover_diag_reasons.csv"), index=False)

    logger.info(f"trajectories: {len(trajectories)}  extracted_vectors: {len(vecs)}")

    # Optional CSV diagnostics
    if out_dir is not None:
        try:
            pd.DataFrame(diags).to_csv(os.path.join(out_dir, "discover_diagnostics.csv"), index=False)
            logger.debug(f"diagnostics -> {os.path.join(out_dir, 'discover_diagnostics.csv')}")

            pd.DataFrame(mode_log).to_csv(os.path.join(out_dir, "decision_mode_used.csv"), index=False)
            logger.debug(f"decision_mode_used -> {os.path.join(out_dir, 'decision_mode_used.csv')}")
        except Exception as e:
            logger.warning(f"could not write diagnostics: {e}")

    n_noentry = sum(1 for r in assign_all_rows if r["branch"] == -2)
    n_off     = sum(1 for r in assign_all_rows if r["branch"] == -1)
    logger.info(f"entered={len(vecs)+n_off}  no_vector={n_off}  no_entry={n_noentry}")

    if len(vecs) == 0:
        empty_assign = pd.DataFrame(columns=["trajectory", "branch"])
        empty_sum = pd.DataFrame(columns=["branch", "count", "percent"])
        return empty_assign, empty_sum, np.zeros((0, 2))

    V = np.vstack(vecs)

    if out_dir is not None and len(vecs):
        pd.DataFrame({"trajectory": ids, "vx": V[:, 0], "vz": V[:, 1]}).to_csv(os.path.join(out_dir, "vectors.csv"), index=False)

    # ---- CLUSTERING ----
    if cluster_method in ("kmeans", "auto"):
        if cluster_method == "auto":
            k_auto, sil = best_k_by_silhouette(V, k_min=k_min, k_max=k_max, seed=seed)
            logger.debug(f"auto-k silhouette -> {k_auto}  scores={sil}")
            k = k_auto
        if k > len(V):
            logger.debug(f"Requested k={k} but only {len(V)} vectors; capping.")
            k = len(V)

        labels, centers = kmeans_2d(V, k=k, seed=seed)
        logger.debug(f"After kmeans: labels={np.unique(labels)}, centers.shape={centers.shape}")

        # merge near-duplicate directions
        centers, labels = merge_close_centers(centers, labels, min_sep_deg=min_sep_deg)
        logger.debug(f"After merge: labels={np.unique(labels)}, centers.shape={centers.shape}")

        # angle-sort centers; remap labels to 0..C-1
        ang = np.arctan2(centers[:, 1], centers[:, 0])
        order = np.argsort(ang)
        mapping = {old: new for new, old in enumerate(order)}
        logger.debug(f"Angle sort order={order}, mapping={mapping}")

        centers = centers[order]
        nrm = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.clip(nrm, 1e-12, None)
        labels = np.array([mapping[l] for l in labels], dtype=int)
        logger.debug(f"Final labels={np.unique(labels)}")

    elif cluster_method == "dbscan":
        # density on angles; can yield outliers labeled -1
        lab, centers = cluster_angles_dbscan(V, eps_deg=angle_eps, min_samples=min_samples)
        labels = lab.copy()  # keep -1 for outliers

        # If no clusters found, all are outliers; centers is (0,2)
        if centers.size == 0:
            pass  # labels already -1; centers OK
        else:
            # Sort centers by angle for stable numbering
            ang = np.arctan2(centers[:, 1], centers[:, 0])
            order = np.argsort(ang)                  # order gives new IDs
            centers = centers[order]

            # Old DBSCAN cluster ids are 0..C-1 in the order they were built.
            # Build mapping: old_id -> new_id (angle-sorted)
            old_ids = np.arange(len(order))
            remap = {int(old_id): int(new_id) for new_id, old_id in enumerate(order)}

            # Remap labels >= 0; keep -1 as is
            for i, l in enumerate(labels):
                if l >= 0:
                    labels[i] = remap[int(l)]


    else:
        raise ValueError(f"Unknown cluster_method={cluster_method}")

    assignments = pd.DataFrame({"trajectory": ids, "branch": labels})
    assignments_all = pd.concat([assignments, pd.DataFrame(assign_all_rows)], ignore_index=True)

    # Summary (main branches only, >=0) stays the same using *assignments*
    mask_main = assignments["branch"] >= 0
    cnt = Counter(assignments.loc[mask_main, "branch"])
    total = int(mask_main.sum())
    summary = pd.DataFrame({
        "branch": sorted(cnt.keys()),
        "count": [cnt[b] for b in sorted(cnt.keys())],
        "percent": [cnt[b] / total * 100.0 if total else 0.0 for b in sorted(cnt.keys())],
    })

    # Write both CSVs if out_dir given and draw intercepts using the *all* table
    if out_dir is not None:
        assignments.to_csv(os.path.join(out_dir, "branch_assignments.csv"), index=False)
        assignments_all.to_csv(os.path.join(out_dir, "branch_assignments_all.csv"), index=False)
        mode_df = pd.DataFrame(mode_log)
        mode_df.to_csv(os.path.join(out_dir, "decision_mode_used.csv"), index=False)
        # persist decision points for reuse in gaze
        try:
            decision_points_path = os.path.join(out_dir, "decision_points.csv")
            pd.DataFrame(decisions_rows).to_csv(decision_points_path, index=False)
            print(f"[discover_debug] Junction {junction_number}: Saved {len(decisions_rows)} decision points to {decision_points_path}")
        except Exception as e:
            print(f"[discover_debug] Junction {junction_number}: Error saving decision points: {e}")
            pass

        # Ensure r_outer has a proper default value for plotting
        plot_r_outer = r_outer if (r_outer is not None and r_outer > junction.r) else (junction.r + 10.0)

        # Load decision points data for plotting
        decision_points_df = None
        try:
            decision_points_path = os.path.join(out_dir, "decision_points.csv")
            if os.path.exists(decision_points_path):
                decision_points_df = pd.read_csv(decision_points_path)
        except Exception:
            pass

        plot_decision_intercepts(
            trajectories=trajectories,
            assignments_df=assignments_all,
            mode_log_df=mode_df,
            centers=centers,
            junction=junction,
            r_outer=plot_r_outer,
            path_length=path_length,
            epsilon=epsilon,
            linger_delta=linger_delta,
            out_path=os.path.join(out_dir, "Decision_Intercepts.png"),
            show_paths=False,
            junction_number=junction_number,
            all_junctions=all_junctions,
            decision_points_df=decision_points_df
        )

    return assignments, summary, centers


def assign_branches(trajectories,
                    centers: np.ndarray,
                    junction: Circle,
                    path_length: float = 100.0,
                    decision_mode="pathlen",
                    r_outer=None,
                    epsilon: float = 0.05,
                    linger_delta: float = 0.0,
                    assign_angle_eps: float = 15.0,
                    out_dir = None) -> pd.DataFrame:
    """Assign branches using fixed centers, consistent with discover."""
    ids, labs = [], []
    dbg_rows, mode_rows = [], []

    min_dot = float(math.cos(math.radians(assign_angle_eps)))

    for tr in trajectories:
        # hard −2 if we never enter the junction
        entered, _ = entered_junction_idx(tr.x, tr.z, junction)
        if not entered:
            ids.append(tr.tid); labs.append(-2)
            continue

        v, mode_used = _pick_vector_and_source(
            tr, junction, decision_mode, path_length, r_outer, epsilon, linger_delta=linger_delta
        )

        # entered but no usable vector → −1
        if v is None or centers.size == 0:
            ids.append(tr.tid); labs.append(-1)
            continue

        v = v / max(1e-12, np.linalg.norm(v))
        dots = centers @ v
        lab = int(np.argmax(dots))
        # too far from any center?  mark −1
        lab = lab if float(dots[lab]) >= min_dot else -1

        ids.append(tr.tid); labs.append(lab)
        dbg_rows.append({
            "trajectory": tr.tid, "vx": float(v[0]), "vz": float(v[1]),
            "assigned_branch": lab, "argmax_dot": float(dots[lab] if lab>=0 else np.max(dots)),
            "best_alt_branch": int(np.argsort(dots)[-2]) if len(dots) > 1 else -1,
            "best_alt_dot": float(np.sort(dots)[-2]) if len(dots) > 1 else float("nan"),
        })
        mode_rows.append({"trajectory": tr.tid, "mode_used": mode_used})

    df = pd.DataFrame({"trajectory": ids, "branch": labs})
    if out_dir is not None:
        pd.DataFrame(dbg_rows).to_csv(os.path.join(out_dir, "assign_vectors.csv"), index=False)
        pd.DataFrame(mode_rows).to_csv(os.path.join(out_dir, "assign_mode_used.csv"), index=False)
    return df


def compute_assignment_vectors(
    trajectories: Sequence[Trajectory],
    junction: Circle,
    *,
    path_length: float = 100.0,
    decision_mode: str = "pathlen",
    r_outer: Optional[float] = None,
    epsilon: float = 0.05,
    linger_delta: float = 0.0,
) -> pd.DataFrame:
    """Compute initial unit vectors per trajectory using the same logic as assignment.

    Returns a DataFrame with columns:
      - trajectory: trajectory id
      - entered: bool (True if trajectory entered junction)
      - usable: bool (True if a non-None vector was obtained)
      - vx, vz: float components of the unit vector (NaN if not usable)
      - mode_used: str ("radial" or "pathlen") when usable
    """
    rows = []
    for tr in trajectories:
        entered, _ = entered_junction_idx(tr.x, tr.z, junction)
        if not entered:
            rows.append({
                "trajectory": tr.tid,
                "entered": False,
                "usable": False,
                "vx": float("nan"),
                "vz": float("nan"),
                "mode_used": None,
            })
            continue

        v, mode_used = _pick_vector_and_source(
            tr=tr,
            junction=junction,
            decision_mode=str(decision_mode),
            path_length=float(path_length),
            r_outer=r_outer,
            epsilon=float(epsilon),
            linger_delta=float(linger_delta),
        )

        if v is None:
            rows.append({
                "trajectory": tr.tid,
                "entered": True,
                "usable": False,
                "vx": float("nan"),
                "vz": float("nan"),
                "mode_used": None,
            })
            continue

        v = v / max(1e-12, float(np.linalg.norm(v)))
        rows.append({
            "trajectory": tr.tid,
            "entered": True,
            "usable": True,
            "vx": float(v[0]),
            "vz": float(v[1]),
            "mode_used": str(mode_used),
        })

    return pd.DataFrame(rows)


# Multi-junction decision chains

def discover_decision_chain(
    trajectories: Sequence[Trajectory],
    junctions: Sequence[Circle],
    *,
    path_length: float = 100.0,
    epsilon: float = 0.05,
    seed: int = 0,
    decision_mode: str = "hybrid",
    r_outer_list: Optional[Sequence[float]] = None,
    linger_delta: float = 0.0,
    out_dir: Optional[str] = None,
    cluster_method: str = "kmeans",
    k: int = 3,
    k_min: int = 2,
    k_max: int = 6,
    min_sep_deg: float = 12.0,
    angle_eps: float = 15.0,
    min_samples: int = 5,
) -> tuple[pd.DataFrame, list[np.ndarray], pd.DataFrame]:
    """
    Discover branches at multiple junctions (a decision chain).

    Returns
    -------
    chain_df : DataFrame with one row per trajectory and columns:
        - trajectory
        - branch_j0, branch_j1, ... for each junction index
    centers_list : list of (C,2) arrays of unit vectors per junction
    """
    if r_outer_list is None:
        r_outer_list = [None] * len(junctions)
    if len(r_outer_list) != len(junctions):
        raise ValueError("r_outer_list length must match junctions length or be omitted")

    all_centers: list[np.ndarray] = []
    per_j_assign: list[pd.DataFrame] = []
    per_j_decisions: list[pd.DataFrame] = []

    # make an optional folder structure: out_dir/junction_{i}
    def _subdir(i: int) -> Optional[str]:
        if out_dir is None:
            return None
        d = os.path.join(out_dir, f"junction_{i}")
        os.makedirs(d, exist_ok=True)
        return d

    # Run discovery per junction
    for i, (junc, r_out) in enumerate(zip(junctions, r_outer_list)):
        sub_out = _subdir(i)
        assign_i, _summary_i, centers_i = discover_branches(
            trajectories=trajectories,
            junction=junc,
            k=int(k),
            path_length=float(path_length),
            epsilon=float(epsilon),
            seed=int(seed + i),  # perturb seed per stage for stability if needed
            decision_mode=str(decision_mode),
            r_outer=r_out,
            linger_delta=float(linger_delta),
            out_dir=sub_out,
            cluster_method=str(cluster_method),
            k_min=int(k_min),
            k_max=int(k_max),
            min_sep_deg=float(min_sep_deg),
            angle_eps=float(angle_eps),
            min_samples=int(min_samples),
            junction_number=i,  # Pass the correct junction number
            all_junctions=junctions
        )
        # rename branch column to junction index specific
        assign_i = assign_i.rename(columns={"branch": f"branch_j{i}"})
        per_j_assign.append(assign_i)
        all_centers.append(centers_i)

        # read decisions file written by discover_branches for this junction
        sub = _subdir(i)
        if sub is not None:
            p = os.path.join(sub, "decision_points.csv")
            try:
                if os.path.exists(p):
                    dfi = pd.read_csv(p)
                    dfi["junction_index"] = i
                    per_j_decisions.append(dfi)
                    print(f"[discover_debug] Junction {i}: Loaded {len(dfi)} decision points from {p}")
                else:
                    print(f"[discover_debug] Junction {i}: No decision points file found at {p}")
            except Exception as e:
                print(f"[discover_debug] Junction {i}: Error loading decision points: {e}")
                pass
        else:
            print(f"[discover_debug] Junction {i}: No output directory created")

    # Merge per-junction assignments into a wide table
    from functools import reduce
    def _merge(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        return a.merge(b, on="trajectory", how="outer")

    chain_df = reduce(_merge, per_j_assign)

    # Optional: order columns (trajectory first)
    cols = ["trajectory"] + [f"branch_j{i}" for i in range(len(junctions))]
    chain_df = chain_df.reindex(columns=cols)

    if out_dir is not None:
        chain_path = os.path.join(out_dir, "branch_assignments_chain.csv")
        chain_df.to_csv(chain_path, index=False)
        # also write centers per junction
        for i, c in enumerate(all_centers):
            np.save(os.path.join(out_dir, f"branch_centers_j{i}.npy"), c)
    # write consolidated decisions
    decisions_chain_df = pd.concat(per_j_decisions, ignore_index=True) if per_j_decisions else pd.DataFrame(columns=["trajectory","junction_index","decision_idx","intercept_x","intercept_z","mode_used"])
    print(f"[discover_debug] Created chain_decisions DataFrame with {len(decisions_chain_df)} rows")
    print(f"[discover_debug] Junction indices in chain_decisions: {sorted(decisions_chain_df['junction_index'].unique()) if not decisions_chain_df.empty else 'None'}")
    if not decisions_chain_df.empty:
        decisions_chain_df.to_csv(os.path.join(out_dir, "branch_decisions_chain.csv"), index=False)
        print(f"[discover_debug] Saved chain_decisions to {os.path.join(out_dir, 'branch_decisions_chain.csv')}")
    else:
        decisions_chain_df = pd.concat(per_j_decisions, ignore_index=True) if per_j_decisions else pd.DataFrame(columns=["trajectory","junction_index","decision_idx","intercept_x","intercept_z","mode_used"])
        print(f"[discover_debug] Chain_decisions is empty, created empty DataFrame")

    return chain_df, all_centers, decisions_chain_df
