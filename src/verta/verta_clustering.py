# ------------------------------
# Clustering
# ------------------------------

from typing import Tuple
import numpy as np
import pandas as pd

def kmeans_2d(vectors: np.ndarray, k: int = 3, max_iter: int = 100, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Tiny k-means for 2D vectors. Returns (labels, centers)."""
    if len(vectors) < k:
        raise ValueError("Not enough vectors for requested k")
    rng = np.random.default_rng(seed)
    centers = vectors[rng.choice(len(vectors), size=k, replace=False)].copy()
    labels = np.zeros(len(vectors), dtype=int)
    for _ in range(max_iter):
        # assign
        d = np.linalg.norm(vectors[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        # update
        new_centers = centers.copy()
        for j in range(k):
            pts = vectors[labels == j]
            if len(pts) > 0:
                new_centers[j] = pts.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers

def best_k_by_silhouette(V: np.ndarray, k_min=2, k_max=6, seed=42):
    scores = {}
    for k in range(k_min, k_max + 1):
        if V.shape[0] <= k:
            continue
        labels, centers = kmeans_2d(V, k=k, seed=seed)
        S = _cosine_silhouette_score(V, labels)
        scores[k] = S
    if not scores:
        return min(3, max(1, V.shape[0])), {}
    best_k = max(scores, key=scores.get)
    return best_k, scores

def _cosine_silhouette_score(V: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette score using cosine distance 1 - dot(u,v) for unit vectors.
    Works without sklearn. Returns mean silhouette over all points with a valid cluster.
    """
    if V.size == 0 or labels.size != V.shape[0]:
        return 0.0
    # Precompute dot-similarity matrix (cosine on unit vectors = dot)
    D = V @ V.T  # in [-1,1]; similarity
    # Convert to distance
    dist = 1.0 - D
    n = len(V)
    s_vals = []
    for i in range(n):
        Li = labels[i]
        if Li < 0:  # noise/outlier: skip
            continue
        same = (labels == Li)
        other = (labels != Li) & (labels >= 0)

        # a(i): mean intra-cluster distance (excluding self)
        si = dist[i, same]
        if si.size <= 1:
            a = 0.0
        else:
            a = float((si.sum() - 0.0) / max(1, si.size - 1))

        # b(i): min mean distance to other clusters
        b = None
        for Lj in set(labels[other]):
            mask = (labels == Lj)
            if not mask.any():
                continue
            b_j = float(dist[i, mask].mean())
            b = b_j if b is None else min(b, b_j)
        if b is None:
            # no other clusters; silhouette undefined → 0
            s = 0.0
        else:
            s = 0.0 if (a == b == 0.0) else (b - a) / max(a, b)
        s_vals.append(s)
    return float(np.mean(s_vals)) if s_vals else 0.0

def merge_close_centers(centers: np.ndarray, labels: np.ndarray, min_sep_deg=12.0):
    if centers.shape[0] <= 1: 
        return centers, labels
    ang = np.arctan2(centers[:,1], centers[:,0])
    keep = np.ones(len(centers), dtype=bool)
    map_to = np.arange(len(centers))
    for i in range(len(centers)):
        if not keep[i]: 
            continue
        for j in range(i+1, len(centers)):
            if not keep[j]: 
                continue
            d = np.abs((ang[i]-ang[j]+np.pi)%(2*np.pi)-np.pi)
            if np.degrees(d) < min_sep_deg:
                # merge j -> i
                keep[j] = False
                map_to[map_to == j] = i
    new_ids = {old: idx for idx, old in enumerate(np.where(keep)[0])}
    new_centers = centers[keep]
    new_labels  = np.array([ new_ids[ map_to[l] ] for l in labels ], dtype=int)
    return new_centers, new_labels

def split_small_branches(assign_df: pd.DataFrame, min_frac=0.05):
    # assign_df: columns ["trajectory","branch"]
    counts = assign_df["branch"].value_counts().sort_index()
    n = int(counts.sum())
    small = set(counts[counts < max(1, int(np.ceil(min_frac*n)))].index)
    main  = assign_df[~assign_df["branch"].isin(small)].copy()
    minor = assign_df[ assign_df["branch"].isin(small)].copy()
    
    print(f"[debug] split_small_branches: original branches={sorted(assign_df['branch'].unique())}")
    print(f"[debug] split_small_branches: small branches={sorted(small)}")
    print(f"[debug] split_small_branches: main branches before renumbering={sorted(main['branch'].unique())}")
    
    # Renumber main branches to start from 0
    if len(main) > 0:
        unique_branches = sorted(main["branch"].unique())
        branch_mapping = {old: new for new, old in enumerate(unique_branches)}
        main["branch"] = main["branch"].map(branch_mapping)
        print(f"[debug] split_small_branches: branch_mapping={branch_mapping}")
        print(f"[debug] split_small_branches: main branches after renumbering={sorted(main['branch'].unique())}")
    
    return main, minor, counts

def cluster_angles_dbscan(V: np.ndarray, eps_deg=15.0, min_samples=5):
    """
    Simple DBSCAN on the unit circle without sklearn.
    - Build neighbor graph by chord distance threshold derived from eps_deg.
    - Core points have >= min_samples neighbors (incl. self).
    - Expand clusters via BFS; others labeled -1.
    Returns (labels, centers[unit vectors]).
    """
    if V.size == 0:
        return np.zeros((0,), dtype=int), np.zeros((0, 2), dtype=float)

    # Convert to angle embedding on unit circle (already unit vectors)
    X = V  # (n,2), assumed ~unit
    # chord distance threshold for angular eps:
    # chord = 2*sin(eps/2)
    eps = 2.0 * np.sin(np.deg2rad(eps_deg) / 2.0)

    # pairwise chord distances on unit circle between X[i], X[j]: ||X[i]-X[j]||
    # use (a-b)^2 = a^2 + b^2 - 2 a·b; here a^2=b^2=1 => ||a-b||^2 = 2 - 2(a·b)
    S = X @ X.T  # dot
    sq_chord = 2.0 - 2.0 * S
    sq_eps = eps * eps
    neigh = (sq_chord <= sq_eps)

    n = len(X)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    core = np.sum(neigh, axis=1) >= min_samples

    cid = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if not core[i]:
            continue
        # start new cluster
        labels[i] = cid
        # expand via BFS over density-reachable points
        queue = [i]
        while queue:
            p = queue.pop()
            Np = np.where(neigh[p])[0]
            for q in Np:
                if not visited[q]:
                    visited[q] = True
                    if core[q]:
                        queue.append(q)
                if labels[q] == -1:
                    labels[q] = cid
        cid += 1

    # compute centers as normalized mean of unit vectors per cluster
    centers = []
    for c in range(cid):
        idx = (labels == c)
        m = X[idx].mean(axis=0)
        nrm = np.linalg.norm(m)
        centers.append(m / nrm if nrm > 0 else np.array([1.0, 0.0]))
    centers = np.array(centers) if centers else np.zeros((0, 2))
    return labels, centers
