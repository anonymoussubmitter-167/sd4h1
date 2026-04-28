"""Conservation scoring for cross-species gene modules.

Measures whether gene modules discovered by joint NMF are conserved
across species using transport plans and aligned embeddings from BRIDGE.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.stats import kruskal, pearsonr, spearmanr
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def module_conservation_transport(
    W_source: NDArray,
    W_target: NDArray,
    transport_plan: NDArray,
) -> NDArray:
    """Compute per-module conservation score via transport plan projection.

    Projects source module activity through the transport plan to predict
    what source activity should look like if modules are conserved, then
    measures Pearson correlation between actual and projected activity.

    Args:
        W_source: (n_s, k) module activity per source spot.
        W_target: (n_t, k) module activity per target spot.
        transport_plan: (n_s, n_t) optimal transport plan.

    Returns:
        conservation: (k,) Pearson correlations per module in [-1, 1].
    """
    k = W_source.shape[1]

    # Normalize transport plan rows to get conditional distribution T[i,j] = P(j|i)
    row_sums = transport_plan.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    T_norm = transport_plan / row_sums

    # Project: predicted source activity = T_norm @ W_target
    W_source_projected = T_norm @ W_target  # (n_s, k)

    conservation = np.zeros(k, dtype=np.float64)
    for m in range(k):
        actual = W_source[:, m]
        predicted = W_source_projected[:, m]
        if np.std(actual) < 1e-12 or np.std(predicted) < 1e-12:
            conservation[m] = 0.0
        else:
            conservation[m] = pearsonr(actual, predicted)[0]

    return conservation


def module_conservation_knn(
    W_source: NDArray,
    W_target: NDArray,
    z_source: NDArray,
    z_target: NDArray,
    k: int = 10,
) -> NDArray:
    """Compute per-module conservation via kNN in aligned embedding space.

    For each source spot, finds k nearest target neighbors in embedding
    space and compares module activity. Works on all spots (not limited
    to transport plan size).

    Args:
        W_source: (n_s, k_modules) module activity per source spot.
        W_target: (n_t, k_modules) module activity per target spot.
        z_source: (n_s, D) source projected embeddings.
        z_target: (n_t, D) target projected embeddings.
        k: Number of nearest neighbors.

    Returns:
        conservation: (k_modules,) mean correlation per module.
    """
    n_modules = W_source.shape[1]
    k_actual = min(k, z_target.shape[0])

    tree = cKDTree(z_target)
    _, indices = tree.query(z_source, k=k_actual)  # (n_s, k)

    # Mean target activity among kNN for each source spot
    neighbor_W = W_target[indices]  # (n_s, k, n_modules)
    W_target_knn_mean = neighbor_W.mean(axis=1)  # (n_s, n_modules)

    conservation = np.zeros(n_modules, dtype=np.float64)
    for m in range(n_modules):
        actual = W_source[:, m]
        predicted = W_target_knn_mean[:, m]
        if np.std(actual) < 1e-12 or np.std(predicted) < 1e-12:
            conservation[m] = 0.0
        else:
            conservation[m] = pearsonr(actual, predicted)[0]

    return conservation


def module_celltype_enrichment(
    W: NDArray,
    labels: NDArray,
) -> pd.DataFrame:
    """Test whether modules are differentially active across cell types.

    Uses Kruskal-Wallis test (non-parametric) per module.

    Args:
        W: (N, k) module activity matrix.
        labels: (N,) cell type labels.

    Returns:
        DataFrame with columns: module, cell_type, H_statistic, p_value,
        mean_activity.
    """
    unique_labels = np.unique(labels)
    k = W.shape[1]
    rows = []

    for m in range(k):
        activity = W[:, m]
        groups = [activity[labels == ct] for ct in unique_labels]

        # Kruskal-Wallis across all cell types for this module
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            try:
                h_stat, p_val = kruskal(*groups)
            except ValueError:
                h_stat, p_val = 0.0, 1.0
        else:
            h_stat, p_val = 0.0, 1.0

        for ct_idx, ct in enumerate(unique_labels):
            mask = labels == ct
            rows.append({
                "module": m,
                "cell_type": ct,
                "H_statistic": float(h_stat),
                "p_value": float(p_val),
                "mean_activity": float(activity[mask].mean()) if mask.any() else 0.0,
            })

    return pd.DataFrame(rows)


def module_spatial_autocorrelation(
    W: NDArray,
    coords: NDArray,
    k: int = 10,
) -> NDArray:
    """Compute Moran's I spatial autocorrelation per module.

    Args:
        W: (N, n_modules) module activity matrix.
        coords: (N, 2) spatial coordinates.
        k: Number of spatial neighbors.

    Returns:
        morans: (n_modules,) Moran's I per module.
    """
    n = W.shape[0]
    n_modules = W.shape[1]

    if n < k + 1:
        return np.zeros(n_modules, dtype=np.float64)

    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    indices = indices[:, 1:]  # remove self

    morans = np.zeros(n_modules, dtype=np.float64)
    for m in range(n_modules):
        x = W[:, m]
        x_mean = x.mean()
        x_dev = x - x_mean
        ss = np.sum(x_dev ** 2)

        if ss < 1e-12:
            morans[m] = 0.0
            continue

        neighbor_devs = x_dev[indices]  # (N, k)
        cross = x_dev[:, np.newaxis] * neighbor_devs  # (N, k)
        w_sum = cross.sum()
        total_w = n * k

        morans[m] = (n / total_w) * (w_sum / ss)

    return morans


def module_conservation_region(
    W_source: NDArray,
    W_target: NDArray,
    coords_source: NDArray,
    coords_target: NDArray,
    z_source: NDArray,
    z_target: NDArray,
    n_regions: int = 20,
    k_nn: int = 50,
) -> NDArray:
    """Compute region-level conservation by clustering spots into spatial domains.

    Instead of comparing individual spot activities (too noisy), clusters spots
    into spatial domains via KMeans on coordinates, computes domain-level mean
    module activity, then matches source and target domains via kNN in aligned
    embedding space and correlates their module activity profiles.

    Args:
        W_source: (n_s, k) module activity per source spot.
        W_target: (n_t, k) module activity per target spot.
        coords_source: (n_s, 2) source spatial coordinates.
        coords_target: (n_t, 2) target spatial coordinates.
        z_source: (n_s, D) source aligned embeddings.
        z_target: (n_t, D) target aligned embeddings.
        n_regions: Number of spatial regions per species.
        k_nn: Number of nearest neighbors for cross-species domain matching.

    Returns:
        conservation: (k,) Spearman correlation per module at region level.
    """
    n_modules = W_source.shape[1]

    # Cluster each species into spatial regions
    km_source = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
    km_target = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
    labels_s = km_source.fit_predict(coords_source)
    labels_t = km_target.fit_predict(coords_target)

    # Compute region-level mean module activity and mean embedding
    region_W_source = np.zeros((n_regions, n_modules))
    region_z_source = np.zeros((n_regions, z_source.shape[1]))
    for r in range(n_regions):
        mask = labels_s == r
        if mask.any():
            region_W_source[r] = W_source[mask].mean(axis=0)
            region_z_source[r] = z_source[mask].mean(axis=0)

    region_W_target = np.zeros((n_regions, n_modules))
    region_z_target = np.zeros((n_regions, z_target.shape[1]))
    for r in range(n_regions):
        mask = labels_t == r
        if mask.any():
            region_W_target[r] = W_target[mask].mean(axis=0)
            region_z_target[r] = z_target[mask].mean(axis=0)

    # Match source regions to target regions via kNN in embedding space
    tree = cKDTree(region_z_target)
    k_actual = min(5, n_regions)
    _, indices = tree.query(region_z_source, k=k_actual)  # (n_regions, k_actual)

    # For each source region, predicted activity = mean of matched target regions
    matched_W_target = region_W_target[indices].mean(axis=1)  # (n_regions, n_modules)

    # Spearman correlation per module across regions
    conservation = np.zeros(n_modules, dtype=np.float64)
    for m in range(n_modules):
        actual = region_W_source[:, m]
        predicted = matched_W_target[:, m]
        if np.std(actual) < 1e-12 or np.std(predicted) < 1e-12:
            conservation[m] = 0.0
        else:
            conservation[m] = spearmanr(actual, predicted)[0]

    return conservation


def match_independent_modules(
    H_source: NDArray,
    H_target: NDArray,
    W_source: NDArray | None = None,
    W_target: NDArray | None = None,
    z_source: NDArray | None = None,
    z_target: NDArray | None = None,
) -> dict:
    """Match independently-discovered NMF modules across species.

    Runs after NMF has been fit independently on each species. Matches modules
    by gene loading correlation (H matrices over shared genes) and optionally
    by embedding-space activity correlation.

    Args:
        H_source: (k, G) gene loadings from source NMF.
        H_target: (k, G) gene loadings from target NMF.
        W_source: (n_s, k) optional source module activity.
        W_target: (n_t, k) optional target module activity.
        z_source: (n_s, D) optional source aligned embeddings.
        z_target: (n_t, D) optional target aligned embeddings.

    Returns:
        dict with keys:
            'matches': list of (source_module, target_module, gene_corr) tuples
            'mean_gene_corr': float, mean gene loading correlation of matched pairs
            'corr_matrix': (k_s, k_t) full correlation matrix
            'embedding_corr': float or None, mean activity correlation in embedding space
    """
    k_s = H_source.shape[0]
    k_t = H_target.shape[0]

    # Gene loading correlation matrix
    corr_matrix = np.zeros((k_s, k_t), dtype=np.float64)
    for i in range(k_s):
        for j in range(k_t):
            if np.std(H_source[i]) < 1e-12 or np.std(H_target[j]) < 1e-12:
                corr_matrix[i, j] = 0.0
            else:
                corr_matrix[i, j] = pearsonr(H_source[i], H_target[j])[0]

    # Greedy matching: assign each source module to best unmatched target module
    matches = []
    used_t = set()
    remaining = list(range(k_s))

    # Sort source modules by their best available match (descending)
    for _ in range(min(k_s, k_t)):
        best_i, best_j, best_corr = -1, -1, -np.inf
        for i in remaining:
            for j in range(k_t):
                if j not in used_t and corr_matrix[i, j] > best_corr:
                    best_corr = corr_matrix[i, j]
                    best_i, best_j = i, j
        if best_i >= 0:
            matches.append((best_i, best_j, float(best_corr)))
            used_t.add(best_j)
            remaining.remove(best_i)

    mean_gene_corr = float(np.mean([m[2] for m in matches])) if matches else 0.0

    result = {
        "matches": matches,
        "mean_gene_corr": mean_gene_corr,
        "corr_matrix": corr_matrix,
        "embedding_corr": None,
    }

    # Optional: also compare activity in aligned embedding space
    if (W_source is not None and W_target is not None
            and z_source is not None and z_target is not None):
        tree = cKDTree(z_target)
        k_nn = min(10, z_target.shape[0])
        _, indices = tree.query(z_source, k=k_nn)
        neighbor_W = W_target[indices].mean(axis=1)  # (n_s, k_t)

        emb_corrs = []
        for src_m, tgt_m, _ in matches:
            actual = W_source[:, src_m]
            predicted = neighbor_W[:, tgt_m]
            if np.std(actual) < 1e-12 or np.std(predicted) < 1e-12:
                emb_corrs.append(0.0)
            else:
                emb_corrs.append(float(pearsonr(actual, predicted)[0]))

        result["embedding_corr"] = float(np.mean(emb_corrs))

    return result
