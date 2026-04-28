"""Joint NMF for cross-species gene module discovery.

Concatenates source and target expression in shared ortholog gene space,
runs NMF to discover gene modules active across both species.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from numpy.typing import NDArray
from sklearn.decomposition import NMF

from rosetta.bridge.shared_space import _extract_genes_by_idx, _get_gene_lookup
from rosetta.data.ortholog_db import OrthologMapping

logger = logging.getLogger(__name__)


def build_nonneg_shared_space(
    adata_source: AnnData,
    adata_target: AnnData,
    ortholog_map: OrthologMapping,
) -> tuple[NDArray, NDArray, list[str]]:
    """Build non-negative shared expression matrices for NMF.

    Like build_shared_gene_space but uses library-size normalization + log1p
    instead of z-score (which produces negatives incompatible with NMF).

    Args:
        adata_source: AnnData for source species.
        adata_target: AnnData for target species.
        ortholog_map: Mapping from source to target gene names.

    Returns:
        X_source: (N_source, G_shared) non-negative expression matrix.
        X_target: (N_target, G_shared) non-negative expression matrix.
        shared_genes: List of source gene names for the shared genes.
    """
    source_lookup = _get_gene_lookup(adata_source)
    target_lookup = _get_gene_lookup(adata_target)

    shared_pairs: list[tuple[str, str]] = []
    for src_gene, tgt_gene in ortholog_map.forward.items():
        if src_gene in source_lookup and tgt_gene in target_lookup:
            shared_pairs.append((src_gene, tgt_gene))

    if len(shared_pairs) == 0:
        logger.warning("No shared ortholog genes found between datasets")
        n_src = adata_source.n_obs
        n_tgt = adata_target.n_obs
        return (
            np.zeros((n_src, 0), dtype=np.float32),
            np.zeros((n_tgt, 0), dtype=np.float32),
            [],
        )

    logger.info("Found %d shared ortholog genes for NMF", len(shared_pairs))

    source_gene_names = [p[0] for p in shared_pairs]
    target_gene_names = [p[1] for p in shared_pairs]

    src_idx = [source_lookup[g] for g in source_gene_names]
    tgt_idx = [target_lookup[g] for g in target_gene_names]
    X_src = _extract_genes_by_idx(adata_source, src_idx)
    X_tgt = _extract_genes_by_idx(adata_target, tgt_idx)

    # Library-size normalization + log1p (non-negative)
    X_src = _libsize_log1p(X_src)
    X_tgt = _libsize_log1p(X_tgt)

    return X_src, X_tgt, source_gene_names


def _libsize_log1p(X: NDArray) -> NDArray:
    """Per-cell library size normalization followed by log1p."""
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0  # avoid division by zero
    X = X / row_sums * 1e4
    return np.log1p(X).astype(np.float32)


def run_joint_nmf(
    X_source: NDArray,
    X_target: NDArray,
    k: int,
    max_iter: int = 500,
    random_state: int = 42,
) -> tuple[NDArray, NDArray, NDArray]:
    """Run joint NMF on concatenated source + target expression.

    Args:
        X_source: (N_source, G) non-negative expression matrix.
        X_target: (N_target, G) non-negative expression matrix.
        k: Number of modules (components).
        max_iter: Maximum NMF iterations.
        random_state: Random seed.

    Returns:
        W_source: (N_source, k) module activity per source spot.
        W_target: (N_target, k) module activity per target spot.
        H: (k, G) gene loadings per module (shared across species).
    """
    n_source = X_source.shape[0]
    X_joint = np.concatenate([X_source, X_target], axis=0)

    model = NMF(
        n_components=k,
        init="nndsvda",
        max_iter=max_iter,
        random_state=random_state,
    )
    W = model.fit_transform(X_joint)
    H = model.components_

    W_source = W[:n_source]
    W_target = W[n_source:]

    return W_source, W_target, H


def consensus_nmf(
    X_source: NDArray,
    X_target: NDArray,
    k: int,
    n_runs: int = 10,
    max_iter: int = 500,
) -> tuple[NDArray, NDArray, NDArray, float]:
    """Run consensus NMF to assess factorization stability.

    Runs NMF n_runs times with different seeds, builds a connectivity matrix
    where C[i,j] = fraction of runs where spots i,j are assigned to the same
    module (by argmax of W). Stability = mean C for co-assigned pairs.

    Returns the run whose connectivity is closest to the consensus.

    Args:
        X_source: (N_source, G) non-negative expression.
        X_target: (N_target, G) non-negative expression.
        k: Number of modules.
        n_runs: Number of NMF runs with different seeds.
        max_iter: Max NMF iterations per run.

    Returns:
        W_source: (N_source, k) best-run module activity.
        W_target: (N_target, k) best-run module activity.
        H: (k, G) best-run gene loadings.
        stability: Consensus stability score in [0, 1].
    """
    n_source = X_source.shape[0]
    X_joint = np.concatenate([X_source, X_target], axis=0)
    n_total = X_joint.shape[0]

    # Collect W matrices and assignments from each run
    all_W: list[NDArray] = []
    all_H: list[NDArray] = []
    all_assignments: list[NDArray] = []

    for run in range(n_runs):
        model = NMF(
            n_components=k,
            init="nndsvda",
            max_iter=max_iter,
            random_state=run * 7 + 42,
        )
        W = model.fit_transform(X_joint)
        H = model.components_
        all_W.append(W)
        all_H.append(H)
        all_assignments.append(np.argmax(W, axis=1))

    # Build consensus connectivity matrix (averaged over runs)
    # C[i,j] = fraction of runs where i and j have the same argmax module
    # For memory efficiency with large N, compute co-assignment counts
    # using vectorized comparison
    assignments_mat = np.stack(all_assignments, axis=0)  # (n_runs, n_total)

    # Compute stability: average pairwise agreement within co-assigned pairs
    # For each pair of runs, fraction of spots with same assignment
    pairwise_agreement = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            agree = np.mean(assignments_mat[i] == assignments_mat[j])
            pairwise_agreement.append(agree)
    stability = float(np.mean(pairwise_agreement)) if pairwise_agreement else 1.0

    # Select best run: highest mean agreement with all other runs
    mean_agreement_per_run = np.zeros(n_runs)
    for i in range(n_runs):
        agreements = []
        for j in range(n_runs):
            if i != j:
                agreements.append(np.mean(assignments_mat[i] == assignments_mat[j]))
        mean_agreement_per_run[i] = np.mean(agreements)

    best_run = int(np.argmax(mean_agreement_per_run))
    W_best = all_W[best_run]
    H_best = all_H[best_run]

    W_source = W_best[:n_source]
    W_target = W_best[n_source:]

    logger.info(
        "Consensus NMF (k=%d, n_runs=%d): stability=%.3f, best_run=%d",
        k, n_runs, stability, best_run,
    )

    return W_source, W_target, H_best, stability


def characterize_modules(
    H: NDArray,
    gene_names: list[str],
    top_n: int = 20,
) -> list[dict]:
    """Characterize each NMF module by its top contributing genes.

    Args:
        H: (k, G) gene loadings matrix.
        gene_names: List of G gene names.
        top_n: Number of top genes per module.

    Returns:
        List of dicts, one per module, with keys:
            'module': int, 'genes': list[str], 'weights': list[float]
    """
    modules = []
    for m in range(H.shape[0]):
        loadings = H[m]
        top_idx = np.argsort(loadings)[::-1][:top_n]
        modules.append({
            "module": m,
            "genes": [gene_names[i] for i in top_idx],
            "weights": [float(loadings[i]) for i in top_idx],
        })
    return modules
