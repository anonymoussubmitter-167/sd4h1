"""Shared gene space construction for cross-species alignment.

Projects two datasets into a common ortholog gene space with per-gene
z-score normalization for cross-species comparability.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from numpy.typing import NDArray

from rosetta.data.ortholog_db import OrthologMapping

logger = logging.getLogger(__name__)


def _get_gene_lookup(adata: AnnData) -> dict[str, int]:
    """Build gene name -> column index lookup.

    If var_names look like Ensembl IDs and a 'gene_symbol' column exists,
    use gene_symbol for matching instead.
    """
    var_names = list(adata.var_names)
    # Check if var_names are Ensembl IDs (start with ENS)
    if len(var_names) > 0 and var_names[0].startswith("ENS") and "gene_symbol" in adata.var.columns:
        symbols = list(adata.var["gene_symbol"].values)
        return {sym: i for i, sym in enumerate(symbols) if sym}
    return {name: i for i, name in enumerate(var_names)}


def build_shared_gene_space(
    adata_source: AnnData,
    adata_target: AnnData,
    ortholog_map: OrthologMapping,
) -> tuple[NDArray, NDArray, list[str]]:
    """Build aligned expression matrices over shared ortholog genes.

    For each ortholog pair (source_gene, target_gene):
    1. Extract expression for that gene from each dataset
    2. Z-score normalize within each species

    Automatically handles Ensembl IDs in var_names by falling back to
    'gene_symbol' column in .var if present.

    Args:
        adata_source: AnnData for source species.
        adata_target: AnnData for target species.
        ortholog_map: Mapping from source to target gene names.

    Returns:
        X_source_shared: (N_source, G_shared) z-scored expression matrix.
        X_target_shared: (N_target, G_shared) z-scored expression matrix.
        shared_genes: List of source gene names for the shared genes.
    """
    source_lookup = _get_gene_lookup(adata_source)
    target_lookup = _get_gene_lookup(adata_target)

    # Find ortholog pairs where both genes are present in their datasets
    shared_pairs: list[tuple[str, str]] = []
    for src_gene, tgt_gene in ortholog_map.forward.items():
        if src_gene in source_lookup and tgt_gene in target_lookup:
            shared_pairs.append((src_gene, tgt_gene))

    if len(shared_pairs) == 0:
        logger.warning("No shared ortholog genes found between datasets")
        n_src = adata_source.n_obs
        n_tgt = adata_target.n_obs
        return np.zeros((n_src, 0), dtype=np.float32), np.zeros((n_tgt, 0), dtype=np.float32), []

    logger.info("Found %d shared ortholog genes", len(shared_pairs))

    source_gene_names = [p[0] for p in shared_pairs]
    target_gene_names = [p[1] for p in shared_pairs]

    # Extract expression matrices for shared genes using column indices
    src_idx = [source_lookup[g] for g in source_gene_names]
    tgt_idx = [target_lookup[g] for g in target_gene_names]
    X_src = _extract_genes_by_idx(adata_source, src_idx)
    X_tgt = _extract_genes_by_idx(adata_target, tgt_idx)

    # Per-gene z-score normalization within each species
    X_src = _zscore_per_gene(X_src)
    X_tgt = _zscore_per_gene(X_tgt)

    return X_src, X_tgt, source_gene_names


def _extract_genes_by_idx(adata: AnnData, gene_idx: list[int]) -> NDArray:
    """Extract expression for specific gene column indices."""
    X = adata.X[:, gene_idx]
    if sp.issparse(X):
        X = X.toarray()
    return np.array(X, dtype=np.float32)


def _zscore_per_gene(X: NDArray) -> NDArray:
    """Z-score normalize each gene (column) independently.

    Genes with zero variance are set to 0.
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0  # avoid division by zero
    return (X - mean) / std
