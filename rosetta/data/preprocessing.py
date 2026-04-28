"""Preprocessing pipeline for spatial transcriptomics data wrapping scanpy."""

from __future__ import annotations

import scanpy as sc
from anndata import AnnData

from rosetta.utils.config import PreprocessingConfig


def filter_cells(adata: AnnData, min_genes: int = 200) -> AnnData:
    """Filter cells with fewer than min_genes expressed genes."""
    sc.pp.filter_cells(adata, min_genes=min_genes)
    return adata


def filter_genes(adata: AnnData, min_cells: int = 3) -> AnnData:
    """Filter genes expressed in fewer than min_cells cells."""
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata


def normalize(adata: AnnData, target_sum: float = 1e4) -> AnnData:
    """Normalize total counts per cell and log-transform."""
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def select_hvg(adata: AnnData, n_top_genes: int = 2000) -> AnnData:
    """Select highly variable genes."""
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata


def preprocess_pipeline(
    adata: AnnData,
    config: PreprocessingConfig | None = None,
) -> AnnData:
    """Full preprocessing pipeline: filter, normalize, HVG selection.

    Operates on a copy and stores raw counts in adata.layers["counts"].
    """
    if config is None:
        config = PreprocessingConfig()

    adata = adata.copy()

    # Store raw counts before any processing
    adata.layers["counts"] = adata.X.copy()

    adata = filter_cells(adata, min_genes=config.min_genes)
    adata = filter_genes(adata, min_cells=config.min_cells)
    normalize(adata, target_sum=config.target_sum)
    # select_hvg with seurat_v3 needs raw counts
    adata_hvg = adata.copy()
    adata_hvg.X = adata_hvg.layers["counts"].copy()
    sc.pp.highly_variable_genes(
        adata_hvg, n_top_genes=config.n_top_genes, flavor="seurat_v3"
    )
    adata.var["highly_variable"] = adata_hvg.var["highly_variable"]
    adata = adata[:, adata.var["highly_variable"]].copy()

    return adata
