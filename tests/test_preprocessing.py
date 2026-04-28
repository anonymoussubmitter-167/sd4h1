"""Tests for data preprocessing pipeline."""

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

from rosetta.data.preprocessing import (
    filter_cells,
    filter_genes,
    normalize,
    preprocess_pipeline,
    select_hvg,
)
from rosetta.utils.config import PreprocessingConfig


def make_synthetic_adata(n_cells=200, n_genes=500, sparse=True):
    """Create a synthetic AnnData for testing."""
    rng = np.random.default_rng(42)
    # Simulate count data: negative binomial-like
    X = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
    # Add some zero rows/columns for filtering tests
    X[:3, :] = 0  # 3 cells with no genes
    X[:, :2] = 0  # 2 genes in no cells

    if sparse:
        X = sp.csr_matrix(X)

    adata = AnnData(X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    # Add spatial coords
    coords = rng.uniform(0, 1000, size=(n_cells, 2)).astype(np.float32)
    adata.obsm["spatial"] = coords

    return adata


class TestFilterCells:
    def test_removes_low_gene_cells(self):
        adata = make_synthetic_adata()
        n_before = adata.n_obs
        adata = filter_cells(adata, min_genes=1)
        assert adata.n_obs < n_before

    def test_preserves_valid_cells(self):
        adata = make_synthetic_adata()
        adata = filter_cells(adata, min_genes=1)
        # All remaining cells should have at least 1 gene
        genes_per_cell = np.array((adata.X > 0).sum(axis=1)).flatten()
        assert (genes_per_cell >= 1).all()


class TestFilterGenes:
    def test_removes_rare_genes(self):
        adata = make_synthetic_adata()
        n_genes_before = adata.n_vars
        adata = filter_genes(adata, min_cells=1)
        assert adata.n_vars < n_genes_before

    def test_preserves_expressed_genes(self):
        adata = make_synthetic_adata()
        adata = filter_genes(adata, min_cells=1)
        cells_per_gene = np.array((adata.X > 0).sum(axis=0)).flatten()
        assert (cells_per_gene >= 1).all()


class TestNormalize:
    def test_log_transformed(self):
        adata = make_synthetic_adata()
        adata = normalize(adata, target_sum=1e4)
        # After log1p, max should be reasonable
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        assert X.max() < 20  # log-transformed counts shouldn't be huge


class TestSelectHVG:
    def test_reduces_genes(self):
        adata = make_synthetic_adata(n_cells=200, n_genes=500)
        n_target = 100
        # Need raw counts for seurat_v3
        adata = select_hvg(adata, n_top_genes=n_target)
        assert adata.n_vars == n_target


class TestPreprocessPipeline:
    def test_full_pipeline(self):
        adata = make_synthetic_adata(n_cells=200, n_genes=500)
        config = PreprocessingConfig(
            min_genes=1, min_cells=1, n_top_genes=100, target_sum=1e4
        )
        result = preprocess_pipeline(adata, config)

        # Should have counts layer
        assert "counts" in result.layers

        # Should have reduced genes
        assert result.n_vars == 100

        # Should preserve spatial coords
        assert "spatial" in result.obsm

    def test_operates_on_copy(self):
        adata = make_synthetic_adata()
        config = PreprocessingConfig(min_genes=1, min_cells=1, n_top_genes=100)
        result = preprocess_pipeline(adata, config)
        # Original should be unchanged
        assert adata.n_vars == 500
        assert adata.n_obs == 200
