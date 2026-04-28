"""Tests for COMPASS conserved gene module discovery."""

import numpy as np
import pandas as pd
import pytest

from rosetta.compass.conservation import (
    match_independent_modules,
    module_celltype_enrichment,
    module_conservation_knn,
    module_conservation_region,
    module_conservation_transport,
    module_spatial_autocorrelation,
)
from rosetta.compass.nmf import (
    build_nonneg_shared_space,
    characterize_modules,
    consensus_nmf,
    run_joint_nmf,
)
from rosetta.data.ortholog_db import OrthologMapping


class TestJointNMF:
    """Test joint NMF factorization."""

    def test_run_joint_nmf_shapes(self):
        """Random non-negative data -> check W_s, W_t, H shapes match."""
        rng = np.random.default_rng(42)
        n_s, n_t, g, k = 100, 80, 50, 5
        X_s = rng.random((n_s, g)).astype(np.float32)
        X_t = rng.random((n_t, g)).astype(np.float32)

        W_s, W_t, H = run_joint_nmf(X_s, X_t, k=k)

        assert W_s.shape == (n_s, k)
        assert W_t.shape == (n_t, k)
        assert H.shape == (k, g)
        assert (W_s >= 0).all()
        assert (W_t >= 0).all()
        assert (H >= 0).all()

    def test_run_joint_nmf_reconstruction(self):
        """Verify W @ H approximately reconstructs X."""
        rng = np.random.default_rng(42)
        n_s, n_t, g, k = 50, 40, 30, 3
        X_s = rng.random((n_s, g)).astype(np.float32)
        X_t = rng.random((n_t, g)).astype(np.float32)

        W_s, W_t, H = run_joint_nmf(X_s, X_t, k=k, max_iter=1000)

        X_s_recon = W_s @ H
        X_t_recon = W_t @ H

        # Reconstruction error should be reasonable (not perfect with k < g)
        error_s = np.mean((X_s - X_s_recon) ** 2)
        error_t = np.mean((X_t - X_t_recon) ** 2)
        # Just check it's finite and not enormous
        assert error_s < np.mean(X_s ** 2)  # better than predicting zeros
        assert error_t < np.mean(X_t ** 2)

    def test_consensus_nmf_stability(self):
        """Block-diagonal data -> stability should be high."""
        rng = np.random.default_rng(42)
        n_per_block = 30
        g = 20
        k = 3

        # Create block-diagonal structure: 3 blocks, each active in different genes
        X_s = np.zeros((n_per_block * k, g), dtype=np.float32)
        X_t = np.zeros((n_per_block * k, g), dtype=np.float32)
        genes_per_block = g // k
        for b in range(k):
            rows = slice(b * n_per_block, (b + 1) * n_per_block)
            cols = slice(b * genes_per_block, (b + 1) * genes_per_block)
            X_s[rows, cols] = rng.random((n_per_block, genes_per_block)) + 1.0
            X_t[rows, cols] = rng.random((n_per_block, genes_per_block)) + 1.0

        W_s, W_t, H, stability = consensus_nmf(X_s, X_t, k=k, n_runs=5)

        assert stability > 0.7, f"Stability should be high for block data, got {stability}"
        assert W_s.shape == (n_per_block * k, k)
        assert W_t.shape == (n_per_block * k, k)


class TestConservation:
    """Test conservation scoring functions."""

    def test_module_conservation_transport_perfect(self):
        """Identical W_s, W_t with identity-like transport -> conservation ~1.0."""
        rng = np.random.default_rng(42)
        n, k = 50, 3
        W = rng.random((n, k)).astype(np.float32)

        # Identity-like transport (diagonal)
        T = np.eye(n, dtype=np.float64) / n

        conservation = module_conservation_transport(W, W.copy(), T)

        assert conservation.shape == (k,)
        for m in range(k):
            assert conservation[m] > 0.99, (
                f"Module {m} conservation={conservation[m]}, expected ~1.0"
            )

    def test_module_conservation_knn_shape(self):
        """Check output shape = (k,)."""
        rng = np.random.default_rng(42)
        n_s, n_t, d, k = 100, 80, 16, 5
        W_s = rng.random((n_s, k)).astype(np.float32)
        W_t = rng.random((n_t, k)).astype(np.float32)
        z_s = rng.standard_normal((n_s, d)).astype(np.float32)
        z_t = rng.standard_normal((n_t, d)).astype(np.float32)

        conservation = module_conservation_knn(W_s, W_t, z_s, z_t, k=10)

        assert conservation.shape == (k,)
        # Values should be in [-1, 1]
        assert np.all(conservation >= -1.0 - 1e-6)
        assert np.all(conservation <= 1.0 + 1e-6)

    def test_module_celltype_enrichment_shape(self):
        """Check DataFrame columns and row count."""
        rng = np.random.default_rng(42)
        n, k = 100, 4
        W = rng.random((n, k)).astype(np.float32)
        labels = np.array(["A"] * 40 + ["B"] * 30 + ["C"] * 30)

        df = module_celltype_enrichment(W, labels)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"module", "cell_type", "H_statistic", "p_value", "mean_activity"}
        # k modules * 3 cell types = 12 rows
        assert len(df) == k * 3

    def test_module_spatial_autocorrelation_spatially_structured(self):
        """Spatially structured module should have positive Moran's I."""
        side = 14
        n = side * side  # 196
        xs = np.repeat(np.arange(side), side)
        ys = np.tile(np.arange(side), side)
        coords = np.stack([xs, ys], axis=1).astype(np.float32)

        # Module 0: spatially smooth (left-right gradient)
        # Module 1: random noise
        W = np.zeros((n, 2), dtype=np.float32)
        W[:, 0] = coords[:, 0].astype(np.float32)  # gradient
        rng = np.random.default_rng(42)
        W[:, 1] = rng.random(n).astype(np.float32)

        morans = module_spatial_autocorrelation(W, coords, k=6)

        assert morans.shape == (2,)
        assert morans[0] > morans[1], (
            f"Spatial gradient should have higher Moran's I: {morans[0]:.3f} vs {morans[1]:.3f}"
        )
        assert morans[0] > 0.1, f"Gradient Moran's I should be > 0.1, got {morans[0]:.3f}"


    def test_module_conservation_region_shape(self):
        """Check region-level conservation output shape."""
        rng = np.random.default_rng(42)
        n_s, n_t, d, n_modules = 200, 150, 16, 4
        W_s = rng.random((n_s, n_modules)).astype(np.float32)
        W_t = rng.random((n_t, n_modules)).astype(np.float32)
        coords_s = rng.random((n_s, 2)).astype(np.float32) * 100
        coords_t = rng.random((n_t, 2)).astype(np.float32) * 100
        z_s = rng.standard_normal((n_s, d)).astype(np.float32)
        z_t = rng.standard_normal((n_t, d)).astype(np.float32)

        conservation = module_conservation_region(
            W_s, W_t, coords_s, coords_t, z_s, z_t, n_regions=10,
        )

        assert conservation.shape == (n_modules,)
        assert np.all(conservation >= -1.0 - 1e-6)
        assert np.all(conservation <= 1.0 + 1e-6)

    def test_module_conservation_region_aligned(self):
        """Aligned embeddings with correlated activity -> positive region conservation."""
        rng = np.random.default_rng(42)
        n, d, n_modules = 200, 16, 3

        # Create embeddings that are well-aligned (similar for corresponding spots)
        z_base = rng.standard_normal((n, d)).astype(np.float32)
        z_s = z_base + rng.standard_normal((n, d)).astype(np.float32) * 0.1
        z_t = z_base + rng.standard_normal((n, d)).astype(np.float32) * 0.1

        # Create module activity correlated with spatial position
        coords = rng.random((n, 2)).astype(np.float32) * 100
        W = np.column_stack([
            coords[:, 0],  # gradient in x
            coords[:, 1],  # gradient in y
            np.sin(coords[:, 0] * 0.1),
        ]).astype(np.float32)

        conservation = module_conservation_region(
            W, W.copy(), coords, coords.copy(), z_s, z_t, n_regions=10,
        )

        # With aligned embeddings and identical activity, should be positive
        assert np.mean(conservation) > 0.0

    def test_match_independent_modules_shape(self):
        """Check independent module matching output structure."""
        rng = np.random.default_rng(42)
        k, g = 5, 50
        H_s = rng.random((k, g)).astype(np.float32)
        H_t = rng.random((k, g)).astype(np.float32)

        result = match_independent_modules(H_s, H_t)

        assert "matches" in result
        assert "mean_gene_corr" in result
        assert "corr_matrix" in result
        assert len(result["matches"]) == k
        assert result["corr_matrix"].shape == (k, k)
        # Each match is (source_idx, target_idx, correlation)
        for src, tgt, corr in result["matches"]:
            assert 0 <= src < k
            assert 0 <= tgt < k
            assert -1.0 <= corr <= 1.0

    def test_match_independent_modules_identical(self):
        """Identical H matrices -> perfect matching."""
        rng = np.random.default_rng(42)
        k, g = 4, 30
        H = rng.random((k, g)).astype(np.float32) + 0.1  # ensure non-zero

        result = match_independent_modules(H, H.copy())

        assert result["mean_gene_corr"] > 0.99

    def test_match_independent_modules_with_embeddings(self):
        """Test that embedding-based correlation is computed when provided."""
        rng = np.random.default_rng(42)
        k, g, n_s, n_t, d = 3, 20, 100, 80, 16
        H_s = rng.random((k, g)).astype(np.float32)
        H_t = rng.random((k, g)).astype(np.float32)
        W_s = rng.random((n_s, k)).astype(np.float32)
        W_t = rng.random((n_t, k)).astype(np.float32)
        z_s = rng.standard_normal((n_s, d)).astype(np.float32)
        z_t = rng.standard_normal((n_t, d)).astype(np.float32)

        result = match_independent_modules(H_s, H_t, W_s, W_t, z_s, z_t)

        assert result["embedding_corr"] is not None
        assert isinstance(result["embedding_corr"], float)


class TestCharacterizeModules:
    """Test module characterization."""

    def test_characterize_modules_top_genes(self):
        """Check returns correct number of genes, sorted by weight."""
        rng = np.random.default_rng(42)
        k, g = 3, 50
        H = rng.random((k, g)).astype(np.float32)
        gene_names = [f"gene_{i}" for i in range(g)]

        modules_info = characterize_modules(H, gene_names, top_n=10)

        assert len(modules_info) == k
        for info in modules_info:
            assert len(info["genes"]) == 10
            assert len(info["weights"]) == 10
            # Weights should be sorted descending
            for i in range(len(info["weights"]) - 1):
                assert info["weights"][i] >= info["weights"][i + 1]


class TestBuildNonnegSharedSpace:
    """Test non-negative shared space construction."""

    def _make_adata(self, n_cells, gene_names):
        import anndata as ad
        rng = np.random.default_rng(42)
        X = rng.poisson(lam=5, size=(n_cells, len(gene_names))).astype(np.float32)
        adata = ad.AnnData(X)
        adata.var_names = gene_names
        return adata

    def test_build_nonneg_shared_space_nonnegative(self):
        """Verify output has no negative values."""
        source_genes = ["TP53", "BRCA1", "EGFR", "SRC_ONLY"]
        target_genes = ["Trp53", "Brca1", "Egfr", "TGT_ONLY"]

        adata_s = self._make_adata(100, source_genes)
        adata_t = self._make_adata(80, target_genes)

        ortholog_map = OrthologMapping(
            source_species="human",
            target_species="mouse",
            forward={"TP53": "Trp53", "BRCA1": "Brca1", "EGFR": "Egfr"},
            reverse={"Trp53": "TP53", "Brca1": "BRCA1", "Egfr": "EGFR"},
        )

        X_src, X_tgt, genes = build_nonneg_shared_space(adata_s, adata_t, ortholog_map)

        assert X_src.shape == (100, 3)
        assert X_tgt.shape == (80, 3)
        assert len(genes) == 3
        assert (X_src >= 0).all(), "Source matrix has negative values"
        assert (X_tgt >= 0).all(), "Target matrix has negative values"
