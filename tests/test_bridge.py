"""Tests for BRIDGE cross-species alignment module."""

import numpy as np
import pytest
import torch

from rosetta.bridge.alignment import (
    FGWAligner,
    barycentric_projection,
    subsample_for_fgw,
)
from rosetta.bridge.losses import CrossSpeciesContrastiveLoss, FGWAlignmentLoss, MMDLoss
from rosetta.bridge.shared_space import _zscore_per_gene, build_shared_gene_space
from rosetta.bridge.training import SharedProjectionHead
from rosetta.data.ortholog_db import OrthologMapping


class TestSharedGeneSpace:
    """Test shared ortholog gene space construction."""

    def _make_adata(self, n_cells, gene_names):
        """Create a simple AnnData with given gene names."""
        import anndata as ad

        rng = np.random.default_rng(42)
        X = rng.poisson(lam=5, size=(n_cells, len(gene_names))).astype(np.float32)
        adata = ad.AnnData(X)
        adata.var_names = gene_names
        return adata

    def test_basic_shared_space(self):
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

        X_src, X_tgt, genes = build_shared_gene_space(adata_s, adata_t, ortholog_map)

        assert X_src.shape == (100, 3)
        assert X_tgt.shape == (80, 3)
        assert len(genes) == 3
        assert set(genes) == {"TP53", "BRCA1", "EGFR"}

    def test_no_shared_genes(self):
        adata_s = self._make_adata(50, ["GENE_A", "GENE_B"])
        adata_t = self._make_adata(30, ["GENE_C", "GENE_D"])

        ortholog_map = OrthologMapping(
            source_species="human",
            target_species="mouse",
            forward={"GENE_A": "GENE_X"},  # GENE_X not in target
        )

        X_src, X_tgt, genes = build_shared_gene_space(adata_s, adata_t, ortholog_map)
        assert X_src.shape == (50, 0)
        assert X_tgt.shape == (30, 0)
        assert len(genes) == 0

    def test_zscore_normalization(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        X_z = _zscore_per_gene(X)
        # Each column should have mean ~0, std ~1
        assert np.abs(X_z.mean(axis=0)).max() < 1e-5
        assert np.abs(X_z.std(axis=0) - 1.0).max() < 1e-5

    def test_zscore_constant_column(self):
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]], dtype=np.float32)
        X_z = _zscore_per_gene(X)
        # Constant column should be all zeros
        assert np.all(X_z[:, 0] == 0.0)


class TestFGWAligner:
    """Test Fused Gromov-Wasserstein alignment."""

    def test_transport_plan_shape(self):
        rng = np.random.default_rng(42)
        n_s, n_t, d = 20, 15, 8
        z_s = rng.standard_normal((n_s, d))
        z_t = rng.standard_normal((n_t, d))
        coords_s = rng.uniform(0, 100, (n_s, 2))
        coords_t = rng.uniform(0, 100, (n_t, 2))

        aligner = FGWAligner(alpha=0.5, epsilon=0.05, max_iter=10)
        T, dist = aligner.compute_transport_plan(z_s, z_t, coords_s, coords_t)

        assert T.shape == (n_s, n_t)
        assert isinstance(dist, float)

    def test_transport_plan_valid_coupling(self):
        """Transport plan should be a valid coupling (marginals match)."""
        rng = np.random.default_rng(42)
        n_s, n_t = 10, 12
        z_s = rng.standard_normal((n_s, 4))
        z_t = rng.standard_normal((n_t, 4))
        coords_s = rng.uniform(0, 10, (n_s, 2))
        coords_t = rng.uniform(0, 10, (n_t, 2))

        aligner = FGWAligner(alpha=0.3, epsilon=0.05, max_iter=50)
        T, _ = aligner.compute_transport_plan(z_s, z_t, coords_s, coords_t)

        # All entries non-negative
        assert (T >= -1e-10).all()

        # Row sums should be ~1/n_s, column sums ~1/n_t
        row_sums = T.sum(axis=1)
        col_sums = T.sum(axis=0)
        assert np.allclose(row_sums, 1.0 / n_s, atol=1e-4)
        assert np.allclose(col_sums, 1.0 / n_t, atol=1e-4)

    def test_identical_distributions_low_distance(self):
        """Identical distributions should yield low FGW distance."""
        rng = np.random.default_rng(42)
        n, d = 15, 4
        z = rng.standard_normal((n, d))
        coords = rng.uniform(0, 10, (n, 2))

        aligner = FGWAligner(alpha=0.5, epsilon=0.05, max_iter=50)
        _, dist_same = aligner.compute_transport_plan(z, z.copy(), coords, coords.copy())

        # Different distributions
        z_diff = rng.standard_normal((n, d)) + 10
        coords_diff = rng.uniform(100, 200, (n, 2))
        _, dist_diff = aligner.compute_transport_plan(z, z_diff, coords, coords_diff)

        assert dist_same < dist_diff

    def test_exact_fgw(self):
        """Test exact FGW (epsilon=0)."""
        rng = np.random.default_rng(42)
        n_s, n_t = 8, 10
        z_s = rng.standard_normal((n_s, 4))
        z_t = rng.standard_normal((n_t, 4))
        coords_s = rng.uniform(0, 10, (n_s, 2))
        coords_t = rng.uniform(0, 10, (n_t, 2))

        aligner = FGWAligner(alpha=0.3, epsilon=0.0, max_iter=50)
        T, dist = aligner.compute_transport_plan(z_s, z_t, coords_s, coords_t)

        assert T.shape == (n_s, n_t)
        assert (T >= -1e-10).all()


class TestBarycentricProjection:
    """Test barycentric projection."""

    def test_identity_transport(self):
        """With identity transport, projection should return the same embeddings."""
        n, d = 10, 4
        z = np.random.randn(n, d)
        T = np.eye(n) / n  # Diagonal transport

        # Normalize T rows to get identity-like mapping
        T_normalized = np.eye(n)
        projected = barycentric_projection(T_normalized * (1.0/n), z)
        # Each row projects to the corresponding target row
        # (but scaled by 1/n in T, then divided by row sum 1/n = identity)
        np.testing.assert_allclose(projected, z, atol=1e-10)

    def test_uniform_transport(self):
        """Uniform transport should project to the mean of target."""
        n_s, n_t, d = 5, 8, 4
        z_t = np.random.randn(n_t, d)
        T = np.ones((n_s, n_t)) / (n_s * n_t)  # Uniform

        projected = barycentric_projection(T, z_t)
        expected = z_t.mean(axis=0)

        for i in range(n_s):
            np.testing.assert_allclose(projected[i], expected, atol=1e-10)

    def test_output_shape(self):
        n_s, n_t, d = 10, 15, 8
        z_t = np.random.randn(n_t, d)
        T = np.random.rand(n_s, n_t)

        projected = barycentric_projection(T, z_t)
        assert projected.shape == (n_s, d)


class TestSubsampling:
    """Test subsampling for FGW."""

    def test_no_subsample_needed(self):
        z = np.random.randn(50, 4)
        coords = np.random.randn(50, 2)
        z_sub, coords_sub, idx = subsample_for_fgw(z, coords, max_nodes=100)
        assert z_sub.shape == z.shape
        assert coords_sub.shape == coords.shape
        np.testing.assert_array_equal(idx, np.arange(50))

    def test_subsample_reduces_size(self):
        z = np.random.randn(200, 4)
        coords = np.random.randn(200, 2)
        z_sub, coords_sub, idx = subsample_for_fgw(z, coords, max_nodes=50)
        assert z_sub.shape == (50, 4)
        assert coords_sub.shape == (50, 2)
        assert len(idx) == 50
        assert np.all(np.diff(idx) > 0)  # sorted

    def test_subsample_preserves_data(self):
        z = np.random.randn(100, 4)
        coords = np.random.randn(100, 2)
        z_sub, coords_sub, idx = subsample_for_fgw(z, coords, max_nodes=30)
        np.testing.assert_array_equal(z_sub, z[idx])
        np.testing.assert_array_equal(coords_sub, coords[idx])


class TestCrossSpeciesContrastiveLoss:
    """Test cross-species contrastive loss."""

    def test_basic_output(self):
        z_s = torch.randn(20, 16)
        z_t = torch.randn(15, 16)
        T = torch.rand(20, 15)
        T = T / T.sum()

        loss_fn = CrossSpeciesContrastiveLoss(top_k=3, temperature=0.1, n_negatives=10)
        loss = loss_fn(z_s, z_t, T)

        assert loss.item() > 0
        assert loss.requires_grad

    def test_gradient_flow(self):
        z_s = torch.randn(20, 16, requires_grad=True)
        z_t = torch.randn(15, 16, requires_grad=True)
        T = torch.rand(20, 15)

        loss_fn = CrossSpeciesContrastiveLoss(top_k=3, temperature=0.1, n_negatives=10)
        loss = loss_fn(z_s, z_t, T)
        loss.backward()

        assert z_s.grad is not None
        assert z_t.grad is not None

    def test_empty_input(self):
        z_s = torch.randn(0, 16)
        z_t = torch.randn(10, 16)
        T = torch.rand(0, 10)

        loss_fn = CrossSpeciesContrastiveLoss(top_k=3)
        loss = loss_fn(z_s, z_t, T)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestFGWAlignmentLoss:
    """Test FGW alignment loss."""

    def test_basic_output(self):
        z_s = torch.randn(20, 16)
        z_t = torch.randn(15, 16)
        T = torch.rand(20, 15)
        T = T / T.sum()

        loss_fn = FGWAlignmentLoss()
        loss = loss_fn(z_s, z_t, T)

        assert loss.item() > 0

    def test_gradient_flow(self):
        z_s = torch.randn(20, 16, requires_grad=True)
        z_t = torch.randn(15, 16, requires_grad=True)
        T = torch.rand(20, 15).detach()

        loss_fn = FGWAlignmentLoss()
        loss = loss_fn(z_s, z_t, T)
        loss.backward()

        assert z_s.grad is not None
        assert z_t.grad is not None

    def test_identical_embeddings_low_loss(self):
        """Identical embeddings with identity transport should yield low loss."""
        n = 10
        z = torch.randn(n, 8)
        # Identity-like transport: each source maps to same target
        T = torch.eye(n) / n

        loss_fn = FGWAlignmentLoss()
        loss = loss_fn(z, z.clone(), T)

        # With identity transport and identical embeddings, cosine distance
        # on diagonal = 0, so T * M sums diagonal zeros -> loss near 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_decreases_with_alignment(self):
        """Loss should be lower when embeddings are more aligned."""
        torch.manual_seed(42)
        n, d = 20, 8
        z_s = torch.randn(n, d)

        # Well-aligned: target similar to source
        z_t_aligned = z_s + 0.01 * torch.randn(n, d)
        # Poorly aligned: random target
        z_t_random = torch.randn(n, d)

        T = torch.eye(n) / n

        loss_fn = FGWAlignmentLoss()
        loss_aligned = loss_fn(z_s, z_t_aligned, T)
        loss_random = loss_fn(z_s, z_t_random, T)

        assert loss_aligned.item() < loss_random.item()


class TestMMDLoss:
    """Test MMD distribution matching loss."""

    def test_same_distribution_low_loss(self):
        """Samples from the same distribution should have low MMD."""
        torch.manual_seed(42)
        z = torch.randn(200, 16)
        loss_fn = MMDLoss()
        # Split same distribution in half
        loss_same = loss_fn(z[:100], z[100:])
        # Different distributions
        loss_diff = loss_fn(z[:100], z[100:] + 5.0)
        assert loss_same.item() < loss_diff.item()

    def test_gradient_flow(self):
        """Gradients propagate through MMD loss."""
        z_s = torch.randn(50, 16, requires_grad=True)
        z_t = torch.randn(50, 16, requires_grad=True)
        loss_fn = MMDLoss()
        loss = loss_fn(z_s, z_t)
        loss.backward()
        assert z_s.grad is not None
        assert z_t.grad is not None

    def test_non_negative(self):
        """MMD^2 should be non-negative."""
        torch.manual_seed(42)
        loss_fn = MMDLoss()
        for _ in range(5):
            z_s = torch.randn(30, 8)
            z_t = torch.randn(30, 8)
            loss = loss_fn(z_s, z_t)
            assert loss.item() >= -1e-6


class TestSharedProjectionHead:
    """Test shared projection head for cross-species embedding alignment."""

    def test_output_shape(self):
        """(100, 64) in -> (100, 64) out."""
        head = SharedProjectionHead(64, 128, 64)
        head.eval()
        x = torch.randn(100, 64)
        out = head(x)
        assert out.shape == (100, 64)

    def test_shared_weights(self):
        """Same input produces same output (deterministic in eval mode)."""
        head = SharedProjectionHead(64, 128, 64)
        head.eval()
        x = torch.randn(50, 64)
        out1 = head(x)
        out2 = head(x)
        torch.testing.assert_close(out1, out2)

    def test_gradient_flow(self):
        """Gradients propagate through projection head to input."""
        head = SharedProjectionHead(64, 128, 64)
        head.train()
        x = torch.randn(32, 64, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_batchnorm(self):
        """Output is not constant (BatchNorm + ReLU produce non-trivial transform)."""
        head = SharedProjectionHead(64, 128, 64)
        head.train()
        x = torch.randn(32, 64)
        out = head(x)
        # Not all outputs should be identical across the batch
        assert not torch.allclose(out[0], out[1], atol=1e-6)


class TestLogDomainFGW:
    """Test log-domain entropic FGW solver."""

    def test_low_epsilon_convergence(self):
        """FGW converges at epsilon=0.005 without NaN."""
        rng = np.random.default_rng(42)
        n_s, n_t, d = 15, 12, 8
        z_s = rng.standard_normal((n_s, d))
        z_t = rng.standard_normal((n_t, d))
        coords_s = rng.uniform(0, 100, (n_s, 2))
        coords_t = rng.uniform(0, 100, (n_t, 2))

        aligner = FGWAligner(alpha=0.3, epsilon=0.005, max_iter=50, sinkhorn_max_iter=200)
        T, dist = aligner.compute_transport_plan(z_s, z_t, coords_s, coords_t)

        assert T.shape == (n_s, n_t)
        assert not np.any(np.isnan(T)), "Transport plan contains NaN"
        assert not np.any(np.isnan(dist)), "FGW distance is NaN"
        assert (T >= -1e-10).all(), "Transport plan has large negative entries"
        # Valid coupling: marginals should match
        assert np.allclose(T.sum(axis=1), 1.0 / n_s, atol=1e-3)
        assert np.allclose(T.sum(axis=0), 1.0 / n_t, atol=1e-3)

    def test_entropy_decreases_with_epsilon(self):
        """Entropy ratio at epsilon=0.005 < entropy ratio at epsilon=0.05."""
        rng = np.random.default_rng(42)
        n_s, n_t, d = 15, 12, 8
        z_s = rng.standard_normal((n_s, d))
        z_t = rng.standard_normal((n_t, d))
        coords_s = rng.uniform(0, 100, (n_s, 2))
        coords_t = rng.uniform(0, 100, (n_t, 2))

        def _entropy_ratio(T):
            T_pos = np.clip(T, 1e-30, None)
            entropy = -(T_pos * np.log(T_pos)).sum()
            max_entropy = np.log(T.shape[0] * T.shape[1])
            return entropy / max_entropy

        aligner_low = FGWAligner(alpha=0.3, epsilon=0.005, max_iter=50, sinkhorn_max_iter=200)
        T_low, _ = aligner_low.compute_transport_plan(z_s, z_t, coords_s, coords_t)

        aligner_high = FGWAligner(alpha=0.3, epsilon=0.05, max_iter=50, sinkhorn_max_iter=200)
        T_high, _ = aligner_high.compute_transport_plan(z_s, z_t, coords_s, coords_t)

        ratio_low = _entropy_ratio(T_low)
        ratio_high = _entropy_ratio(T_high)
        assert ratio_low < ratio_high, (
            f"Low epsilon should produce sharper plans: {ratio_low:.4f} >= {ratio_high:.4f}"
        )
