"""Tests for CHISEL self-supervised loss functions."""

import pytest
import torch

from rosetta.chisel.losses import GeneMasker, MaskedExpressionLoss, SpatialContrastiveLoss


class TestGeneMasker:
    """Test gene expression masking."""

    def test_mask_ratio(self):
        """Correct fraction of genes are masked."""
        x = torch.randn(100, 50).abs()  # All positive (nonzero)
        masker = GeneMasker(mask_ratio=0.2, mask_nonzero_only=False)
        masked_x, mask = masker(x)

        # Each row should have ~20% masked (allow tolerance for rounding)
        per_row_masked = mask.sum(dim=1).float()
        expected = 0.2 * 50  # 10
        assert (per_row_masked >= 1).all(), "At least 1 gene masked per row"
        assert per_row_masked.mean().item() == pytest.approx(expected, abs=2)

    def test_mask_nonzero_only(self):
        """Only nonzero genes are masked when mask_nonzero_only=True."""
        x = torch.zeros(50, 100)
        # Set 10 genes per spot to nonzero
        for i in range(50):
            x[i, :10] = torch.randn(10).abs() + 0.1

        masker = GeneMasker(mask_ratio=0.5, mask_nonzero_only=True)
        masked_x, mask = masker(x)

        # Masked positions should only be where original was nonzero
        assert (mask & (x == 0)).sum() == 0, "Should not mask zero entries"

    def test_masked_values_are_zero(self):
        """Masked positions should be set to 0."""
        x = torch.randn(20, 30).abs() + 1.0  # All > 0
        masker = GeneMasker(mask_ratio=0.3, mask_nonzero_only=False)
        masked_x, mask = masker(x)

        assert (masked_x[mask] == 0).all(), "Masked positions should be 0"

    def test_unmasked_values_preserved(self):
        """Unmasked positions should retain original values."""
        x = torch.randn(20, 30)
        masker = GeneMasker(mask_ratio=0.2, mask_nonzero_only=False)
        masked_x, mask = masker(x)

        assert torch.allclose(masked_x[~mask], x[~mask])

    def test_output_shapes(self):
        """Output shapes match input."""
        x = torch.randn(50, 100)
        masker = GeneMasker(mask_ratio=0.2)
        masked_x, mask = masker(x)

        assert masked_x.shape == x.shape
        assert mask.shape == x.shape
        assert mask.dtype == torch.bool


class TestMaskedExpressionLoss:
    """Test masked expression reconstruction loss."""

    def test_perfect_prediction(self):
        """Loss should be 0 when prediction matches target."""
        pred = torch.randn(20, 50)
        target = pred.clone()
        mask = torch.zeros(20, 50, dtype=torch.bool)
        mask[:, :10] = True

        loss_fn = MaskedExpressionLoss()
        loss = loss_fn(pred, target, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_for_wrong_prediction(self):
        """Loss should be > 0 when prediction differs from target."""
        pred = torch.randn(20, 50)
        target = torch.randn(20, 50)
        mask = torch.ones(20, 50, dtype=torch.bool)

        loss_fn = MaskedExpressionLoss()
        loss = loss_fn(pred, target, mask)
        assert loss.item() > 0

    def test_gradient_flow(self):
        """Gradients should flow back through the loss."""
        pred = torch.randn(20, 50, requires_grad=True)
        target = torch.randn(20, 50)
        mask = torch.ones(20, 50, dtype=torch.bool)

        loss_fn = MaskedExpressionLoss()
        loss = loss_fn(pred, target, mask)
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_node_mask(self):
        """Loss should only consider nodes in node_mask."""
        pred = torch.zeros(20, 50, requires_grad=True)
        target = torch.ones(20, 50)
        gene_mask = torch.ones(20, 50, dtype=torch.bool)
        node_mask = torch.zeros(20, dtype=torch.bool)
        node_mask[:5] = True  # Only first 5 nodes

        loss_fn = MaskedExpressionLoss()
        loss = loss_fn(pred, target, gene_mask, node_mask)

        # Loss should be 1.0 (MSE of 0 vs 1)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_empty_mask_returns_zero(self):
        """Empty mask should return 0 loss."""
        pred = torch.randn(20, 50, requires_grad=True)
        target = torch.randn(20, 50)
        mask = torch.zeros(20, 50, dtype=torch.bool)

        loss_fn = MaskedExpressionLoss()
        loss = loss_fn(pred, target, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestSpatialContrastiveLoss:
    """Test spatial contrastive loss."""

    def _make_graph(self, n=100, k=6, dim=32):
        """Create a simple test graph."""
        z = torch.randn(n, dim)
        # Create KNN-like edges
        src = torch.arange(n).repeat_interleave(k)
        dst = torch.randint(0, n, (n * k,))
        edge_index = torch.stack([src, dst])
        return z, edge_index

    def test_positive_loss(self):
        """Loss should be > 0 for random embeddings."""
        z, edge_index = self._make_graph()
        loss_fn = SpatialContrastiveLoss(temperature=0.1, n_negatives=32)
        loss = loss_fn(z, edge_index)
        assert loss.item() > 0

    def test_gradient_flow(self):
        """Gradients should flow through the loss."""
        z = torch.randn(100, 32, requires_grad=True)
        src = torch.arange(100).repeat_interleave(4)
        dst = torch.randint(0, 100, (400,))
        edge_index = torch.stack([src, dst])

        loss_fn = SpatialContrastiveLoss(temperature=0.1, n_negatives=32)
        loss = loss_fn(z, edge_index)
        loss.backward()

        assert z.grad is not None

    def test_similar_neighbors_lower_loss(self):
        """Loss should be lower when neighbors have similar embeddings."""
        torch.manual_seed(42)
        n, dim = 100, 32

        # Create two clusters in embedding space with within-cluster edges
        # Cluster 0: nodes 0-49, Cluster 1: nodes 50-99
        z_clustered = torch.randn(n, dim)
        z_clustered[:50] = torch.randn(1, dim) + 0.05 * torch.randn(50, dim)
        z_clustered[50:] = torch.randn(1, dim) + 0.05 * torch.randn(50, dim)
        z_clustered.requires_grad_(True)

        # Edges only within clusters (neighbors are similar)
        src_0 = torch.arange(49)
        dst_0 = torch.arange(1, 50)
        src_1 = torch.arange(50, 99)
        dst_1 = torch.arange(51, 100)
        edge_index = torch.stack([
            torch.cat([src_0, src_1]),
            torch.cat([dst_0, dst_1]),
        ])

        # Random embeddings with same edges
        z_random = torch.randn(n, dim, requires_grad=True)

        loss_fn = SpatialContrastiveLoss(temperature=0.1, n_negatives=64)

        loss_clustered = loss_fn(z_clustered, edge_index)
        loss_random = loss_fn(z_random, edge_index)

        # Clustered neighbors should produce lower contrastive loss
        assert loss_clustered.item() < loss_random.item()

    def test_node_mask(self):
        """Loss should respect node_mask."""
        z, edge_index = self._make_graph(n=50)
        node_mask = torch.zeros(50, dtype=torch.bool)
        node_mask[:10] = True

        loss_fn = SpatialContrastiveLoss(temperature=0.1, n_negatives=16)
        loss = loss_fn(z, edge_index, node_mask)
        assert loss.item() >= 0

    def test_empty_edges_returns_zero(self):
        """Empty edge_index should return 0 loss."""
        z = torch.randn(50, 32)
        node_mask = torch.zeros(50, dtype=torch.bool)  # No nodes selected

        edge_index = torch.zeros(2, 10, dtype=torch.long)
        loss_fn = SpatialContrastiveLoss()
        loss = loss_fn(z, edge_index, node_mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
