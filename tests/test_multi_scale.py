"""Tests for multi-scale DiffPool encoder."""

import torch
import pytest

from rosetta.chisel.multi_scale import DiffPoolLevel, MultiScaleEncoder
from rosetta.utils.config import MultiScaleConfig


class TestDiffPoolLevel:
    def test_output_shapes(self):
        batch_size, n_nodes, in_dim = 2, 20, 32
        n_clusters = 5

        level = DiffPoolLevel(in_dim, embed_dim=64, n_clusters=n_clusters)
        x = torch.randn(batch_size, n_nodes, in_dim)
        adj = torch.rand(batch_size, n_nodes, n_nodes)
        adj = (adj + adj.transpose(1, 2)) / 2  # symmetric

        x_pool, adj_pool, link_loss, ent_loss = level(x, adj)

        assert x_pool.shape == (batch_size, n_clusters, 64)
        assert adj_pool.shape == (batch_size, n_clusters, n_clusters)

    def test_node_reduction(self):
        level = DiffPoolLevel(in_dim=32, embed_dim=64, n_clusters=5)
        x = torch.randn(2, 20, 32)
        adj = torch.rand(2, 20, 20)

        x_pool, _, _, _ = level(x, adj)
        assert x_pool.shape[1] < x.shape[1]

    def test_losses_nonnegative(self):
        level = DiffPoolLevel(in_dim=32, embed_dim=64, n_clusters=5)
        x = torch.randn(2, 20, 32)
        adj = torch.rand(2, 20, 20)
        adj = (adj + adj.transpose(1, 2)) / 2

        _, _, link_loss, ent_loss = level(x, adj)
        assert link_loss.item() >= 0
        assert ent_loss.item() >= 0


class TestMultiScaleEncoder:
    def test_output_shapes(self):
        config = MultiScaleConfig(
            pool_ratios=[0.5, 0.5], embed_dim=64, num_gnn_layers_per_level=2
        )
        max_nodes = 20
        encoder = MultiScaleEncoder(config, input_dim=32, max_nodes=max_nodes)

        n1 = max(1, int(max_nodes * 0.5))  # 10
        n2 = max(1, int(n1 * 0.5))  # 5

        x = torch.randn(2, max_nodes, 32)
        adj = torch.rand(2, max_nodes, max_nodes)

        out = encoder(x, adj)

        assert out["z_niche"].shape == (2, n1, 64)
        assert out["z_region"].shape == (2, n2, 64)

    def test_hierarchical_reduction(self):
        config = MultiScaleConfig(
            pool_ratios=[0.25, 0.25], embed_dim=64, num_gnn_layers_per_level=2
        )
        max_nodes = 40
        encoder = MultiScaleEncoder(config, input_dim=32, max_nodes=max_nodes)

        x = torch.randn(1, max_nodes, 32)
        adj = torch.rand(1, max_nodes, max_nodes)

        out = encoder(x, adj)

        # Niche should have fewer nodes than input
        assert out["z_niche"].shape[1] < max_nodes
        # Region should have fewer nodes than niche
        assert out["z_region"].shape[1] < out["z_niche"].shape[1]

    def test_losses_present(self):
        config = MultiScaleConfig(pool_ratios=[0.5, 0.5], embed_dim=64)
        encoder = MultiScaleEncoder(config, input_dim=32, max_nodes=20)

        x = torch.randn(1, 20, 32)
        adj = torch.rand(1, 20, 20)

        out = encoder(x, adj)

        assert "link_loss" in out
        assert "entropy_loss" in out
        assert out["link_loss"].item() >= 0
        assert out["entropy_loss"].item() >= 0
