"""Tests for the full CHISELEncoder."""

import torch
import pytest

from rosetta.chisel.encoders import CHISELEncoder
from rosetta.utils.config import CHISELConfig, MultiScaleConfig, SpatialGNNConfig


def make_test_data(n=50, input_dim=100, k=4):
    """Create test sparse graph data."""
    from rosetta.chisel.graph_construction import _knn_graph_scipy

    rng = torch.Generator().manual_seed(42)
    pos = torch.randn(n, 2, generator=rng)
    x = torch.randn(n, input_dim, generator=rng)
    edge_index = _knn_graph_scipy(pos, k=k)

    src, dst = edge_index
    diff = pos[dst] - pos[src]
    dist = torch.norm(diff, dim=1)
    weight = torch.exp(-dist**2 / 2)
    edge_attr = torch.stack(
        [weight, dist, diff[:, 0], diff[:, 1], torch.zeros(edge_index.shape[1])],
        dim=1,
    )
    return x, edge_index, edge_attr


class TestCHISELEncoder:
    def get_config(self, input_dim=100):
        return CHISELConfig(
            spatial_gnn=SpatialGNNConfig(
                input_dim=input_dim, hidden_dim=64, num_layers=2, num_heads=4
            ),
            multi_scale=MultiScaleConfig(
                pool_ratios=[0.5, 0.5], embed_dim=64, num_gnn_layers_per_level=2
            ),
        )

    def test_forward_shapes(self):
        n, input_dim = 50, 100
        config = self.get_config(input_dim)
        encoder = CHISELEncoder(config, max_nodes=n)

        x, edge_index, edge_attr = make_test_data(n=n, input_dim=input_dim)
        out = encoder(x, edge_index, edge_attr)

        assert out["z_spot"].shape == (n, 64)
        assert out["z_niche"].ndim == 3  # (B, n_niche, embed)
        assert out["z_region"].ndim == 3  # (B, n_region, embed)
        assert out["z_niche"].shape[2] == 64
        assert out["z_region"].shape[2] == 64

    def test_with_batch(self):
        """Test with two graphs batched together."""
        n1, n2, input_dim = 30, 20, 100
        config = self.get_config(input_dim)
        max_nodes = max(n1, n2)
        encoder = CHISELEncoder(config, max_nodes=max_nodes)

        x1, ei1, ea1 = make_test_data(n=n1, input_dim=input_dim)
        x2, ei2, ea2 = make_test_data(n=n2, input_dim=input_dim)

        # Manual batching
        x = torch.cat([x1, x2], dim=0)
        edge_index = torch.cat([ei1, ei2 + n1], dim=1)
        edge_attr = torch.cat([ea1, ea2], dim=0)
        batch = torch.cat([
            torch.zeros(n1, dtype=torch.long),
            torch.ones(n2, dtype=torch.long),
        ])

        out = encoder(x, edge_index, edge_attr, batch=batch)

        assert out["z_spot"].shape[0] == n1 + n2
        assert out["z_niche"].shape[0] == 2  # batch size
        assert out["z_region"].shape[0] == 2

    def test_gradient_flow(self):
        n, input_dim = 40, 100
        config = self.get_config(input_dim)
        encoder = CHISELEncoder(config, max_nodes=n)

        x, edge_index, edge_attr = make_test_data(n=n, input_dim=input_dim)
        x.requires_grad_(True)

        out = encoder(x, edge_index, edge_attr)
        loss = out["z_spot"].sum() + out["z_niche"].sum() + out["z_region"].sum()
        loss = loss + out["link_loss"] + out["entropy_loss"]
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_no_nan_inf(self):
        n, input_dim = 40, 100
        config = self.get_config(input_dim)
        encoder = CHISELEncoder(config, max_nodes=n)

        x, edge_index, edge_attr = make_test_data(n=n, input_dim=input_dim)
        out = encoder(x, edge_index, edge_attr)

        for key in ["z_spot", "z_niche", "z_region"]:
            assert not torch.isnan(out[key]).any(), f"NaN in {key}"
            assert not torch.isinf(out[key]).any(), f"Inf in {key}"
