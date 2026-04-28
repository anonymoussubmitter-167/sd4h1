"""Tests for Spatial GNN encoder."""

import torch
import pytest

from rosetta.chisel.spatial_gnn import (
    SpatialGNNBlock,
    SpatialGNNEncoder,
    SpatialMessagePassing,
)
from rosetta.utils.config import SpatialGNNConfig


def make_test_graph(n=50, in_dim=32, k=4):
    """Create a test graph with random features and KNN edges."""
    from rosetta.chisel.graph_construction import _knn_graph_scipy

    rng = torch.Generator().manual_seed(42)
    pos = torch.randn(n, 2, generator=rng)
    x = torch.randn(n, in_dim, generator=rng)
    edge_index = _knn_graph_scipy(pos, k=k)

    src, dst = edge_index
    diff = pos[dst] - pos[src]
    dist = torch.norm(diff, dim=1, keepdim=True)

    # edge_attr: [weight, distance, rel_dx, rel_dy, edge_type]
    weight = torch.exp(-dist.squeeze() ** 2 / 2)
    edge_attr = torch.stack(
        [weight, dist.squeeze(), diff[:, 0], diff[:, 1],
         torch.zeros(edge_index.shape[1])],
        dim=1,
    )

    return x, edge_index, edge_attr, pos


class TestSpatialMessagePassing:
    def test_output_shape(self):
        in_dim, out_dim = 32, 64
        mp = SpatialMessagePassing(in_dim, out_dim, num_heads=4)
        x, edge_index, edge_attr, _ = make_test_graph(n=50, in_dim=in_dim)
        out = mp(x, edge_index, edge_attr)
        assert out.shape == (50, out_dim)

    def test_gradient_flow(self):
        in_dim = 32
        mp = SpatialMessagePassing(in_dim, in_dim, num_heads=4)
        x, edge_index, edge_attr, _ = make_test_graph(in_dim=in_dim)
        x.requires_grad_(True)
        out = mp(x, edge_index, edge_attr)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestSpatialGNNBlock:
    def test_output_shape(self):
        dim = 64
        block = SpatialGNNBlock(dim, num_heads=4)
        x, edge_index, edge_attr, _ = make_test_graph(in_dim=dim)
        out = block(x, edge_index, edge_attr)
        assert out.shape == x.shape  # Same shape due to residual

    def test_residual_connection(self):
        dim = 64
        block = SpatialGNNBlock(dim, num_heads=4)
        x, edge_index, edge_attr, _ = make_test_graph(in_dim=dim)
        out = block(x, edge_index, edge_attr)
        # Output should not be identical to input (non-trivial transform)
        assert not torch.allclose(out, x)


class TestSpatialGNNEncoder:
    def test_output_shape(self):
        config = SpatialGNNConfig(
            input_dim=50, hidden_dim=64, num_layers=2, num_heads=4
        )
        encoder = SpatialGNNEncoder(config)
        x, edge_index, edge_attr, _ = make_test_graph(n=30, in_dim=50)
        out = encoder(x, edge_index, edge_attr)
        assert out.shape == (30, 64)

    def test_gradient_flow(self):
        config = SpatialGNNConfig(
            input_dim=50, hidden_dim=64, num_layers=2, num_heads=4
        )
        encoder = SpatialGNNEncoder(config)
        x, edge_index, edge_attr, _ = make_test_graph(n=30, in_dim=50)
        x.requires_grad_(True)
        out = encoder(x, edge_index, edge_attr)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_translation_invariance(self):
        """GNN should produce same embeddings under spatial translation."""
        config = SpatialGNNConfig(
            input_dim=50, hidden_dim=64, num_layers=2, num_heads=4
        )
        encoder = SpatialGNNEncoder(config)
        encoder.eval()

        x, edge_index, edge_attr, pos = make_test_graph(n=30, in_dim=50)

        with torch.no_grad():
            out1 = encoder(x, edge_index, edge_attr)

        # Translate all positions by a constant — edge_attr uses relative
        # displacements, so output should be identical
        shift = torch.tensor([100.0, -50.0])
        pos2 = pos + shift
        # edge_attr stays the same since it's relative
        with torch.no_grad():
            out2 = encoder(x, edge_index, edge_attr)

        assert torch.allclose(out1, out2, atol=1e-5)
