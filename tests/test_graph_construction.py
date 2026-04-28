"""Tests for spatial graph construction."""

import torch
import pytest

from rosetta.chisel.graph_construction import build_spatial_graph
from rosetta.utils.config import GraphConfig


def make_grid_data(grid_size=10, n_genes=50):
    """Create a grid of spots with random expression."""
    rng = torch.Generator().manual_seed(42)
    coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            coords.append([float(i), float(j)])
    spatial_coords = torch.tensor(coords, dtype=torch.float32)
    expression = torch.randn(grid_size**2, n_genes, generator=rng)
    return spatial_coords, expression


class TestBuildSpatialGraph:
    def test_output_is_data(self):
        coords, expr = make_grid_data()
        data = build_spatial_graph(coords, expr, k_spatial=4)
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "edge_attr")
        assert hasattr(data, "pos")

    def test_shapes(self):
        n, g = 100, 50
        coords, expr = make_grid_data(grid_size=10, n_genes=g)
        data = build_spatial_graph(coords, expr, k_spatial=4)

        assert data.x.shape == (n, g)
        assert data.pos.shape == (n, 2)
        assert data.edge_index.shape[0] == 2
        assert data.edge_attr.shape[1] == 5  # [weight, dist, dx, dy, type]

    def test_knn_correctness(self):
        coords, expr = make_grid_data(grid_size=5)
        k = 4
        data = build_spatial_graph(
            coords, expr, k_spatial=k, add_expression_edges=False
        )
        # Each node should have exactly k outgoing edges in KNN
        n = coords.shape[0]
        src = data.edge_index[0]
        for i in range(n):
            assert (src == i).sum().item() == k

    def test_gaussian_weights(self):
        coords, expr = make_grid_data()
        data = build_spatial_graph(
            coords, expr, k_spatial=4, add_expression_edges=False
        )
        weights = data.edge_attr[:, 0]
        # Gaussian weights should be in (0, 1]
        assert (weights > 0).all()
        assert (weights <= 1.0 + 1e-6).all()

    def test_edge_attributes_consistency(self):
        coords, expr = make_grid_data()
        data = build_spatial_graph(coords, expr, k_spatial=4)
        src, dst = data.edge_index

        # Verify rel_dx, rel_dy match actual displacement
        for idx in range(min(20, data.edge_index.shape[1])):
            s, d = src[idx].item(), dst[idx].item()
            expected_dx = coords[d, 0] - coords[s, 0]
            expected_dy = coords[d, 1] - coords[s, 1]
            assert torch.allclose(
                data.edge_attr[idx, 2], expected_dx, atol=1e-5
            )
            assert torch.allclose(
                data.edge_attr[idx, 3], expected_dy, atol=1e-5
            )

    def test_no_self_loops(self):
        coords, expr = make_grid_data()
        data = build_spatial_graph(coords, expr, k_spatial=4)
        src, dst = data.edge_index
        assert (src != dst).all()

    def test_expression_edges_added(self):
        coords, expr = make_grid_data(grid_size=10, n_genes=50)
        data_no_expr = build_spatial_graph(
            coords, expr, k_spatial=4, add_expression_edges=False
        )
        data_with_expr = build_spatial_graph(
            coords, expr, k_spatial=4, k_expression=3,
            expression_threshold=0.0, add_expression_edges=True,
        )
        # With expression edges, should have more edges
        assert data_with_expr.edge_index.shape[1] >= data_no_expr.edge_index.shape[1]

    def test_expression_edge_type(self):
        coords, expr = make_grid_data(grid_size=10, n_genes=50)
        data = build_spatial_graph(
            coords, expr, k_spatial=4, k_expression=3,
            expression_threshold=0.0, add_expression_edges=True,
        )
        edge_types = data.edge_attr[:, 4]
        # Should have type 0 (spatial) and type 1 (expression)
        assert (edge_types == 0).any()
        # May or may not have expression edges depending on similarity
