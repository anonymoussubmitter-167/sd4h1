"""Spatial graph construction for spatial transcriptomics data.

Converts spatial coordinates and gene expression to PyG Data objects
with spatial KNN edges, optional expression-based long-range edges,
and edge attributes (weight, distance, relative displacement, edge type).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import Tensor
from torch_geometric.data import Data

from rosetta.data.loaders import SpatialDataset
from rosetta.utils.config import GraphConfig


def _knn_graph_scipy(coords: Tensor, k: int) -> Tensor:
    """Build a KNN graph using scipy cKDTree (no torch-cluster needed).

    Returns edge_index (2, N*k) where edges go from each node to its k neighbors.
    """
    coords_np = coords.detach().cpu().numpy()
    tree = cKDTree(coords_np)
    _, indices = tree.query(coords_np, k=k + 1)  # +1 because query includes self
    # indices: (N, k+1), first column is self
    indices = indices[:, 1:]  # remove self-loop

    n = coords.shape[0]
    src = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = torch.from_numpy(indices.reshape(-1)).long()
    return torch.stack([src, dst], dim=0)


def build_spatial_graph(
    spatial_coords: Tensor,
    expression: Tensor,
    k_spatial: int = 6,
    k_expression: int = 3,
    sigma: Optional[float] = None,
    expression_threshold: float = 0.5,
    add_expression_edges: bool = True,
) -> Data:
    """Build a spatial graph from coordinates and expression data.

    Args:
        spatial_coords: (N, 2) spatial coordinates.
        expression: (N, G) gene expression matrix.
        k_spatial: Number of spatial nearest neighbors.
        k_expression: Number of expression-based neighbors to add.
        sigma: Gaussian kernel bandwidth. If None, auto-computed from
            median nearest-neighbor distance.
        expression_threshold: Minimum cosine similarity for expression edges.
        add_expression_edges: Whether to add long-range expression edges.

    Returns:
        PyG Data object with:
            x: (N, G) node features (expression)
            edge_index: (2, E) edge indices
            edge_attr: (E, 5) [weight, distance, rel_dx, rel_dy, edge_type]
            pos: (N, 2) spatial coordinates
    """
    n = spatial_coords.shape[0]

    # 1. Build KNN spatial edges using scipy
    edge_index_spatial = _knn_graph_scipy(spatial_coords, k=k_spatial)

    # Compute pairwise distances for spatial edges
    src, dst = edge_index_spatial
    diff = spatial_coords[dst] - spatial_coords[src]  # (E, 2)
    distances = torch.norm(diff, dim=1)  # (E,)

    # 2. Auto-sigma from median NN distance
    if sigma is None:
        sigma = float(torch.median(distances).item())
        if sigma < 1e-8:
            sigma = 1.0

    # 3. Gaussian kernel weights
    weights_spatial = torch.exp(-distances**2 / (2 * sigma**2))

    # Relative displacement
    rel_dx = diff[:, 0]
    rel_dy = diff[:, 1]

    # Edge type: 0 = spatial
    edge_type_spatial = torch.zeros(edge_index_spatial.shape[1])

    # 4. Optionally add expression-based long-range edges
    if add_expression_edges and k_expression > 0 and n > k_spatial + 1:
        expr_norm = F.normalize(expression, p=2, dim=1)  # (N, G)
        sim_matrix = expr_norm @ expr_norm.t()  # (N, N)

        # Mask existing spatial edges and self-loops
        mask = torch.ones(n, n, dtype=torch.bool)
        mask[src, dst] = False
        mask.fill_diagonal_(False)

        # Threshold
        sim_matrix = sim_matrix * mask.float()
        sim_matrix[sim_matrix < expression_threshold] = 0

        # Top-k expression neighbors per node
        k_expr = min(k_expression, n - 1)
        topk_vals, topk_idx = torch.topk(sim_matrix, k_expr, dim=1)

        # Build edge list from nonzero entries
        row_idx = torch.arange(n).unsqueeze(1).expand_as(topk_idx)
        valid = topk_vals > 0
        expr_src = row_idx[valid]
        expr_dst = topk_idx[valid]

        if expr_src.numel() > 0:
            edge_index_expr = torch.stack([expr_src, expr_dst], dim=0)

            # Compute attributes for expression edges
            diff_expr = spatial_coords[expr_dst] - spatial_coords[expr_src]
            dist_expr = torch.norm(diff_expr, dim=1)
            weights_expr = topk_vals[valid]  # Use cosine similarity as weight
            rel_dx_expr = diff_expr[:, 0]
            rel_dy_expr = diff_expr[:, 1]
            edge_type_expr = torch.ones(edge_index_expr.shape[1])

            # Concatenate
            edge_index = torch.cat([edge_index_spatial, edge_index_expr], dim=1)
            weights = torch.cat([weights_spatial, weights_expr])
            distances_all = torch.cat([distances, dist_expr])
            rel_dx = torch.cat([rel_dx, rel_dx_expr])
            rel_dy = torch.cat([rel_dy, rel_dy_expr])
            edge_type = torch.cat([edge_type_spatial, edge_type_expr])
        else:
            edge_index = edge_index_spatial
            weights = weights_spatial
            distances_all = distances
            edge_type = edge_type_spatial
    else:
        edge_index = edge_index_spatial
        weights = weights_spatial
        distances_all = distances
        edge_type = edge_type_spatial

    # 5. Edge attributes: [weight, distance, rel_dx, rel_dy, edge_type]
    edge_attr = torch.stack([weights, distances_all, rel_dx, rel_dy, edge_type], dim=1)

    return Data(
        x=expression,
        edge_index=edge_index.long(),
        edge_attr=edge_attr,
        pos=spatial_coords,
    )


def spatial_dataset_to_pyg(
    dataset: SpatialDataset,
    config: GraphConfig | None = None,
) -> Data:
    """Convenience wrapper to convert a SpatialDataset to a PyG Data object."""
    if config is None:
        config = GraphConfig()

    import numpy as np
    import scipy.sparse as sp

    spatial_coords = torch.tensor(dataset.spatial_coords, dtype=torch.float32)

    # Convert expression matrix (may be sparse)
    X = dataset.adata.X
    if sp.issparse(X):
        X = X.toarray()
    expression = torch.tensor(np.array(X), dtype=torch.float32)

    return build_spatial_graph(
        spatial_coords=spatial_coords,
        expression=expression,
        k_spatial=config.k_spatial,
        k_expression=config.k_expression,
        sigma=config.sigma,
        expression_threshold=config.expression_threshold,
    )
