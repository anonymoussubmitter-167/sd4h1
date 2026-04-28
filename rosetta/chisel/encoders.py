"""Full CHISEL encoder: SpatialGNN + MultiScale pooling.

Ties together the spatial GNN encoder and multi-scale DiffPool
to produce embeddings at spot, niche, and region scales.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import to_dense_adj, to_dense_batch

from rosetta.chisel.multi_scale import MultiScaleEncoder
from rosetta.chisel.spatial_gnn import SpatialGNNEncoder
from rosetta.utils.config import CHISELConfig


class CHISELEncoder(nn.Module):
    """Complete CHISEL encoder module.

    Pipeline:
    1. SpatialGNNEncoder on sparse graph -> spot embeddings
    2. Convert sparse -> dense (to_dense_batch / to_dense_adj)
    3. MultiScaleEncoder on dense graph -> niche/region embeddings
    4. Output projection heads for each scale

    Args:
        config: CHISELConfig composing graph, GNN, and multi-scale configs.
        max_nodes: Maximum number of nodes per graph (needed for DiffPool sizing).
    """

    def __init__(self, config: CHISELConfig, max_nodes: int):
        super().__init__()
        self.config = config

        self.gnn_encoder = SpatialGNNEncoder(config.spatial_gnn)

        self.multi_scale = MultiScaleEncoder(
            config=config.multi_scale,
            input_dim=config.spatial_gnn.hidden_dim,
            max_nodes=max_nodes,
        )

        # Output projection heads
        hidden = config.spatial_gnn.hidden_dim
        embed = config.multi_scale.embed_dim
        self.proj_spot = nn.Linear(hidden, embed)
        self.proj_niche = nn.Linear(embed, embed)
        self.proj_region = nn.Linear(embed, embed)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: (N, input_dim) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, 5) edge attributes
            batch: (N,) batch assignment vector. If None, assumes single graph.

        Returns:
            dict with keys:
                z_spot: (N, embed_dim) spot-level embeddings
                z_niche: (B, n_niche, embed_dim) niche-level embeddings
                z_region: (B, n_region, embed_dim) region-level embeddings
                link_loss: scalar link prediction loss from DiffPool
                entropy_loss: scalar entropy loss from DiffPool
        """
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # 1. Sparse GNN encoding
        z_spot_sparse = self.gnn_encoder(x, edge_index, edge_attr)  # (N, hidden)

        # 2. Convert to dense for DiffPool
        z_dense, mask = to_dense_batch(z_spot_sparse, batch)  # (B, N_max, hidden)
        adj_dense = to_dense_adj(edge_index, batch)  # (B, N_max, N_max)

        # 3. Multi-scale encoding
        ms_out = self.multi_scale(z_dense, adj_dense, mask)

        # 4. Projection heads
        z_spot = self.proj_spot(z_spot_sparse)  # (N, embed)
        z_niche = self.proj_niche(ms_out["z_niche"])  # (B, n_niche, embed)
        z_region = self.proj_region(ms_out["z_region"])  # (B, n_region, embed)

        return {
            "z_spot": z_spot,
            "z_niche": z_niche,
            "z_region": z_region,
            "link_loss": ms_out["link_loss"],
            "entropy_loss": ms_out["entropy_loss"],
        }
