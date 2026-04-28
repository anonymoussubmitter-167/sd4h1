"""Multi-scale graph encoding using hierarchical DiffPool.

Produces embeddings at three scales:
- z_spot: per-spot (passthrough from GNN)
- z_niche: local neighborhood (after first pooling)
- z_region: regional (after second pooling)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import DenseGCNConv, dense_diff_pool

from rosetta.utils.config import MultiScaleConfig


class DiffPoolGNN(nn.Module):
    """Simple GNN for DiffPool (uses DenseGCNConv for dense adjacency)."""

    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else out_dim
            self.layers.append(DenseGCNConv(d_in, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))

    def forward(self, x: Tensor, adj: Tensor, mask: Tensor | None = None) -> Tensor:
        for conv, norm in zip(self.layers, self.norms):
            x = conv(x, adj, mask)
            x = norm(x)
            x = torch.relu(x)
        return x


class DiffPoolLevel(nn.Module):
    """One level of DiffPool: embed GNN + assignment GNN.

    Args:
        in_dim: Input feature dimension.
        embed_dim: Output embedding dimension.
        n_clusters: Number of clusters (pooled nodes) to produce.
        num_gnn_layers: Number of GNN layers in embed/assign networks.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        n_clusters: int,
        num_gnn_layers: int = 2,
    ):
        super().__init__()
        self.embed_gnn = DiffPoolGNN(in_dim, embed_dim, num_gnn_layers)
        self.assign_gnn = DiffPoolGNN(in_dim, n_clusters, num_gnn_layers)

    def forward(
        self, x: Tensor, adj: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x: (B, N, D) node features
            adj: (B, N, N) adjacency matrix
            mask: (B, N) node mask

        Returns:
            x_pool: (B, n_clusters, embed_dim) pooled features
            adj_pool: (B, n_clusters, n_clusters) coarsened adjacency
            link_loss: link prediction loss
            entropy_loss: assignment entropy loss
        """
        z = self.embed_gnn(x, adj, mask)
        s = self.assign_gnn(x, adj, mask)

        x_pool, adj_pool, link_loss, entropy_loss = dense_diff_pool(z, adj, s, mask)

        return x_pool, adj_pool, link_loss, entropy_loss


class MultiScaleEncoder(nn.Module):
    """Hierarchical multi-scale encoder using two DiffPool levels.

    Produces embeddings at three scales:
    - z_spot: input features (passthrough)
    - z_niche: after first pooling level
    - z_region: after second pooling level
    """

    def __init__(self, config: MultiScaleConfig, input_dim: int, max_nodes: int):
        """
        Args:
            config: MultiScaleConfig with pool_ratios, embed_dim, etc.
            input_dim: Dimension of input node features.
            max_nodes: Maximum number of nodes in the graph (for computing cluster sizes).
        """
        super().__init__()
        self.config = config

        # Compute cluster sizes from pool ratios
        n1 = max(1, int(max_nodes * config.pool_ratios[0]))
        n2 = max(1, int(n1 * config.pool_ratios[1]))

        self.pool1 = DiffPoolLevel(
            in_dim=input_dim,
            embed_dim=config.embed_dim,
            n_clusters=n1,
            num_gnn_layers=config.num_gnn_layers_per_level,
        )
        self.pool2 = DiffPoolLevel(
            in_dim=config.embed_dim,
            embed_dim=config.embed_dim,
            n_clusters=n2,
            num_gnn_layers=config.num_gnn_layers_per_level,
        )

    def forward(
        self, x: Tensor, adj: Tensor, mask: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: (B, N, D) node features from SpatialGNNEncoder
            adj: (B, N, N) dense adjacency matrix
            mask: (B, N) node mask

        Returns:
            dict with z_niche, z_region, link_loss, entropy_loss
        """
        # Level 1: spot -> niche
        z_niche, adj1, link1, ent1 = self.pool1(x, adj, mask)

        # Level 2: niche -> region
        z_region, adj2, link2, ent2 = self.pool2(z_niche, adj1)

        return {
            "z_niche": z_niche,
            "z_region": z_region,
            "link_loss": link1 + link2,
            "entropy_loss": ent1 + ent2,
        }
