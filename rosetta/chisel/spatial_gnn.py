"""Spatial GNN encoder with spatially-aware message passing.

Implements position-aware attention using relative displacements and
distances, making the model translation-invariant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from rosetta.utils.config import SpatialGNNConfig


class SpatialMessagePassing(MessagePassing):
    """Spatially-aware message passing with multi-head attention.

    Computes:
        h_i^(l+1) = sigma(W_self * h_i + sum_j alpha_ij * W_msg * h_j)
        alpha_ij = softmax_j(MLP([h_i || h_j || r_ij || d_ij]))

    where r_ij is relative displacement (translation-invariant) and d_ij is distance.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        spatial_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_msg = nn.Linear(in_dim, out_dim)

        # Attention MLP: takes [h_i, h_j, r_ij, d_ij]
        attn_input_dim = 2 * in_dim + spatial_dim + 1
        self.attn_mlp = nn.Sequential(
            nn.Linear(attn_input_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, num_heads),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, 5) [weight, distance, rel_dx, rel_dy, edge_type]
        """
        # Self-transform
        h_self = self.W_self(x)

        # Message passing
        h_msg = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        return h_self + h_msg

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, index: Tensor) -> Tensor:
        """Compute messages with spatial attention.

        Args:
            x_i: (E, in_dim) features of target nodes
            x_j: (E, in_dim) features of source nodes
            edge_attr: (E, 5) edge attributes
            index: (E,) target node indices for softmax
        """
        # Extract spatial info: rel_dx, rel_dy, distance
        rel_pos = edge_attr[:, 2:4]  # (E, 2) rel_dx, rel_dy
        dist = edge_attr[:, 1:2]     # (E, 1) distance

        # Attention scores
        attn_input = torch.cat([x_i, x_j, rel_pos, dist], dim=-1)
        attn_logits = self.attn_mlp(attn_input)  # (E, num_heads)

        # Softmax over neighbors
        attn_weights = softmax(attn_logits, index)  # (E, num_heads)
        attn_weights = self.dropout(attn_weights)

        # Transform source features
        msg = self.W_msg(x_j)  # (E, out_dim)

        # Reshape for multi-head: (E, num_heads, head_dim)
        msg = msg.view(-1, self.num_heads, self.head_dim)
        attn_weights = attn_weights.unsqueeze(-1)  # (E, num_heads, 1)

        # Apply attention
        msg = (attn_weights * msg).view(-1, self.num_heads * self.head_dim)

        return msg


class SpatialGNNBlock(nn.Module):
    """SpatialMessagePassing + LayerNorm + residual + GELU."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        spatial_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mp = SpatialMessagePassing(dim, dim, num_heads, spatial_dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        residual = x
        x = self.mp(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x


class SpatialGNNEncoder(nn.Module):
    """Full spatial GNN encoder: input projection + stack of SpatialGNNBlocks.

    Args:
        config: SpatialGNNConfig with model hyperparameters.

    Returns:
        (N, hidden_dim) node embeddings.
    """

    def __init__(self, config: SpatialGNNConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.input_norm = nn.LayerNorm(config.hidden_dim)

        # GNN blocks
        self.blocks = nn.ModuleList([
            SpatialGNNBlock(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                spatial_dim=config.spatial_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (N, input_dim) raw node features
            edge_index: (2, E) edge indices
            edge_attr: (E, 5) edge attributes

        Returns:
            (N, hidden_dim) node embeddings
        """
        x = self.input_proj(x)
        x = self.input_norm(x)

        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        return x
