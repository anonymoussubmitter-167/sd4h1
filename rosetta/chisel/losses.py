"""Self-supervised loss functions for CHISEL training.

Three objectives:
1. Masked gene expression prediction (primary)
2. Spatial contrastive learning (secondary)
3. DiffPool regularization (link_loss + entropy_loss from encoder)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GeneMasker:
    """Randomly masks gene expression values for self-supervised learning.

    Replaces a fraction of gene expression values with 0 and returns
    the mask indicating which positions were masked.

    Args:
        mask_ratio: Fraction of genes to mask per spot.
        mask_nonzero_only: If True, only mask positions with nonzero expression.
    """

    def __init__(self, mask_ratio: float = 0.2, mask_nonzero_only: bool = True):
        self.mask_ratio = mask_ratio
        self.mask_nonzero_only = mask_nonzero_only

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Mask gene expression.

        Args:
            x: (N, G) gene expression matrix.

        Returns:
            masked_x: (N, G) expression with masked positions set to 0.
            mask: (N, G) boolean mask where True = masked position.
        """
        n, g = x.shape
        mask = torch.zeros_like(x, dtype=torch.bool)

        if self.mask_nonzero_only:
            # For each spot, mask a fraction of its nonzero genes
            for i in range(n):
                nonzero_idx = torch.nonzero(x[i], as_tuple=True)[0]
                if len(nonzero_idx) == 0:
                    continue
                n_mask = max(1, int(len(nonzero_idx) * self.mask_ratio))
                perm = torch.randperm(len(nonzero_idx), device=x.device)[:n_mask]
                mask[i, nonzero_idx[perm]] = True
        else:
            # Mask a fraction of all genes uniformly
            n_mask = max(1, int(g * self.mask_ratio))
            for i in range(n):
                perm = torch.randperm(g, device=x.device)[:n_mask]
                mask[i, perm] = True

        masked_x = x.clone()
        masked_x[mask] = 0.0
        return masked_x, mask


class MaskedExpressionLoss(nn.Module):
    """MSE loss computed only on masked gene positions.

    Forces the GNN to learn that spatial neighbors share expression
    patterns — it must borrow information through message passing to
    reconstruct masked genes.
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute masked reconstruction loss.

        Args:
            pred: (N, G) predicted expression.
            target: (N, G) original expression (before masking).
            mask: (N, G) boolean mask where True = masked position.
            node_mask: (N,) optional boolean mask to restrict loss to
                specific nodes (e.g., train split only).

        Returns:
            Scalar MSE loss on masked positions.
        """
        if node_mask is not None:
            pred = pred[node_mask]
            target = target[node_mask]
            mask = mask[node_mask]

        masked_pred = pred[mask]
        masked_target = target[mask]

        if masked_pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return F.mse_loss(masked_pred, masked_target)


class SpatialContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss using spatial neighbors.

    Positive pairs: connected nodes in the spatial graph (edges).
    Negative pairs: randomly sampled non-neighbor nodes.
    Encourages embeddings to preserve spatial locality.

    Args:
        temperature: Temperature scaling for similarity scores.
        n_negatives: Number of negative samples per anchor.
    """

    def __init__(self, temperature: float = 0.1, n_negatives: int = 256):
        super().__init__()
        self.temperature = temperature
        self.n_negatives = n_negatives

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute InfoNCE contrastive loss.

        Args:
            z: (N, D) node embeddings.
            edge_index: (2, E) edge indices defining positive pairs.
            node_mask: (N,) optional boolean mask to restrict anchors.

        Returns:
            Scalar InfoNCE loss.
        """
        n = z.shape[0]
        src, dst = edge_index

        # Filter to anchors in node_mask if provided
        if node_mask is not None:
            edge_mask = node_mask[src]
            src = src[edge_mask]
            dst = dst[edge_mask]

        if src.numel() == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Subsample edges if too many (for memory efficiency)
        max_edges = 4096
        if src.shape[0] > max_edges:
            perm = torch.randperm(src.shape[0], device=z.device)[:max_edges]
            src = src[perm]
            dst = dst[perm]

        # L2 normalize embeddings
        z_norm = F.normalize(z, p=2, dim=1)

        # Positive scores: similarity between connected nodes
        pos_scores = (z_norm[src] * z_norm[dst]).sum(dim=1) / self.temperature  # (E,)

        # Sample negatives for each anchor
        n_neg = min(self.n_negatives, n - 1)
        neg_idx = torch.randint(0, n, (src.shape[0], n_neg), device=z.device)

        # Negative scores: similarity between anchor and random nodes
        anchor_emb = z_norm[src].unsqueeze(1)  # (E, 1, D)
        neg_emb = z_norm[neg_idx]  # (E, n_neg, D)
        neg_scores = (anchor_emb * neg_emb).sum(dim=2) / self.temperature  # (E, n_neg)

        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # (E, 1+n_neg)
        labels = torch.zeros(src.shape[0], dtype=torch.long, device=z.device)
        loss = F.cross_entropy(logits, labels)

        return loss
