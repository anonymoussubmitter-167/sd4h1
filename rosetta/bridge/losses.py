"""Cross-species loss functions for BRIDGE alignment training.

Three losses:
1. CrossSpeciesContrastiveLoss — InfoNCE using transport plan for positive pairs
2. FGWAlignmentLoss — Differentiable alignment loss guided by fixed transport plan
3. MMDLoss — Maximum Mean Discrepancy for global distribution matching
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossSpeciesContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss using FGW transport plan for cross-species pairs.

    For each source spot i, the top-k target spots by T[i,:] are positives.
    Negatives are randomly sampled from the opposite species.

    Args:
        top_k: Number of top matches from transport plan to use as positives.
        temperature: Temperature scaling for similarity scores.
        n_negatives: Number of negative samples per anchor.
    """

    def __init__(
        self,
        top_k: int = 5,
        temperature: float = 0.1,
        n_negatives: int = 256,
    ):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.n_negatives = n_negatives

    def forward(
        self,
        z_source: Tensor,
        z_target: Tensor,
        T: Tensor,
    ) -> Tensor:
        """Compute cross-species contrastive loss.

        Args:
            z_source: (N_s, D) source embeddings.
            z_target: (N_t, D) target embeddings.
            T: (N_s, N_t) transport plan (detached, from FGW step).

        Returns:
            Scalar InfoNCE loss.
        """
        n_s = z_source.shape[0]
        n_t = z_target.shape[0]

        if n_s == 0 or n_t == 0:
            return torch.tensor(0.0, device=z_source.device, requires_grad=True)

        # Find top-k matches for each source spot
        k = min(self.top_k, n_t)
        _, top_k_idx = torch.topk(T, k, dim=1)  # (N_s, k)

        # L2 normalize embeddings
        z_s_norm = F.normalize(z_source, p=2, dim=1)
        z_t_norm = F.normalize(z_target, p=2, dim=1)

        # Subsample anchors if too many
        max_anchors = 4096
        if n_s > max_anchors:
            perm = torch.randperm(n_s, device=z_source.device)[:max_anchors]
            z_s_norm = z_s_norm[perm]
            top_k_idx = top_k_idx[perm]
            n_s = max_anchors

        # Positive scores: average similarity to top-k matches
        pos_emb = z_t_norm[top_k_idx]  # (N_s, k, D)
        pos_scores = (z_s_norm.unsqueeze(1) * pos_emb).sum(dim=2) / self.temperature  # (N_s, k)

        # Sample negatives from target species
        n_neg = min(self.n_negatives, n_t - 1)
        if n_neg <= 0:
            return torch.tensor(0.0, device=z_source.device, requires_grad=True)

        neg_idx = torch.randint(0, n_t, (n_s, n_neg), device=z_source.device)
        neg_emb = z_t_norm[neg_idx]  # (N_s, n_neg, D)
        neg_scores = (z_s_norm.unsqueeze(1) * neg_emb).sum(dim=2) / self.temperature  # (N_s, n_neg)

        # InfoNCE: for each anchor, treat each positive separately
        # logits = [pos_i, neg_1, ..., neg_n] for each positive
        loss = torch.tensor(0.0, device=z_source.device, requires_grad=True)
        for p in range(k):
            logits = torch.cat([pos_scores[:, p:p+1], neg_scores], dim=1)  # (N_s, 1+n_neg)
            labels = torch.zeros(n_s, dtype=torch.long, device=z_source.device)
            loss = loss + F.cross_entropy(logits, labels)
        loss = loss / k

        return loss


class FGWAlignmentLoss(nn.Module):
    """Differentiable alignment loss guided by fixed transport plan.

    Computes sum(T * M) where T is the fixed transport plan and M is the
    cosine distance matrix between current embeddings. Provides gradient
    signal to push matched pairs closer.
    """

    def forward(
        self,
        z_source: Tensor,
        z_target: Tensor,
        T: Tensor,
    ) -> Tensor:
        """Compute alignment loss.

        Args:
            z_source: (N_s, D) source embeddings.
            z_target: (N_t, D) target embeddings.
            T: (N_s, N_t) transport plan (detached).

        Returns:
            Scalar alignment loss.
        """
        # Cosine distance matrix
        z_s_norm = F.normalize(z_source, p=2, dim=1)
        z_t_norm = F.normalize(z_target, p=2, dim=1)
        M = 1.0 - z_s_norm @ z_t_norm.t()  # (N_s, N_t)

        # Weighted sum: T is detached, gradients flow through M
        loss = (T * M).sum()
        return loss


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss for global distribution matching.

    Forces the projected embedding distributions of two species to overlap
    globally, not just at transport-matched pairs. Uses multi-scale RBF
    kernel for robustness across different embedding scales.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    """

    def __init__(self, bandwidths: tuple[float, ...] = (0.1, 1.0, 10.0)):
        super().__init__()
        self.bandwidths = bandwidths

    def forward(self, z_source: Tensor, z_target: Tensor) -> Tensor:
        """Compute MMD^2 between source and target embeddings.

        Args:
            z_source: (N_s, D) source projected embeddings.
            z_target: (N_t, D) target projected embeddings.

        Returns:
            Scalar MMD^2 loss (non-negative).
        """
        n_s = z_source.shape[0]
        n_t = z_target.shape[0]

        if n_s == 0 or n_t == 0:
            return torch.tensor(0.0, device=z_source.device, requires_grad=True)

        # Pairwise squared distances
        xx = torch.cdist(z_source, z_source).pow(2)  # (N_s, N_s)
        yy = torch.cdist(z_target, z_target).pow(2)  # (N_t, N_t)
        xy = torch.cdist(z_source, z_target).pow(2)  # (N_s, N_t)

        # Multi-scale RBF kernel
        k_xx = torch.zeros_like(xx)
        k_yy = torch.zeros_like(yy)
        k_xy = torch.zeros_like(xy)

        for bw in self.bandwidths:
            k_xx = k_xx + torch.exp(-xx / (2.0 * bw))
            k_yy = k_yy + torch.exp(-yy / (2.0 * bw))
            k_xy = k_xy + torch.exp(-xy / (2.0 * bw))

        mmd2 = k_xx.sum() / (n_s * n_s) + k_yy.sum() / (n_t * n_t) - 2.0 * k_xy.sum() / (n_s * n_t)
        return mmd2
