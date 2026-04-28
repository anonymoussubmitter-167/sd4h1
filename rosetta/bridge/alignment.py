"""FGW alignment solver for cross-species embedding alignment.

Uses POT's Fused Gromov-Wasserstein to compute optimal transport plans
between spatially-informed embeddings from different species.
"""

from __future__ import annotations

import logging
import sys

# POT (ot) imports TensorFlow internally via ot.backend, which can corrupt
# PyTorch's CUDA context if TF's CUDA initialisation fails silently.
# Block TF before POT loads so ot.backend treats it as absent.
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None  # type: ignore[assignment]

import numpy as np
import ot
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FGWAligner:
    """Fused Gromov-Wasserstein alignment between two species' embeddings.

    Combines feature-level cost (cosine distance between embeddings) with
    structure-level cost (spatial distance preservation) to compute an
    optimal transport plan.

    Args:
        alpha: FGW trade-off (0 = feature only, 1 = structure only).
        epsilon: Entropic regularization. 0 for exact FGW.
        max_iter: Maximum FGW iterations.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        epsilon: float = 0.005,
        max_iter: int = 200,
        sinkhorn_max_iter: int = 500,
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.sinkhorn_max_iter = sinkhorn_max_iter

    def compute_transport_plan(
        self,
        z_source: NDArray,
        z_target: NDArray,
        coords_source: NDArray,
        coords_target: NDArray,
    ) -> tuple[NDArray, float]:
        """Compute FGW transport plan between source and target.

        Args:
            z_source: (N_s, D) source embeddings.
            z_target: (N_t, D) target embeddings.
            coords_source: (N_s, 2) source spatial coordinates.
            coords_target: (N_t, 2) target spatial coordinates.

        Returns:
            T: (N_s, N_t) transport plan.
            fgw_dist: FGW distance value.
        """
        n_s = z_source.shape[0]
        n_t = z_target.shape[0]

        # Uniform marginals
        p = np.ones(n_s, dtype=np.float64) / n_s
        q = np.ones(n_t, dtype=np.float64) / n_t

        # Feature cost matrix M: cosine distance between embeddings
        M = _cosine_distance_matrix(z_source, z_target)

        # Normalize M to [0, 1] for numerical stability
        m_max = M.max()
        if m_max > 1e-12:
            M = M / m_max

        # Structure matrices: normalized pairwise spatial distances within each species
        C1 = _normalized_spatial_distance(coords_source)
        C2 = _normalized_spatial_distance(coords_target)

        # Ensure float64 for POT
        M = np.ascontiguousarray(M, dtype=np.float64)
        C1 = np.ascontiguousarray(C1, dtype=np.float64)
        C2 = np.ascontiguousarray(C2, dtype=np.float64)

        logger.info(
            "Computing FGW: N_s=%d, N_t=%d, alpha=%.2f, epsilon=%.3f",
            n_s, n_t, self.alpha, self.epsilon,
        )

        if self.epsilon > 0:
            # Log-domain entropic FGW (numerically stable at low epsilon)
            T, fgw_dist = self._entropic_fgw_log_domain(M, C1, C2, p, q)
        else:
            # Exact FGW
            T, log = ot.gromov.fused_gromov_wasserstein(
                M, C1, C2, p, q,
                loss_fun="square_loss",
                alpha=self.alpha,
                max_iter=self.max_iter,
                log=True,
            )
            fgw_dist = float(log.get("fgw_dist", 0.0))

        return np.array(T, dtype=np.float64), fgw_dist

    def _entropic_fgw_log_domain(
        self,
        M: NDArray,
        C1: NDArray,
        C2: NDArray,
        p: NDArray,
        q: NDArray,
    ) -> tuple[NDArray, float]:
        """Entropic FGW using log-domain Sinkhorn for numerical stability.

        Proximal gradient descent outer loop with log-domain Sinkhorn inner
        projection. Stable at low epsilon values (e.g., 0.005) where standard
        Sinkhorn underflows.
        """
        from ot.bregman import sinkhorn_log
        from ot.gromov._utils import gwggrad, gwloss, init_matrix

        constC, hC1, hC2 = init_matrix(C1, C2, p, q, "square_loss")
        T = np.outer(p, q)

        for it in range(self.max_iter):
            T_prev = T.copy()
            tens = self.alpha * gwggrad(constC, hC1, hC2, T) + (1 - self.alpha) * M
            T = sinkhorn_log(
                p, q, tens, self.epsilon,
                numItermax=self.sinkhorn_max_iter,
                stopThr=1e-9,
                warn=False,
            )
            if it % 10 == 0 and np.linalg.norm(T - T_prev) < 1e-9:
                break

        fgw_dist = float(
            (1 - self.alpha) * np.sum(M * T)
            + self.alpha * gwloss(constC, hC1, hC2, T)
        )
        return T, fgw_dist


def barycentric_projection(
    T: NDArray,
    z_target: NDArray,
) -> NDArray:
    """Project target embeddings into source space using transport plan.

    Computes weighted average of target embeddings for each source spot,
    weighted by the transport plan.

    Args:
        T: (N_s, N_t) transport plan.
        z_target: (N_t, D) target embeddings.

    Returns:
        z_projected: (N_s, D) projected embeddings in source space.
    """
    # Normalize rows of T to get per-source weights
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    T_normalized = T / row_sums
    return T_normalized @ z_target


def subsample_for_fgw(
    z: NDArray,
    coords: NDArray,
    max_nodes: int,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Subsample embeddings and coordinates for FGW memory efficiency.

    Args:
        z: (N, D) embeddings.
        coords: (N, 2) spatial coordinates.
        max_nodes: Maximum number of nodes.
        rng: Random generator.

    Returns:
        z_sub: (M, D) subsampled embeddings.
        coords_sub: (M, 2) subsampled coordinates.
        indices: (M,) indices of selected nodes.
    """
    n = z.shape[0]
    if n <= max_nodes:
        return z, coords, np.arange(n)

    if rng is None:
        rng = np.random.default_rng(42)

    indices = rng.choice(n, size=max_nodes, replace=False)
    indices.sort()
    return z[indices], coords[indices], indices


def _cosine_distance_matrix(X: NDArray, Y: NDArray) -> NDArray:
    """Compute pairwise cosine distance matrix between X and Y.

    Returns (N_x, N_y) matrix where entry (i, j) = 1 - cosine_sim(X[i], Y[j]).
    """
    # L2 normalize
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    sim = X_norm @ Y_norm.T
    # Clip to avoid numerical issues
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def _normalized_spatial_distance(coords: NDArray) -> NDArray:
    """Compute normalized pairwise spatial distance matrix.

    Normalizes by the maximum distance so values are in [0, 1].
    """
    from scipy.spatial.distance import cdist

    D = cdist(coords, coords, metric="euclidean")
    d_max = D.max()
    if d_max > 1e-12:
        D = D / d_max
    return D
