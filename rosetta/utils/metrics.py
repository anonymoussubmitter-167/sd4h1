"""Metric functions for evaluating spatial transcriptomics integration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors


def morans_i(
    embeddings: NDArray,
    spatial_coords: NDArray,
    k: int = 6,
) -> float:
    """Compute Moran's I spatial autocorrelation of embeddings.

    Measures whether nearby spots have similar embeddings. Trained
    embeddings should have Moran's I >> 0 (random ~ 0).

    Uses KNN spatial weight matrix for efficiency.

    Args:
        embeddings: (N, D) embedding matrix.
        spatial_coords: (N, 2) spatial coordinates.
        k: Number of spatial neighbors for weight matrix.

    Returns:
        Mean Moran's I across embedding dimensions.
    """
    n = embeddings.shape[0]
    if n < k + 1:
        return 0.0

    # Build KNN spatial weight matrix
    tree = cKDTree(spatial_coords)
    _, indices = tree.query(spatial_coords, k=k + 1)
    indices = indices[:, 1:]  # remove self

    # Compute Moran's I per embedding dimension, then average
    morans_values = []
    for d in range(embeddings.shape[1]):
        x = embeddings[:, d]
        x_mean = x.mean()
        x_dev = x - x_mean
        ss = np.sum(x_dev ** 2)

        if ss < 1e-12:
            morans_values.append(0.0)
            continue

        # Sum of w_ij * (x_i - mean) * (x_j - mean) over all neighbor pairs
        neighbor_devs = x_dev[indices]  # (N, k)
        cross = x_dev[:, np.newaxis] * neighbor_devs  # (N, k)
        w_sum = cross.sum()

        # Total weight count (binary weights, each node has k neighbors)
        total_w = n * k

        morans_i_d = (n / total_w) * (w_sum / ss)
        morans_values.append(float(morans_i_d))

    return float(np.mean(morans_values))


def silhouette_score_spatial(
    embeddings: NDArray,
    cluster_labels: NDArray,
    spatial_coords: NDArray | None = None,
) -> float:
    """Compute silhouette score, optionally weighted by spatial proximity.

    Args:
        embeddings: (N, D) embedding matrix.
        cluster_labels: (N,) integer cluster labels.
        spatial_coords: (N, 2) optional spatial coordinates (unused for now,
            standard silhouette in embedding space).

    Returns:
        Silhouette score in [-1, 1].
    """
    from sklearn.metrics import silhouette_score

    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) < 2:
        return 0.0
    if len(unique_labels) >= len(embeddings):
        return 0.0

    return float(silhouette_score(embeddings, cluster_labels))


def reconstruction_accuracy(
    pred: NDArray,
    target: NDArray,
    mask: NDArray,
) -> dict[str, float]:
    """Compute reconstruction metrics on masked positions.

    Args:
        pred: (N, G) predicted expression.
        target: (N, G) original expression.
        mask: (N, G) boolean mask where True = masked positions.

    Returns:
        Dict with 'mse', 'pearson_r', and 'cosine_sim'.
    """
    masked_pred = pred[mask]
    masked_target = target[mask]

    if len(masked_pred) == 0:
        return {"mse": 0.0, "pearson_r": 0.0, "cosine_sim": 0.0}

    # MSE
    mse = float(np.mean((masked_pred - masked_target) ** 2))

    # Pearson correlation
    if np.std(masked_pred) < 1e-12 or np.std(masked_target) < 1e-12:
        pearson = 0.0
    else:
        pearson = float(pearsonr(masked_pred, masked_target)[0])

    # Cosine similarity
    norm_pred = np.linalg.norm(masked_pred)
    norm_target = np.linalg.norm(masked_target)
    if norm_pred < 1e-12 or norm_target < 1e-12:
        cosine = 0.0
    else:
        cosine = float(np.dot(masked_pred, masked_target) / (norm_pred * norm_target))

    return {"mse": mse, "pearson_r": pearson, "cosine_sim": cosine}


def alignment_score(
    embeddings: NDArray,
    species_labels: NDArray,
    k: int = 50,
) -> float:
    """Compute cross-species alignment score using kNN mixing, normalized.

    For each spot, computes the fraction of k-nearest neighbors that
    come from the other species, then normalizes by the expected fraction
    under random mixing to account for class imbalance.

    Returns 1.0 for perfect mixing, 0.0 for no mixing. Values > 1 are
    possible if species are more mixed than random.

    Args:
        embeddings: (N, D) combined embedding matrix (both species).
        species_labels: (N,) binary labels (0 = species A, 1 = species B).
        k: Number of nearest neighbors.

    Returns:
        Normalized mixing score in [0, ~1]. Higher = better mixing.
    """
    n = embeddings.shape[0]
    if n < k + 1:
        k = max(1, n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    indices = indices[:, 1:]  # remove self (first neighbor is always self)

    # Vectorized: fraction of neighbors from opposite species
    neighbor_species = species_labels[indices]  # (N, k)
    different = neighbor_species != species_labels[:, np.newaxis]  # (N, k)
    raw_mixing = float(np.mean(different))

    # Expected mixing under random: for each cell of species A, the
    # probability of picking species B from the remaining N-1 cells is
    # n_B / (N - 1), and vice versa.  The overall expected mixing is
    # weighted by species proportions.
    n_a = int(np.sum(species_labels == 0))
    n_b = n - n_a
    if n <= 1 or n_a == 0 or n_b == 0:
        return 0.0
    expected = (n_a * n_b + n_b * n_a) / (n * (n - 1))  # = 2*n_a*n_b / (n*(n-1))
    if expected < 1e-12:
        return 0.0

    return float(raw_mixing / expected)


def label_transfer_accuracy(
    z_source: NDArray,
    z_target: NDArray,
    labels_source: NDArray,
    labels_target: NDArray,
    k: int = 10,
) -> float:
    """Compute label transfer accuracy using distance-weighted kNN classifier.

    Trains a kNN classifier on source embeddings + labels, evaluates
    on target embeddings + labels. Uses inverse-distance weighting.

    Args:
        z_source: (N_s, D) source embeddings.
        z_target: (N_t, D) target embeddings.
        labels_source: (N_s,) source labels (string or int).
        labels_target: (N_t,) target labels (string or int).
        k: Number of neighbors for kNN.

    Returns:
        Classification accuracy on target set.
    """
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    nn.fit(z_source)
    distances, indices = nn.kneighbors(z_target)

    # Encode labels to integers for vectorised aggregation
    unique_labels, labels_source_int = np.unique(labels_source, return_inverse=True)
    _, labels_target_int = np.unique(labels_target, return_inverse=True)
    n_classes = len(unique_labels)
    n_t = z_target.shape[0]

    # neighbor label ints: (N_t, k)
    neighbor_label_ints = labels_source_int[indices]
    # inverse-distance weights: (N_t, k)
    weights = 1.0 / (distances + 1e-8)

    # Accumulate weighted votes: (N_t, n_classes)
    vote_matrix = np.zeros((n_t, n_classes), dtype=np.float64)
    cell_idx = np.repeat(np.arange(n_t), k)
    np.add.at(vote_matrix, (cell_idx, neighbor_label_ints.ravel()), weights.ravel())

    predicted_ints = vote_matrix.argmax(axis=1)

    # Map target labels to the same encoding
    unique_src = unique_labels
    tgt_ints = np.searchsorted(unique_src, labels_target)
    tgt_ints = np.clip(tgt_ints, 0, n_classes - 1)
    match = unique_src[tgt_ints] == labels_target  # handle labels not in source
    correct = np.sum((predicted_ints == tgt_ints) & match)
    return float(correct) / n_t


def spatial_label_smoothness(
    coords: NDArray,
    labels: NDArray,
    k: int = 10,
) -> float:
    """Compute spatial label smoothness: fraction of spatial kNN sharing the same label.

    For each cell, computes the fraction of k spatial nearest neighbors that
    share the same label. Averaged over all cells. Random baseline ~ 1/C
    where C = number of unique classes.

    Args:
        coords: (N, 2) spatial coordinates.
        labels: (N,) categorical labels (string or int).
        k: Number of spatial neighbors.

    Returns:
        Mean fraction of same-label spatial neighbors in [0, 1].
    """
    n = len(labels)
    if n < k + 1:
        k = max(1, n - 1)

    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    indices = indices[:, 1:]  # remove self

    # Vectorized comparison
    neighbor_labels = labels[indices]  # (N, k)
    same = neighbor_labels == labels[:, np.newaxis]  # (N, k)
    return float(np.mean(same))


def label_transfer_with_confidence(
    z_source: NDArray,
    z_target: NDArray,
    labels_source: NDArray,
    k: int = 10,
) -> tuple[NDArray, NDArray]:
    """Transfer labels from source to target via distance-weighted kNN.

    Like label_transfer_accuracy but does not require ground truth on target.
    Confidence = fraction of total weight going to the winning label.

    Args:
        z_source: (N_s, D) source embeddings.
        z_target: (N_t, D) target embeddings.
        labels_source: (N_s,) source labels (string or int).
        k: Number of neighbors for kNN.

    Returns:
        predicted_labels: (N_t,) predicted labels for each target cell.
        confidences: (N_t,) confidence scores in [0, 1].
    """
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    nn.fit(z_source)
    distances, indices = nn.kneighbors(z_target)

    n_t = z_target.shape[0]

    # Encode source labels to integers
    unique_labels, labels_source_int = np.unique(labels_source, return_inverse=True)
    n_classes = len(unique_labels)

    # neighbor label ints: (N_t, k)
    neighbor_label_ints = labels_source_int[indices]
    # inverse-distance weights: (N_t, k)
    weights = 1.0 / (distances + 1e-8)

    # Accumulate weighted votes: (N_t, n_classes)
    vote_matrix = np.zeros((n_t, n_classes), dtype=np.float64)
    cell_idx = np.repeat(np.arange(n_t), k)
    np.add.at(vote_matrix, (cell_idx, neighbor_label_ints.ravel()), weights.ravel())

    winner_ints = vote_matrix.argmax(axis=1)
    total_weights = weights.sum(axis=1)
    winner_weights = vote_matrix[np.arange(n_t), winner_ints]

    predicted_labels = unique_labels[winner_ints]
    confidences = winner_weights / (total_weights + 1e-12)

    return predicted_labels, confidences


