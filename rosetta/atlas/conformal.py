"""Conformal prediction for annotation transfer with uncertainty quantification.

Provides valid prediction sets with guaranteed coverage: for a given
significance level alpha, the true label is in the prediction set with
probability >= 1 - alpha, regardless of the distribution.

Workflow:
1. Split labeled data into proper training set + calibration set.
2. Fit kNN on training set.
3. Compute nonconformity scores on calibration set.
4. For unlabeled data, produce prediction sets using calibration quantile.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def _knn_class_probabilities(
    z_train: NDArray,
    z_query: NDArray,
    labels_train: NDArray,
    classes: NDArray,
    k: int = 20,
) -> NDArray:
    """Compute class probability estimates via kNN voting.

    Args:
        z_train: (N_train, D) training embeddings.
        z_query: (N_query, D) query embeddings.
        labels_train: (N_train,) training labels.
        classes: (C,) unique class labels.
        k: Number of neighbors.

    Returns:
        probs: (N_query, C) class probabilities (soft voting).
    """
    tree = cKDTree(z_train)
    dists, indices = tree.query(z_query, k=k)

    # Distance-weighted voting: weight = 1 / (dist + eps)
    weights = 1.0 / (dists + 1e-8)  # (N_query, k)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_query = z_query.shape[0]
    n_classes = len(classes)
    probs = np.zeros((n_query, n_classes), dtype=np.float64)

    neighbor_labels = labels_train[indices]  # (N_query, k)
    for i in range(n_query):
        for j in range(k):
            c_idx = class_to_idx.get(neighbor_labels[i, j])
            if c_idx is not None:
                probs[i, c_idx] += weights[i, j]

    # Normalize to probabilities
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    probs /= row_sums

    return probs


def calibrate_conformal(
    z_train: NDArray,
    z_cal: NDArray,
    labels_train: NDArray,
    labels_cal: NDArray,
    classes: NDArray,
    k: int = 20,
) -> NDArray:
    """Compute nonconformity scores on calibration set.

    The nonconformity score for sample i is 1 - P(true_label_i | x_i),
    where P is the kNN-based class probability.

    Args:
        z_train: (N_train, D) training embeddings.
        z_cal: (N_cal, D) calibration embeddings.
        labels_train: (N_train,) training labels.
        labels_cal: (N_cal,) calibration labels.
        classes: (C,) unique class labels.
        k: Number of neighbors.

    Returns:
        scores: (N_cal,) nonconformity scores in [0, 1].
    """
    probs = _knn_class_probabilities(z_train, z_cal, labels_train, classes, k)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    scores = np.zeros(len(labels_cal), dtype=np.float64)
    for i, label in enumerate(labels_cal):
        c_idx = class_to_idx.get(label)
        if c_idx is not None:
            scores[i] = 1.0 - probs[i, c_idx]
        else:
            scores[i] = 1.0  # unknown class -> max nonconformity

    return scores


def conformal_prediction_sets(
    z_train: NDArray,
    z_test: NDArray,
    labels_train: NDArray,
    classes: NDArray,
    cal_scores: NDArray,
    alpha: float = 0.1,
    k: int = 20,
) -> tuple[list[list[str]], NDArray, NDArray]:
    """Produce conformal prediction sets for test data.

    For each test sample, includes all classes whose probability exceeds
    1 - q_hat, where q_hat is the (1-alpha) quantile of calibration scores.

    Args:
        z_train: (N_train, D) training embeddings.
        z_test: (N_test, D) test embeddings.
        labels_train: (N_train,) training labels.
        classes: (C,) unique class labels.
        cal_scores: (N_cal,) nonconformity scores from calibration.
        alpha: Significance level (e.g., 0.1 for 90% coverage).
        k: Number of neighbors.

    Returns:
        prediction_sets: list of lists of class labels (one set per test sample).
        set_sizes: (N_test,) size of each prediction set.
        probs: (N_test, C) underlying class probabilities.
    """
    # Compute quantile threshold
    n_cal = len(cal_scores)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_hat = np.quantile(cal_scores, min(q_level, 1.0))

    logger.info("Conformal: alpha=%.2f, q_hat=%.4f, n_cal=%d", alpha, q_hat, n_cal)

    # Get class probabilities for test data
    probs = _knn_class_probabilities(z_train, z_test, labels_train, classes, k)

    # Build prediction sets: include class c if 1 - P(c|x) <= q_hat
    prediction_sets = []
    set_sizes = np.zeros(len(z_test), dtype=np.int32)

    for i in range(len(z_test)):
        pred_set = []
        for c_idx, c in enumerate(classes):
            if 1.0 - probs[i, c_idx] <= q_hat:
                pred_set.append(str(c))
        # Always include at least the most probable class
        if not pred_set:
            best_idx = np.argmax(probs[i])
            pred_set = [str(classes[best_idx])]
        prediction_sets.append(pred_set)
        set_sizes[i] = len(pred_set)

    return prediction_sets, set_sizes, probs


def conformal_label_transfer(
    z_labeled: NDArray,
    z_unlabeled: NDArray,
    labels: NDArray,
    alpha: float = 0.1,
    cal_fraction: float = 0.2,
    k: int = 20,
    random_state: int = 42,
) -> dict:
    """Full conformal prediction pipeline for cross-species label transfer.

    Splits labeled data into train/calibration, calibrates nonconformity
    scores, and produces prediction sets for unlabeled data.

    Args:
        z_labeled: (N_lab, D) labeled embeddings (from species with annotations).
        z_unlabeled: (N_unlab, D) unlabeled embeddings (from other species).
        labels: (N_lab,) cell type labels.
        alpha: Significance level (0.1 → 90% coverage guarantee).
        cal_fraction: Fraction of labeled data for calibration.
        k: Number of neighbors.
        random_state: Random seed for train/cal split.

    Returns:
        dict with:
            'predicted_labels': (N_unlab,) most probable label per cell.
            'prediction_sets': list of prediction sets.
            'set_sizes': (N_unlab,) size of each prediction set.
            'probabilities': (N_unlab, C) class probabilities.
            'classes': (C,) class labels.
            'coverage_guarantee': float, 1 - alpha.
            'mean_set_size': float.
            'singleton_fraction': float, fraction with exactly 1 prediction.
            'calibration_quantile': float.
    """
    classes = np.unique(labels)
    n_labeled = len(labels)

    # Stratified split: maintain class proportions in train/cal
    rng = np.random.default_rng(random_state)
    train_idx = []
    cal_idx = []
    for c in classes:
        c_indices = np.where(labels == c)[0]
        n_cal = max(1, int(len(c_indices) * cal_fraction))
        perm = rng.permutation(len(c_indices))
        cal_idx.extend(c_indices[perm[:n_cal]])
        train_idx.extend(c_indices[perm[n_cal:]])

    train_idx = np.array(train_idx)
    cal_idx = np.array(cal_idx)

    z_train = z_labeled[train_idx]
    z_cal = z_labeled[cal_idx]
    labels_train = labels[train_idx]
    labels_cal = labels[cal_idx]

    logger.info("Conformal split: %d train, %d calibration, %d classes",
                len(train_idx), len(cal_idx), len(classes))

    # Calibrate
    cal_scores = calibrate_conformal(
        z_train, z_cal, labels_train, labels_cal, classes, k=k,
    )

    # Predict
    prediction_sets, set_sizes, probs = conformal_prediction_sets(
        z_train, z_unlabeled, labels_train, classes, cal_scores,
        alpha=alpha, k=k,
    )

    # Most probable label
    predicted_labels = np.array([classes[np.argmax(probs[i])] for i in range(len(z_unlabeled))])

    # Calibration coverage check (on cal set itself)
    cal_pred_sets, cal_sizes, _ = conformal_prediction_sets(
        z_train, z_cal, labels_train, classes, cal_scores,
        alpha=alpha, k=k,
    )
    cal_coverage = np.mean([
        str(labels_cal[i]) in cal_pred_sets[i]
        for i in range(len(labels_cal))
    ])

    logger.info("Calibration coverage: %.3f (target: %.3f)", cal_coverage, 1 - alpha)
    logger.info("Mean prediction set size: %.2f", np.mean(set_sizes))
    logger.info("Singleton fraction: %.3f", np.mean(set_sizes == 1))

    return {
        "predicted_labels": predicted_labels,
        "prediction_sets": prediction_sets,
        "set_sizes": set_sizes,
        "probabilities": probs,
        "classes": classes,
        "coverage_guarantee": 1 - alpha,
        "mean_set_size": float(np.mean(set_sizes)),
        "singleton_fraction": float(np.mean(set_sizes == 1)),
        "calibration_coverage": float(cal_coverage),
        "calibration_quantile": float(
            np.quantile(cal_scores, min(
                np.ceil((len(cal_scores) + 1) * (1 - alpha)) / len(cal_scores), 1.0
            ))
        ),
    }
