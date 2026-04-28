#!/usr/bin/env python
"""Compute per-cell-type alignment metrics for ROSETTA cross-species label transfer.

For each tissue pair, loads transferred labels (.npz) and the unlabeled species'
expression data (.h5ad), computes Leiden pseudo-labels at the best resolution,
then reports per-class transfer statistics and Hungarian-matched precision/recall/F1.

Usage:
    python scripts/per_celltype_metrics.py

Output: Markdown tables on stdout.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Tissue pair definitions
# ---------------------------------------------------------------------------

TISSUE_PAIRS = [
    {
        "name": "Human<>Mouse Brain",
        "pair_dir": "visium_human_brain_merfish_mouse_brain",
        "npz_candidates": [
            "seeds/seed0/evaluation/transferred_labels_class.npz",
            "evaluation/transferred_labels_class.npz",
        ],
        "metrics_candidates": [
            "seeds/seed0/evaluation/metrics.json",
            "evaluation/metrics.json",
        ],
        "unlabeled_adata": "visium_human_brain.h5ad",
        "labeled_adata": "merfish_mouse_brain.h5ad",
        "label_col": "class",
    },
    {
        "name": "Mouse<>Zebrafish Brain",
        "pair_dir": "merfish_mouse_brain_stereoseq_zebrafish_brain",
        "npz_candidates": [
            "evaluation/transferred_labels_class.npz",
        ],
        "metrics_candidates": [
            "evaluation/metrics.json",
        ],
        "unlabeled_adata": "stereoseq_zebrafish_brain.h5ad",
        "labeled_adata": "merfish_mouse_brain.h5ad",
        "label_col": "class",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_first(base_dir: Path, candidates: list[str]) -> Path | None:
    """Return the first candidate path that exists under base_dir."""
    for c in candidates:
        p = base_dir / c
        if p.exists():
            return p
    return None


def _spatial_smoothness_per_class(
    coords: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """Compute per-class spatial smoothness (fraction of k spatial NN with same label).

    Returns a dict mapping each unique label to its mean smoothness score.
    """
    n = len(labels)
    k_actual = min(k, n - 1)
    if k_actual < 1:
        return {}

    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k_actual + 1)
    indices = indices[:, 1:]  # remove self

    neighbor_labels = labels[indices]  # (N, k)
    same = neighbor_labels == labels[:, np.newaxis]  # (N, k)
    per_spot_smoothness = np.mean(same, axis=1)  # (N,)

    result = {}
    for label in np.unique(labels):
        mask = labels == label
        result[label] = float(np.mean(per_spot_smoothness[mask]))
    return result


def _load_adata_for_leiden(h5ad_path: Path) -> ad.AnnData:
    """Load and minimally preprocess an AnnData for Leiden clustering."""
    adata = ad.read_h5ad(h5ad_path)
    adata = adata.copy()
    # Normalize if not already done (check if values look raw)
    X = adata.X
    if sp.issparse(X):
        max_val = X.max()
    else:
        max_val = np.max(X)

    # If max value is large, data is likely raw counts
    if max_val > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # HVG selection for PCA
    if adata.n_vars > 3000:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=3000)
            adata_pca = adata[:, adata.var["highly_variable"]].copy()
        except Exception:
            # Fallback: top by variance
            X_dense = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
            var = np.var(X_dense, axis=0)
            top_idx = np.argsort(var)[-3000:]
            adata_pca = adata[:, top_idx].copy()
    else:
        adata_pca = adata.copy()

    sc.pp.pca(adata_pca, n_comps=min(50, adata_pca.n_vars - 1))
    sc.pp.neighbors(adata_pca, n_neighbors=15)
    return adata_pca


def _hungarian_match_labels(
    transferred: np.ndarray,
    leiden: np.ndarray,
) -> dict[str, str]:
    """Compute Hungarian matching between transferred cell-type labels and Leiden clusters.

    Returns a mapping from each Leiden cluster ID to its best-matched transferred label.
    """
    transferred_classes = np.unique(transferred)
    leiden_classes = np.unique(leiden)

    # Build a contingency (cost) matrix: rows=transferred, cols=Leiden
    # We want to maximize overlap, so cost = -count
    n_trans = len(transferred_classes)
    n_leid = len(leiden_classes)
    cost = np.zeros((n_trans, n_leid), dtype=np.int64)

    trans_to_idx = {c: i for i, c in enumerate(transferred_classes)}
    leid_to_idx = {c: i for i, c in enumerate(leiden_classes)}

    for t_label, l_label in zip(transferred, leiden):
        cost[trans_to_idx[t_label], leid_to_idx[l_label]] += 1

    # Hungarian assignment minimizes cost; we negate to maximize overlap
    row_ind, col_ind = linear_sum_assignment(-cost)

    # Build mapping: leiden_cluster -> matched transferred label
    leiden_to_transferred = {}
    for r, c in zip(row_ind, col_ind):
        leiden_to_transferred[leiden_classes[c]] = transferred_classes[r]

    return leiden_to_transferred


def _compute_hungarian_prf(
    transferred: np.ndarray,
    leiden: np.ndarray,
) -> dict:
    """Compute per-class precision, recall, F1 after Hungarian matching.

    The Hungarian algorithm finds the optimal 1-to-1 assignment between
    transferred cell types and Leiden clusters. We then relabel Leiden
    clusters according to this matching and compute P/R/F1 per class.

    Returns dict with per-class and macro-average metrics.
    """
    leiden_to_trans = _hungarian_match_labels(transferred, leiden)

    # Relabel Leiden according to Hungarian matching
    matched_leiden = np.array(
        [leiden_to_trans.get(l, "__unmatched__") for l in leiden],
        dtype=str,
    )

    # Only evaluate on classes present in both transferred and matched
    all_classes = sorted(set(np.unique(transferred)) | set(np.unique(matched_leiden)))
    # Remove __unmatched__ from evaluation classes
    eval_classes = [c for c in all_classes if c != "__unmatched__"]

    if not eval_classes:
        return {"per_class": {}, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    # sklearn precision_recall_fscore_support with labels and zero_division
    precision, recall, f1, support = precision_recall_fscore_support(
        transferred, matched_leiden, labels=eval_classes, average=None, zero_division=0,
    )

    per_class = {}
    for i, cls in enumerate(eval_classes):
        per_class[cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    # Macro averages (only over classes that actually appear in transferred labels)
    transferred_unique = set(np.unique(transferred))
    valid_idx = [i for i, cls in enumerate(eval_classes) if cls in transferred_unique]
    if valid_idx:
        macro_p = float(np.mean(precision[valid_idx]))
        macro_r = float(np.mean(recall[valid_idx]))
        macro_f1 = float(np.mean(f1[valid_idx]))
    else:
        macro_p = macro_r = macro_f1 = 0.0

    return {
        "per_class": per_class,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "leiden_to_transferred": leiden_to_trans,
    }


# ---------------------------------------------------------------------------
# Main analysis per tissue pair
# ---------------------------------------------------------------------------


def analyze_tissue_pair(pair_config: dict) -> str:
    """Analyze one tissue pair and return a markdown-formatted report string."""
    name = pair_config["name"]
    pair_dir = OUTPUTS_DIR / pair_config["pair_dir"]
    label_col = pair_config["label_col"]

    lines = []
    lines.append(f"## {name}")
    lines.append("")

    # -----------------------------------------------------------------------
    # 1. Load transferred labels
    # -----------------------------------------------------------------------
    npz_path = _resolve_first(pair_dir, pair_config["npz_candidates"])
    if npz_path is None:
        lines.append(f"**ERROR**: No transferred labels found for {name}")
        lines.append(f"  Searched: {[str(pair_dir / c) for c in pair_config['npz_candidates']]}")
        lines.append("")
        return "\n".join(lines)

    logger.info("Loading transferred labels from %s", npz_path)
    npz_data = np.load(npz_path, allow_pickle=True)
    transferred_labels = npz_data["labels"]
    confidences = npz_data["confidences"]
    n_spots = len(transferred_labels)
    logger.info("  Loaded %d spots, %d unique classes", n_spots, len(np.unique(transferred_labels)))

    lines.append(f"**Source**: `{npz_path.relative_to(PROJECT_ROOT)}`")
    lines.append(f"**N spots (unlabeled species)**: {n_spots}")
    lines.append("")

    # -----------------------------------------------------------------------
    # 2. Load metrics.json (for best Leiden resolution)
    # -----------------------------------------------------------------------
    metrics_path = _resolve_first(pair_dir, pair_config["metrics_candidates"])
    best_resolution = 0.5  # default
    if metrics_path is not None:
        with open(metrics_path) as f:
            metrics = json.load(f)
        key = f"pseudo_validation_{label_col}_best_resolution"
        if key in metrics:
            best_resolution = metrics[key]
            logger.info("  Best Leiden resolution from metrics.json: %.1f", best_resolution)
    else:
        logger.warning("  No metrics.json found, using default resolution=%.1f", best_resolution)

    # -----------------------------------------------------------------------
    # 3. Load unlabeled species adata + compute Leiden
    # -----------------------------------------------------------------------
    unlabeled_h5ad = PROCESSED_DIR / pair_config["unlabeled_adata"]
    logger.info("Loading unlabeled adata from %s", unlabeled_h5ad)
    adata_unlabeled = ad.read_h5ad(unlabeled_h5ad)
    coords = np.array(adata_unlabeled.obsm["spatial"][:, :2], dtype=np.float32)

    # The transferred labels may be from preprocessed (filtered) data.
    # If sizes don't match, we truncate coords to match transferred labels.
    if len(coords) != n_spots:
        logger.warning(
            "  Size mismatch: adata has %d spots, transferred labels have %d. "
            "Attempting to align (taking first %d).",
            len(coords), n_spots, n_spots,
        )
        # This can happen if adata was filtered during preprocessing.
        # We truncate to the smaller size for the spatial smoothness computation.
        n_use = min(len(coords), n_spots)
        coords = coords[:n_use]
        transferred_labels = transferred_labels[:n_use]
        confidences = confidences[:n_use]
        n_spots = n_use

    # Leiden clustering on the unlabeled species
    logger.info("Computing Leiden clustering (resolution=%.1f) on unlabeled species...", best_resolution)
    adata_cluster = _load_adata_for_leiden(unlabeled_h5ad)

    # If clustering adata has different size (due to filtering), align
    n_cluster = adata_cluster.n_obs
    if n_cluster != n_spots:
        logger.warning(
            "  Clustering adata has %d cells vs %d transferred labels. "
            "Using min(%d, %d) cells.",
            n_cluster, n_spots, n_cluster, n_spots,
        )
        n_use = min(n_cluster, n_spots)
        transferred_labels_leiden = transferred_labels[:n_use]
    else:
        n_use = n_spots
        transferred_labels_leiden = transferred_labels

    sc.tl.leiden(adata_cluster, resolution=best_resolution, key_added="leiden_best")
    leiden_labels = np.array(adata_cluster.obs["leiden_best"].values[:n_use], dtype=str)
    n_leiden = len(np.unique(leiden_labels))
    logger.info("  Leiden: %d clusters at resolution=%.1f", n_leiden, best_resolution)

    # -----------------------------------------------------------------------
    # 4. Per-class metrics
    # -----------------------------------------------------------------------
    unique_classes = sorted(np.unique(transferred_labels))
    n_classes = len(unique_classes)

    # 4a. Basic per-class statistics
    class_stats = {}
    for cls in unique_classes:
        mask = transferred_labels == cls
        n_cls = int(np.sum(mask))
        proportion = n_cls / n_spots
        mean_conf = float(np.mean(confidences[mask]))
        class_stats[cls] = {
            "n_spots": n_cls,
            "proportion": proportion,
            "mean_confidence": mean_conf,
        }

    # 4b. Per-class spatial smoothness
    smoothness_per_class = _spatial_smoothness_per_class(coords, transferred_labels, k=10)
    for cls in unique_classes:
        class_stats[cls]["smoothness"] = smoothness_per_class.get(cls, 0.0)

    # 4c. Hungarian matching + per-class P/R/F1
    logger.info("Computing Hungarian matching between %d transferred types and %d Leiden clusters...",
                n_classes, n_leiden)
    hungarian_results = _compute_hungarian_prf(transferred_labels_leiden, leiden_labels)

    # Merge Hungarian P/R/F1 into class_stats
    for cls in unique_classes:
        if cls in hungarian_results["per_class"]:
            h = hungarian_results["per_class"][cls]
            class_stats[cls]["precision"] = h["precision"]
            class_stats[cls]["recall"] = h["recall"]
            class_stats[cls]["f1"] = h["f1"]
        else:
            class_stats[cls]["precision"] = 0.0
            class_stats[cls]["recall"] = 0.0
            class_stats[cls]["f1"] = 0.0

    # -----------------------------------------------------------------------
    # 5. Format output tables
    # -----------------------------------------------------------------------
    lines.append(f"**Leiden resolution**: {best_resolution} ({n_leiden} clusters)")
    lines.append(f"**Transferred classes**: {n_classes}")
    lines.append("")

    # Main per-class table
    lines.append("### Per-Cell-Type Metrics")
    lines.append("")
    header = "| Cell Type | N Spots | Proportion | Mean Confidence | Smoothness | Precision | Recall | F1 |"
    sep =    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines.append(header)
    lines.append(sep)

    for cls in unique_classes:
        s = class_stats[cls]
        lines.append(
            f"| {cls} "
            f"| {s['n_spots']} "
            f"| {s['proportion']:.4f} "
            f"| {s['mean_confidence']:.4f} "
            f"| {s['smoothness']:.4f} "
            f"| {s['precision']:.4f} "
            f"| {s['recall']:.4f} "
            f"| {s['f1']:.4f} |"
        )

    lines.append("")

    # Summary row
    overall_smoothness = float(np.mean(list(smoothness_per_class.values())))
    overall_conf = float(np.mean(confidences))
    lines.append("### Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"| --- | ---: |")
    lines.append(f"| Total spots | {n_spots} |")
    lines.append(f"| Transferred classes | {n_classes} |")
    lines.append(f"| Leiden clusters | {n_leiden} |")
    lines.append(f"| Overall mean confidence | {overall_conf:.4f} |")
    lines.append(f"| Overall mean smoothness (macro) | {overall_smoothness:.4f} |")
    lines.append(f"| Macro Precision | {hungarian_results['macro_precision']:.4f} |")
    lines.append(f"| Macro Recall | {hungarian_results['macro_recall']:.4f} |")
    lines.append(f"| Macro F1 | {hungarian_results['macro_f1']:.4f} |")
    lines.append("")

    # Hungarian matching table
    lines.append("### Hungarian Matching (Leiden -> Transferred)")
    lines.append("")
    lines.append("| Leiden Cluster | Matched Cell Type |")
    lines.append("| ---: | --- |")
    if "leiden_to_transferred" in hungarian_results:
        for leid_cls in sorted(hungarian_results["leiden_to_transferred"].keys(), key=lambda x: int(x) if x.isdigit() else x):
            matched = hungarian_results["leiden_to_transferred"][leid_cls]
            lines.append(f"| {leid_cls} | {matched} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    print("# ROSETTA Per-Cell-Type Alignment Metrics")
    print("")
    print(f"Project root: `{PROJECT_ROOT}`")
    print("")

    for pair_config in TISSUE_PAIRS:
        try:
            report = analyze_tissue_pair(pair_config)
            print(report)
        except Exception as e:
            logger.error("Failed to analyze %s: %s", pair_config["name"], e)
            import traceback
            traceback.print_exc()
            print(f"## {pair_config['name']}")
            print(f"**ERROR**: {e}")
            print("")

    print("---")
    print("*Generated by `scripts/per_celltype_metrics.py`*")


if __name__ == "__main__":
    main()
