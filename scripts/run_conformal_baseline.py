#!/usr/bin/env python
"""Conformal prediction baseline: naive kNN confidence thresholding.

Compares ATLAS conformal prediction (calibrated coverage guarantees) against
naive kNN confidence thresholding (uncalibrated). Shows that naive approach
does NOT achieve guaranteed coverage while conformal does.

Usage:
    python scripts/run_conformal_baseline.py \
        --source visium_human_brain --target merfish_mouse_brain
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bridge import DATASET_CONFIGS
from evaluate_bridge import _preprocess_adata

from rosetta.atlas.conformal import conformal_label_transfer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

KNOWN_LABEL_COLUMNS = [
    "class", "subclass", "neurotransmitter", "supertype", "cluster",
    "Cell_Type", "cell_type", "celltype",
]


def naive_knn_confidence(z_labeled, z_unlabeled, labels, k=20,
                         cal_fraction=0.2, random_state=42):
    """Naive kNN confidence thresholding.

    Simply uses max kNN vote fraction as confidence. Rejects predictions
    below a threshold tuned on a held-out set to match ~90% coverage.
    No formal coverage guarantee.
    """
    classes = np.unique(labels)
    rng = np.random.default_rng(random_state)

    # Same split as conformal for fair comparison
    train_idx, cal_idx = [], []
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

    # Build KDTree
    tree = cKDTree(z_train)

    def get_predictions(z_query):
        dists, indices = tree.query(z_query, k=k)
        weights = 1.0 / (dists + 1e-8)

        class_to_idx = {c: i for i, c in enumerate(classes)}
        probs = np.zeros((len(z_query), len(classes)))
        for i in range(len(z_query)):
            for j in range(k):
                c_idx = class_to_idx.get(labels_train[indices[i, j]])
                if c_idx is not None:
                    probs[i, c_idx] += weights[i, j]
            probs[i] /= probs[i].sum() + 1e-12

        pred_labels = np.array([classes[np.argmax(probs[i])] for i in range(len(z_query))])
        max_conf = np.array([probs[i].max() for i in range(len(z_query))])
        return pred_labels, max_conf, probs

    # Calibration predictions
    cal_preds, cal_conf, _ = get_predictions(z_cal)

    # Find threshold for ~90% coverage on calibration set
    # (reject if confidence < threshold → must cover 90%)
    cal_correct = (cal_preds == labels_cal)

    # Try different thresholds
    thresholds = np.linspace(0, 1, 200)
    best_thresh = 0.0
    for t in thresholds:
        mask = cal_conf >= t
        if mask.sum() > 0:
            coverage = cal_correct[mask].mean()
            acceptance = mask.mean()
            # Want coverage ~0.9 with high acceptance
            if coverage >= 0.9:
                best_thresh = t

    # Test predictions
    test_preds, test_conf, test_probs = get_predictions(z_unlabeled)

    # Compute coverage at various thresholds
    coverage_curve = {}
    for target_cov in [0.8, 0.85, 0.9, 0.95]:
        # Find threshold that gives this coverage on cal set
        thresh = 0.0
        for t in thresholds:
            mask = cal_conf >= t
            if mask.sum() > 0 and cal_correct[mask].mean() >= target_cov:
                thresh = t
        mask_test = test_conf >= thresh
        acceptance_rate = float(mask_test.mean())
        coverage_curve[f"target_{target_cov}"] = {
            "threshold": float(thresh),
            "acceptance_rate": acceptance_rate,
        }

    # Key metric: at threshold giving 90% cal coverage, what's acceptance rate?
    mask_90 = test_conf >= best_thresh
    return {
        "method": "naive_knn_threshold",
        "k": k,
        "threshold_90pct": float(best_thresh),
        "cal_n": len(cal_idx),
        "train_n": len(train_idx),
        "test_acceptance_rate": float(mask_90.mean()),
        "test_mean_confidence": float(test_conf.mean()),
        "cal_accuracy": float(cal_correct.mean()),
        "cal_accuracy_at_threshold": float(
            cal_correct[cal_conf >= best_thresh].mean() if (cal_conf >= best_thresh).sum() > 0 else 0
        ),
        "coverage_curve": coverage_curve,
        "predicted_labels": test_preds,
        "confidences": test_conf,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    import anndata as ad

    pair_name = f"{args.source}_{args.target}"
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / pair_name
    eval_dir = output_dir / "evaluation"

    # Load projected embeddings
    z_src = np.load(eval_dir / "z_source_proj.npy")
    z_tgt = np.load(eval_dir / "z_target_proj.npy")
    logger.info("Loaded embeddings: source %s, target %s", z_src.shape, z_tgt.shape)

    # Load labels
    source_config = DATASET_CONFIGS[args.source]
    target_config = DATASET_CONFIGS[args.target]
    adata_src = _preprocess_adata(PROCESSED_DIR / f"{args.source}.h5ad", source_config)
    adata_tgt = _preprocess_adata(PROCESSED_DIR / f"{args.target}.h5ad", target_config)

    target_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_tgt.obs.columns]
    source_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_src.obs.columns]

    if target_label_cols:
        label_cols = target_label_cols
        adata_labeled = adata_tgt
        z_labeled, z_unlabeled = z_tgt, z_src
    elif source_label_cols:
        label_cols = source_label_cols
        adata_labeled = adata_src
        z_labeled, z_unlabeled = z_src, z_tgt
    else:
        print("No annotation columns found.")
        sys.exit(1)

    conformal_dir = eval_dir / "conformal"
    conformal_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for col in label_cols:
        labels = np.array(adata_labeled.obs[col].values, dtype=str)
        n_classes = len(np.unique(labels))
        if n_classes < 2 or n_classes > 500:
            continue

        logger.info("=" * 60)
        logger.info("Label: %s (%d classes)", col, n_classes)

        # Run conformal (our method)
        logger.info("Running conformal prediction...")
        conformal_result = conformal_label_transfer(
            z_labeled, z_unlabeled, labels,
            alpha=args.alpha, k=args.k,
        )

        # Run naive kNN threshold (baseline)
        logger.info("Running naive kNN thresholding...")
        naive_result = naive_knn_confidence(
            z_labeled, z_unlabeled, labels,
            k=args.k, cal_fraction=0.2,
        )

        all_results[col] = {
            "n_classes": n_classes,
            "conformal": {
                "coverage_guarantee": conformal_result["coverage_guarantee"],
                "calibration_coverage": conformal_result["calibration_coverage"],
                "mean_set_size": conformal_result["mean_set_size"],
                "singleton_fraction": conformal_result["singleton_fraction"],
            },
            "naive": {
                "threshold_90pct": naive_result["threshold_90pct"],
                "cal_accuracy": naive_result["cal_accuracy"],
                "cal_accuracy_at_threshold": naive_result["cal_accuracy_at_threshold"],
                "test_acceptance_rate": naive_result["test_acceptance_rate"],
                "test_mean_confidence": naive_result["test_mean_confidence"],
                "coverage_curve": naive_result["coverage_curve"],
            },
        }

        logger.info("  Conformal: coverage=%.3f, mean_set=%.2f, singletons=%.3f",
                     conformal_result["calibration_coverage"],
                     conformal_result["mean_set_size"],
                     conformal_result["singleton_fraction"])
        logger.info("  Naive: cal_acc=%.3f, threshold=%.3f, acceptance=%.3f",
                     naive_result["cal_accuracy"],
                     naive_result["threshold_90pct"],
                     naive_result["test_acceptance_rate"])

    # Save results
    out_file = conformal_dir / "conformal_vs_naive.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Comparison plot
    if all_results:
        cols = list(all_results.keys())
        fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5))
        if len(cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, cols):
            r = all_results[col]
            methods = ["Conformal\n(ATLAS)", "Naive kNN\nThreshold"]
            coverages = [
                r["conformal"]["calibration_coverage"],
                r["naive"]["cal_accuracy_at_threshold"],
            ]
            colors = ["steelblue", "coral"]

            bars = ax.bar(methods, coverages, color=colors, width=0.5)
            ax.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="90% target")
            ax.set_ylabel("Coverage")
            ax.set_title(f"{col} ({r['n_classes']} classes)")
            ax.set_ylim(0, 1.05)
            ax.legend()

            # Add value labels
            for bar, val in zip(bars, coverages):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", fontsize=10)

        fig.suptitle("Conformal vs Naive Coverage Comparison", fontsize=14)
        fig.tight_layout()
        fig.savefig(conformal_dir / "conformal_vs_naive.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Results saved to %s", out_file)

    print(f"\n{'='*60}")
    print(f"Conformal vs Naive: {args.source} <-> {args.target}")
    print(f"{'='*60}")
    for col, r in all_results.items():
        print(f"\n  {col} ({r['n_classes']} classes):")
        print(f"    Conformal coverage: {r['conformal']['calibration_coverage']:.3f} (guaranteed ≥{r['conformal']['coverage_guarantee']:.0%})")
        print(f"    Conformal set size: {r['conformal']['mean_set_size']:.2f}")
        print(f"    Naive accuracy: {r['naive']['cal_accuracy']:.3f}")
        print(f"    Naive acceptance@90%%: {r['naive']['test_acceptance_rate']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
