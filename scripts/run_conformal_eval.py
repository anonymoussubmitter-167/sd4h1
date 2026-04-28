#!/usr/bin/env python
"""Run conformal prediction evaluation on BRIDGE-aligned embeddings.

Loads projected embeddings from a completed evaluation run and applies
conformal prediction for uncertainty-calibrated label transfer.

Usage:
    python scripts/run_conformal_eval.py \
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

from rosetta.atlas.conformal import conformal_label_transfer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bridge import DATASET_CONFIGS
from evaluate_bridge import _preprocess_adata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

KNOWN_LABEL_COLUMNS = [
    "class", "subclass", "neurotransmitter", "supertype", "cluster",
    "Cell_Type", "cell_type", "celltype",
]


def main():
    parser = argparse.ArgumentParser(description="Conformal prediction evaluation")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Significance level (0.1 = 90%% coverage)")
    parser.add_argument("--k", type=int, default=20, help="kNN neighbors")
    args = parser.parse_args()

    import anndata as ad

    pair_name = f"{args.source}_{args.target}"
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / pair_name
    eval_dir = output_dir / "evaluation"

    # Load projected embeddings (from evaluate_bridge.py)
    z_src_path = eval_dir / "z_source_proj.npy"
    z_tgt_path = eval_dir / "z_target_proj.npy"
    if not z_src_path.exists() or not z_tgt_path.exists():
        print(f"Projected embeddings not found in {eval_dir}")
        print("Run scripts/evaluate_bridge.py first.")
        sys.exit(1)

    z_source = np.load(z_src_path)
    z_target = np.load(z_tgt_path)
    logger.info("Loaded embeddings: source %s, target %s", z_source.shape, z_target.shape)

    # Load and preprocess AnnData (same pipeline as evaluate_bridge to match embeddings)
    source_h5ad = PROCESSED_DIR / f"{args.source}.h5ad"
    target_h5ad = PROCESSED_DIR / f"{args.target}.h5ad"
    source_config = DATASET_CONFIGS[args.source]
    target_config = DATASET_CONFIGS[args.target]
    adata_source = _preprocess_adata(source_h5ad, source_config)
    adata_target = _preprocess_adata(target_h5ad, target_config)

    # Detect which side has labels
    target_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_target.obs.columns]
    source_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_source.obs.columns]

    if target_label_cols:
        label_cols = target_label_cols
        adata_labeled = adata_target
        z_labeled, z_unlabeled = z_target, z_source
        coords_unlabeled = np.array(adata_source.obsm["spatial"][:, :2], dtype=np.float32)
        unlabeled_name = args.source
    elif source_label_cols:
        label_cols = source_label_cols
        adata_labeled = adata_source
        z_labeled, z_unlabeled = z_source, z_target
        coords_unlabeled = np.array(adata_target.obsm["spatial"][:, :2], dtype=np.float32)
        unlabeled_name = args.target
    else:
        print("No annotation columns found in either dataset.")
        sys.exit(1)

    conformal_dir = eval_dir / "conformal"
    conformal_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for col in label_cols:
        labels = np.array(adata_labeled.obs[col].values, dtype=str)
        logger.info("Conformal prediction for '%s' (%d classes)", col, len(np.unique(labels)))

        result = conformal_label_transfer(
            z_labeled, z_unlabeled, labels,
            alpha=args.alpha, k=args.k,
        )

        all_results[col] = {
            "coverage_guarantee": result["coverage_guarantee"],
            "calibration_coverage": result["calibration_coverage"],
            "mean_set_size": result["mean_set_size"],
            "singleton_fraction": result["singleton_fraction"],
            "calibration_quantile": result["calibration_quantile"],
            "n_classes": len(result["classes"]),
        }

        logger.info("  Coverage guarantee: %.1f%%", result["coverage_guarantee"] * 100)
        logger.info("  Calibration coverage: %.3f", result["calibration_coverage"])
        logger.info("  Mean set size: %.2f", result["mean_set_size"])
        logger.info("  Singleton fraction: %.3f", result["singleton_fraction"])

        # Save prediction sets
        np.savez(
            conformal_dir / f"conformal_{col}.npz",
            predicted_labels=result["predicted_labels"],
            set_sizes=result["set_sizes"],
            probabilities=result["probabilities"],
            classes=result["classes"],
        )

        # Spatial uncertainty map (color by prediction set size)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sc_plot = ax.scatter(
            coords_unlabeled[:, 0], coords_unlabeled[:, 1],
            c=result["set_sizes"], cmap="YlOrRd", s=8, alpha=0.8,
            vmin=1, vmax=max(5, np.percentile(result["set_sizes"], 95)),
            rasterized=True,
        )
        fig.colorbar(sc_plot, ax=ax, label="Prediction set size", shrink=0.7)
        ax.set_title(f"{unlabeled_name} - Conformal uncertainty ({col}, "
                     f"alpha={args.alpha})", fontsize=13)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_aspect("equal")
        fig.savefig(conformal_dir / f"spatial_uncertainty_{col}.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved spatial_uncertainty_%s.png", col)

    # Save summary
    with open(conformal_dir / "conformal_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Conformal Prediction: {args.source} <-> {args.target}")
    print(f"Alpha: {args.alpha} (coverage guarantee: {1-args.alpha:.0%})")
    print(f"{'='*60}")
    for col, r in all_results.items():
        print(f"\n  {col} ({r['n_classes']} classes):")
        print(f"    Calibration coverage: {r['calibration_coverage']:.3f}")
        print(f"    Mean set size: {r['mean_set_size']:.2f}")
        print(f"    Singleton fraction: {r['singleton_fraction']:.3f}")
    print(f"\nOutputs: {conformal_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
