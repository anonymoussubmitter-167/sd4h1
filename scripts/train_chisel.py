#!/usr/bin/env python
"""Train CHISEL encoder with self-supervised objectives on a single dataset.

Usage:
    python scripts/train_chisel.py --dataset visium_human_brain --max-epochs 100
    python scripts/train_chisel.py --dataset merfish_mouse_brain --max-nodes 5000
    python scripts/train_chisel.py --dataset allen_brain_ish --max-epochs 50
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytorch_lightning as pl
import torch

from rosetta.chisel.training import CHISELLitModule, SpatialGraphDataModule
from rosetta.utils.config import (
    CHISELConfig,
    MultiScaleConfig,
    PreprocessingConfig,
    SpatialGNNConfig,
    TrainingConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Per-dataset configs (same as validate_real_data.py)
DATASET_CONFIGS = {
    "visium_human_brain": {
        "species": "human",
        "platform": "visium",
        "preprocessing": PreprocessingConfig(
            min_genes=200, min_cells=3, n_top_genes=2000, target_sum=1e4,
        ),
        "graph_k_spatial": 6,
        "graph_k_expression": 3,
        "skip_hvg": False,
        "skip_normalize": False,
    },
    "merfish_mouse_brain": {
        "species": "mouse",
        "platform": "merfish",
        "preprocessing": PreprocessingConfig(
            min_genes=5, min_cells=10, n_top_genes=500, target_sum=100.0,
        ),
        "graph_k_spatial": 10,
        "graph_k_expression": 5,
        "skip_hvg": True,
        "skip_normalize": False,
    },
    "stereoseq_zebrafish_brain": {
        "species": "zebrafish",
        "platform": "stereoseq",
        "preprocessing": PreprocessingConfig(
            min_genes=100, min_cells=5, n_top_genes=2000, target_sum=1e4,
        ),
        "graph_k_spatial": 8,
        "graph_k_expression": 3,
        "skip_hvg": False,
        "skip_normalize": False,
    },
    "allen_brain_ish": {
        "species": "mouse",
        "platform": "allen_ish",
        "preprocessing": PreprocessingConfig(
            min_genes=1, min_cells=1, n_top_genes=2000, target_sum=1e4,
        ),
        "graph_k_spatial": 6,
        "graph_k_expression": 3,
        "skip_hvg": True,
        "skip_normalize": True,
    },
    "visium_mouse_liver": {
        "species": "mouse",
        "platform": "visium",
        "preprocessing": PreprocessingConfig(
            min_genes=200, min_cells=3, n_top_genes=2000, target_sum=1e4,
        ),
        "graph_k_spatial": 6,
        "graph_k_expression": 3,
        "skip_hvg": False,
        "skip_normalize": False,
    },
    "merfish_human_liver": {
        "species": "human",
        "platform": "merfish",
        "preprocessing": PreprocessingConfig(
            min_genes=5, min_cells=10, n_top_genes=500, target_sum=100.0,
        ),
        "graph_k_spatial": 10,
        "graph_k_expression": 5,
        "skip_hvg": True,
        "skip_normalize": False,
    },
    "visium_mouse_intestine": {
        "species": "mouse",
        "platform": "visium",
        "preprocessing": PreprocessingConfig(
            min_genes=200, min_cells=3, n_top_genes=2000, target_sum=1e4,
        ),
        "graph_k_spatial": 6,
        "graph_k_expression": 3,
        "skip_hvg": False,
        "skip_normalize": False,
    },
    "visium_human_intestine": {
        "species": "human",
        "platform": "visium",
        "preprocessing": PreprocessingConfig(
            min_genes=200, min_cells=3, n_top_genes=2000, target_sum=1e4,
        ),
        "graph_k_spatial": 6,
        "graph_k_expression": 3,
        "skip_hvg": False,
        "skip_normalize": False,
    },
}


class MetricsLogger(pl.Callback):
    """Callback to collect per-epoch metrics for CSV logging."""

    def __init__(self):
        self.epoch_metrics: list[dict] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = {k: v.item() if torch.is_tensor(v) else v
                   for k, v in trainer.callback_metrics.items()}
        metrics["epoch"] = trainer.current_epoch
        self.epoch_metrics.append(metrics)


def main():
    parser = argparse.ArgumentParser(description="Train CHISEL encoder")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., visium_human_brain)")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--list", action="store_true",
                        help="List available datasets and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for name in sorted(DATASET_CONFIGS.keys()):
            h5ad = PROCESSED_DIR / f"{name}.h5ad"
            exists = "EXISTS" if h5ad.exists() else "MISSING"
            print(f"  {name:<30s} [{exists}]")
        return

    if args.dataset not in DATASET_CONFIGS:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {sorted(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    dataset_config = DATASET_CONFIGS[args.dataset]
    h5ad_path = PROCESSED_DIR / f"{args.dataset}.h5ad"

    if not h5ad_path.exists():
        print(f"Data file not found: {h5ad_path}")
        print("Run scripts/download_data.py first.")
        sys.exit(1)

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training config
    training_config = TrainingConfig(
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        mask_ratio=args.mask_ratio,
        max_nodes=args.max_nodes,
    )

    # Data module
    data_module = SpatialGraphDataModule(
        h5ad_path=h5ad_path,
        dataset_config=dataset_config,
        training_config=training_config,
        max_nodes=args.max_nodes,
    )
    data_module.setup()

    # CHISEL config
    chisel_config = CHISELConfig(
        spatial_gnn=SpatialGNNConfig(
            input_dim=data_module.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        ),
        multi_scale=MultiScaleConfig(
            pool_ratios=[0.25, 0.25],
            embed_dim=args.embed_dim,
            num_gnn_layers_per_level=2,
        ),
    )

    # Lightning module
    model = CHISELLitModule(
        chisel_config=chisel_config,
        training_config=training_config,
        input_dim=data_module.input_dim,
        max_nodes=data_module.n_nodes,
    )

    # Callbacks
    metrics_logger = MetricsLogger()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="chisel-{epoch:03d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    # Trainer
    logger.info("=" * 60)
    logger.info("Training CHISEL on %s", args.dataset)
    logger.info("  Nodes: %d, Genes: %d", data_module.n_nodes, data_module.input_dim)
    logger.info("  Hidden dim: %d, Embed dim: %d", args.hidden_dim, args.embed_dim)
    logger.info("  LR: %g, Epochs: %d, Mask ratio: %.2f",
                args.lr, args.max_epochs, args.mask_ratio)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    t0 = time.time()

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[metrics_logger, checkpoint_callback],
        gradient_clip_val=training_config.gradient_clip_val,
        enable_progress_bar=True,
        log_every_n_steps=1,
        default_root_dir=str(output_dir),
    )

    trainer.fit(model, data_module)

    elapsed = time.time() - t0
    logger.info("Training complete in %.1f seconds", elapsed)

    # Save metrics to CSV
    csv_path = output_dir / "training_metrics.csv"
    if metrics_logger.epoch_metrics:
        all_keys = set()
        for m in metrics_logger.epoch_metrics:
            all_keys.update(m.keys())
        all_keys = sorted(all_keys)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for m in metrics_logger.epoch_metrics:
                writer.writerow({k: m.get(k, "") for k in all_keys})
        logger.info("Metrics saved to %s", csv_path)

    # Print final metrics
    final = trainer.callback_metrics
    print(f"\n{'='*50}")
    print(f"Final metrics for {args.dataset}:")
    for k, v in sorted(final.items()):
        val = v.item() if torch.is_tensor(v) else v
        print(f"  {k}: {val:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
