#!/usr/bin/env python
"""Evaluate a trained CHISEL model on a dataset.

Usage:
    python scripts/evaluate_chisel.py --dataset visium_human_brain --checkpoint outputs/visium_human_brain/checkpoints/last.ckpt
    python scripts/evaluate_chisel.py --dataset allen_brain_ish --checkpoint outputs/allen_brain_ish/checkpoints/last.ckpt
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
import torch

from rosetta.chisel.losses import GeneMasker
from rosetta.chisel.training import CHISELLitModule, SpatialGraphDataModule
from rosetta.utils.config import (
    CHISELConfig,
    MultiScaleConfig,
    PreprocessingConfig,
    SpatialGNNConfig,
    TrainingConfig,
)
from rosetta.utils.metrics import morans_i, reconstruction_accuracy, silhouette_score_spatial
from rosetta.utils.visualization import (
    plot_spatial_embeddings,
    plot_training_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Same dataset configs as train_chisel.py
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
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CHISEL model")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dataset not in DATASET_CONFIGS:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    h5ad_path = PROCESSED_DIR / f"{args.dataset}.h5ad"
    if not h5ad_path.exists():
        print(f"Data file not found: {h5ad_path}")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.dataset / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset_config = DATASET_CONFIGS[args.dataset]
    training_config = TrainingConfig(max_nodes=args.max_nodes)

    data_module = SpatialGraphDataModule(
        h5ad_path=h5ad_path,
        dataset_config=dataset_config,
        training_config=training_config,
        max_nodes=args.max_nodes,
    )
    data_module.setup()

    # Reconstruct config matching training
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

    # Load model from checkpoint
    logger.info("Loading model from %s", ckpt_path)
    model = CHISELLitModule.load_from_checkpoint(
        ckpt_path,
        chisel_config=chisel_config,
        training_config=training_config,
        input_dim=data_module.input_dim,
        max_nodes=data_module.n_nodes,
    )
    model.eval()

    # Determine device and move model to CPU for evaluation (avoids device mismatch)
    model = model.cpu()

    # Get graph data
    graph_data = data_module._graph_data
    assert graph_data is not None

    results = {"dataset": args.dataset}

    with torch.no_grad():
        # Forward pass (no masking)
        out = model.encoder(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        z_spot = out["z_spot"].cpu().numpy()
        spatial_coords = graph_data.pos.cpu().numpy()

        # 1. Moran's I
        mi = morans_i(z_spot, spatial_coords, k=6)
        results["morans_i"] = mi
        logger.info("Moran's I: %.4f", mi)

        # 2. Moran's I with random embeddings (baseline)
        z_random = np.random.randn(*z_spot.shape).astype(np.float32)
        mi_random = morans_i(z_random, spatial_coords, k=6)
        results["morans_i_random"] = mi_random
        logger.info("Moran's I (random baseline): %.4f", mi_random)

        # 3. Reconstruction accuracy
        masker = GeneMasker(mask_ratio=0.2, mask_nonzero_only=True)
        original_x = graph_data.x
        masked_x, gene_mask = masker(original_x)

        out_masked = model.encoder(masked_x, graph_data.edge_index, graph_data.edge_attr)
        pred_expr = model.mask_decoder(out_masked["z_spot"])

        recon = reconstruction_accuracy(
            pred_expr.cpu().numpy(),
            original_x.cpu().numpy(),
            gene_mask.cpu().numpy(),
        )
        results["reconstruction"] = recon
        logger.info("Reconstruction — MSE: %.4f, Pearson r: %.4f, Cosine: %.4f",
                     recon["mse"], recon["pearson_r"], recon["cosine_sim"])

        # 4. Mean-imputation baseline MSE
        mean_expr = original_x.mean(dim=0).unsqueeze(0).expand_as(original_x)
        baseline_recon = reconstruction_accuracy(
            mean_expr.cpu().numpy(),
            original_x.cpu().numpy(),
            gene_mask.cpu().numpy(),
        )
        results["reconstruction_baseline"] = baseline_recon
        logger.info("Baseline (mean imputation) — MSE: %.4f", baseline_recon["mse"])

    # 5. Spatial embedding plot
    logger.info("Generating spatial embedding plot...")
    fig = plot_spatial_embeddings(spatial_coords, z_spot, title=args.dataset)
    fig.savefig(output_dir / "spatial_embeddings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6. Training curves (if CSV exists)
    csv_path = PROJECT_ROOT / "outputs" / args.dataset / "training_metrics.csv"
    if csv_path.exists():
        import csv as csv_mod

        metrics: dict[str, list[float]] = {}
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                for k, v in row.items():
                    if v and k != "epoch":
                        try:
                            metrics.setdefault(k, []).append(float(v))
                        except ValueError:
                            pass

        if metrics:
            fig = plot_training_curves(metrics, title=f"{args.dataset} Training")
            fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Training curves saved")

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {args.dataset}")
    print(f"{'='*60}")
    print(f"  Moran's I (trained):  {results['morans_i']:.4f}")
    print(f"  Moran's I (random):   {results['morans_i_random']:.4f}")
    print(f"  Reconstruction MSE:   {recon['mse']:.4f}")
    print(f"  Baseline MSE:         {baseline_recon['mse']:.4f}")
    print(f"  Pearson r:            {recon['pearson_r']:.4f}")
    print(f"  Cosine similarity:    {recon['cosine_sim']:.4f}")
    print(f"  Output directory:     {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
