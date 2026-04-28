#!/usr/bin/env python
"""Train BRIDGE cross-species alignment between two datasets.

Usage:
    python scripts/train_bridge.py --source visium_human_brain --target merfish_mouse_brain
    python scripts/train_bridge.py --source merfish_human_liver --target visium_mouse_liver
    python scripts/train_bridge.py --source visium_human_intestine --target visium_mouse_intestine
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

import numpy as np
import pytorch_lightning as pl
import torch

from rosetta.bridge.training import BRIDGELitModule, CrossSpeciesDataModule
from rosetta.chisel.encoders import CHISELEncoder
from rosetta.chisel.training import CHISELLitModule
from rosetta.data.ortholog_db import (
    build_ortholog_mapping_from_gene_names,
    load_ortholog_mapping,
)
from rosetta.utils.config import (
    BRIDGEConfig,
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
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Per-dataset configs (mirroring train_chisel.py)
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


def _find_checkpoint(dataset: str) -> Path | None:
    """Find the best or last CHISEL checkpoint for a dataset."""
    ckpt_dir = OUTPUTS_DIR / dataset / "checkpoints"
    if not ckpt_dir.exists():
        return None

    # Prefer 'last.ckpt'
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last

    # Otherwise, find the best checkpoint by name
    ckpts = sorted(ckpt_dir.glob("chisel-*.ckpt"))
    return ckpts[-1] if ckpts else None


def _load_encoder_from_checkpoint(
    ckpt_path: Path,
    input_dim: int,
    max_nodes: int,
    hidden_dim: int = 64,
    embed_dim: int = 64,
) -> CHISELEncoder:
    """Load a CHISEL encoder from a Lightning checkpoint.

    Reads max_nodes from checkpoint hyperparameters to ensure DiffPool
    assignment layer dimensions match the saved weights.
    """
    # Read max_nodes from checkpoint (DiffPool dimensions depend on it)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt_hparams = ckpt.get("hyper_parameters", {})
    ckpt_max_nodes = ckpt_hparams.get("max_nodes", max_nodes)
    ckpt_input_dim = ckpt_hparams.get("input_dim", input_dim)
    del ckpt  # free memory

    chisel_config = CHISELConfig(
        spatial_gnn=SpatialGNNConfig(
            input_dim=ckpt_input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        ),
        multi_scale=MultiScaleConfig(
            pool_ratios=[0.25, 0.25],
            embed_dim=embed_dim,
            num_gnn_layers_per_level=2,
        ),
    )
    training_config = TrainingConfig()

    lit_module = CHISELLitModule.load_from_checkpoint(
        str(ckpt_path),
        chisel_config=chisel_config,
        training_config=training_config,
        input_dim=ckpt_input_dim,
        max_nodes=ckpt_max_nodes,
    )
    return lit_module.encoder


def main():
    parser = argparse.ArgumentParser(description="Train BRIDGE cross-species alignment")
    parser.add_argument("--source", type=str, required=True,
                        help="Source dataset name")
    parser.add_argument("--target", type=str, required=True,
                        help="Target dataset name")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="FGW alpha (0=feature, 1=structure)")
    parser.add_argument("--n-iters", type=int, default=5,
                        help="Number of alternating iterations")
    parser.add_argument("--finetune-epochs", type=int, default=10,
                        help="Fine-tune epochs per iteration")
    parser.add_argument("--max-nodes", type=int, default=3000,
                        help="Max nodes per species")
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.005,
                        help="Entropic regularization (lower = sharper plans)")
    parser.add_argument("--projection-hidden-dim", type=int, default=128,
                        help="Hidden dim for shared projection head")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=10,
                        help="Epochs to keep encoders frozen (projection head trains alone)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override total epochs (default: n_iters * finetune_epochs)")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--lambda-mmd", type=float, default=10.0,
                        help="MMD distribution matching loss weight (0 to disable)")
    parser.add_argument("--lambda-cross-contrastive", type=float, default=0.5,
                        help="Cross-species contrastive loss weight (0 to disable)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    for name in [args.source, args.target]:
        if name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {name}")
            print(f"Available: {sorted(DATASET_CONFIGS.keys())}")
            sys.exit(1)

    source_config = DATASET_CONFIGS[args.source]
    target_config = DATASET_CONFIGS[args.target]
    source_species = source_config["species"]
    target_species = target_config["species"]

    # Output directory
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else OUTPUTS_DIR / f"{args.source}_{args.target}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check data files exist
    source_h5ad = PROCESSED_DIR / f"{args.source}.h5ad"
    target_h5ad = PROCESSED_DIR / f"{args.target}.h5ad"
    for p in [source_h5ad, target_h5ad]:
        if not p.exists():
            print(f"Data file not found: {p}")
            print("Run scripts/download_data.py first.")
            sys.exit(1)

    # Bridge config
    bridge_config = BRIDGEConfig(
        alpha=args.alpha,
        epsilon=args.epsilon,
        n_alternating_iters=args.n_iters,
        finetune_epochs_per_iter=args.finetune_epochs,
        max_nodes_per_species=args.max_nodes,
        finetune_lr=args.finetune_lr,
        projection_hidden_dim=args.projection_hidden_dim,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        lambda_mmd=args.lambda_mmd,
        lambda_cross_contrastive=args.lambda_cross_contrastive,
    )

    training_config = TrainingConfig()

    # Load ortholog mapping
    logger.info("Loading ortholog mapping: %s -> %s", source_species, target_species)
    ortholog_map = load_ortholog_mapping(source_species, target_species)
    if ortholog_map.n_orthologs == 0:
        logger.warning("No cached orthologs found; will use gene name matching fallback")

    # Data module
    data_module = CrossSpeciesDataModule(
        source_h5ad=source_h5ad,
        target_h5ad=target_h5ad,
        source_config=source_config,
        target_config=target_config,
        bridge_config=bridge_config,
        training_config=training_config,
        ortholog_map=ortholog_map,
    )
    data_module.setup()

    # Load pretrained encoders
    source_ckpt = _find_checkpoint(args.source)
    target_ckpt = _find_checkpoint(args.target)

    if source_ckpt and target_ckpt:
        logger.info("Loading source encoder from %s", source_ckpt)
        encoder_source = _load_encoder_from_checkpoint(
            source_ckpt, data_module.input_dim_source,
            data_module.n_nodes_source, args.hidden_dim, args.embed_dim,
        )
        logger.info("Loading target encoder from %s", target_ckpt)
        encoder_target = _load_encoder_from_checkpoint(
            target_ckpt, data_module.input_dim_target,
            data_module.n_nodes_target, args.hidden_dim, args.embed_dim,
        )
    else:
        logger.warning("Pretrained checkpoints not found, initializing random encoders")
        chisel_cfg_s = CHISELConfig(
            spatial_gnn=SpatialGNNConfig(
                input_dim=data_module.input_dim_source,
                hidden_dim=args.hidden_dim, num_layers=2, num_heads=4,
            ),
            multi_scale=MultiScaleConfig(
                pool_ratios=[0.25, 0.25], embed_dim=args.embed_dim,
            ),
        )
        chisel_cfg_t = CHISELConfig(
            spatial_gnn=SpatialGNNConfig(
                input_dim=data_module.input_dim_target,
                hidden_dim=args.hidden_dim, num_layers=2, num_heads=4,
            ),
            multi_scale=MultiScaleConfig(
                pool_ratios=[0.25, 0.25], embed_dim=args.embed_dim,
            ),
        )
        encoder_source = CHISELEncoder(chisel_cfg_s, max_nodes=data_module.n_nodes_source)
        encoder_target = CHISELEncoder(chisel_cfg_t, max_nodes=data_module.n_nodes_target)

    # BRIDGE module
    model = BRIDGELitModule(
        encoder_source=encoder_source,
        encoder_target=encoder_target,
        bridge_config=bridge_config,
        training_config=training_config,
        input_dim_source=data_module.input_dim_source,
        input_dim_target=data_module.input_dim_target,
    )

    # Training: alternating iterations
    logger.info("=" * 60)
    logger.info("BRIDGE alignment: %s <-> %s", args.source, args.target)
    logger.info("  Source: %d nodes, %d genes", data_module.n_nodes_source, data_module.input_dim_source)
    logger.info("  Target: %d nodes, %d genes", data_module.n_nodes_target, data_module.input_dim_target)
    logger.info("  Alpha: %.2f, Epsilon: %.4f, Iters: %d, FT epochs/iter: %d",
                args.alpha, args.epsilon, args.n_iters, args.finetune_epochs)
    logger.info("  Projection hidden dim: %d, Freeze encoder epochs: %d",
                args.projection_hidden_dim, args.freeze_encoder_epochs)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="bridge-{epoch:03d}",
        save_last=True,
    )

    t0 = time.time()
    total_epochs = args.max_epochs if args.max_epochs else args.n_iters * args.finetune_epochs

    trainer = pl.Trainer(
        max_epochs=total_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        # No gradient_clip_val — BRIDGELitModule uses manual optimization
        # with explicit clip_grad_norm_ in training_step
        enable_progress_bar=True,
        log_every_n_steps=1,
        default_root_dir=str(output_dir),
    )

    trainer.fit(model, data_module, ckpt_path=args.resume_from)

    elapsed = time.time() - t0
    logger.info("BRIDGE training complete in %.1f seconds", elapsed)

    # Save transport plan
    if model.transport_plan is not None:
        T_np = model.transport_plan.cpu().numpy()
        np.save(output_dir / "transport_plan.npy", T_np)
        logger.info("Transport plan saved to %s", output_dir / "transport_plan.npy")

    # Print final metrics
    final = trainer.callback_metrics
    print(f"\n{'='*50}")
    print(f"Final metrics for {args.source} <-> {args.target}:")
    for k, v in sorted(final.items()):
        val = v.item() if torch.is_tensor(v) else v
        print(f"  {k}: {val:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
