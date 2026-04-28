#!/usr/bin/env python
"""Evaluate the PCA encoder ablation checkpoint.

Loads PCA ablation checkpoint, extracts projected embeddings, and computes
alignment metrics (kNN label transfer ARI) for comparison with CHISEL-based BRIDGE.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bridge import DATASET_CONFIGS
from train_bridge_pca_ablation import PCAEncoder

from rosetta.bridge.training import BRIDGELitModule
from rosetta.chisel.graph_construction import build_spatial_graph
from rosetta.utils.config import BRIDGEConfig, TrainingConfig
from rosetta.utils.metrics import (
    alignment_score,
    label_transfer_accuracy,
    morans_i,
    spatial_label_smoothness,
)

import anndata as ad
import scanpy as sc
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

from rosetta.utils.config import PreprocessingConfig

KNOWN_LABEL_COLUMNS = [
    "class", "subclass", "neurotransmitter", "supertype", "cluster",
    "Cell_Type", "cell_type", "celltype",
]


def _preprocess_adata(h5ad_path, config):
    adata = ad.read_h5ad(h5ad_path)
    adata = adata.copy()
    adata.layers["counts"] = adata.X.copy()
    pp = config.get("preprocessing", PreprocessingConfig())
    sc.pp.filter_cells(adata, min_genes=pp.min_genes)
    sc.pp.filter_genes(adata, min_cells=pp.min_cells)
    if not config.get("skip_normalize", False):
        sc.pp.normalize_total(adata, target_sum=pp.target_sum)
        sc.pp.log1p(adata)
    if not config.get("skip_hvg", False) and adata.n_vars > pp.n_top_genes:
        try:
            adata_hvg = adata.copy()
            adata_hvg.X = adata_hvg.layers["counts"].copy()
            sc.pp.highly_variable_genes(adata_hvg, n_top_genes=pp.n_top_genes, flavor="seurat_v3")
            adata.var["highly_variable"] = adata_hvg.var["highly_variable"]
            adata = adata[:, adata.var["highly_variable"]].copy()
        except Exception:
            X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
            var = np.var(X, axis=0)
            top_idx = np.argsort(var)[-pp.n_top_genes:]
            adata = adata[:, top_idx].copy()
    return adata


def _adata_to_graph(adata, config):
    spatial_coords = torch.tensor(np.array(adata.obsm["spatial"][:, :2], dtype=np.float32))
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    expression = torch.tensor(np.array(X, dtype=np.float32))
    return build_spatial_graph(
        spatial_coords=spatial_coords,
        expression=expression,
        k_spatial=config.get("graph_k_spatial", 6),
        k_expression=config.get("graph_k_expression", 3),
        expression_threshold=0.3,
    )


@torch.no_grad()
def _extract_pca_embeddings(encoder, data, device):
    """Extract embeddings via PCAEncoder (just linear projection, no graph)."""
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    out = encoder(x, edge_index, edge_attr)
    return out["z_spot"]


def main():
    source_name = "visium_human_brain"
    target_name = "merfish_mouse_brain"
    ckpt_path = OUTPUTS_DIR / f"{source_name}_{target_name}" / "ablations" / "pca_encoder" / "checkpoints" / "last.ckpt"
    device = torch.device("cpu")

    source_config = DATASET_CONFIGS[source_name]
    target_config = DATASET_CONFIGS[target_name]

    # Load checkpoint
    logger.info("Loading PCA ablation checkpoint: %s", ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    input_dim_source = hparams["input_dim_source"]
    input_dim_target = hparams["input_dim_target"]
    state = ckpt["state_dict"]

    # Reconstruct model with PCAEncoders
    encoder_source = PCAEncoder(input_dim_source)
    encoder_target = PCAEncoder(input_dim_target)

    proj_hidden = state["projection_head.net.0.weight"].shape[0] if "projection_head.net.0.weight" in state else 0
    bridge_config = BRIDGEConfig(projection_hidden_dim=proj_hidden)
    training_config = TrainingConfig()

    model = BRIDGELitModule(
        encoder_source=encoder_source,
        encoder_target=encoder_target,
        bridge_config=bridge_config,
        training_config=training_config,
        input_dim_source=input_dim_source,
        input_dim_target=input_dim_target,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    del ckpt
    logger.info("Model loaded")

    # Load data
    logger.info("Loading datasets...")
    adata_source = _preprocess_adata(PROCESSED_DIR / f"{source_name}.h5ad", source_config)
    adata_target = _preprocess_adata(PROCESSED_DIR / f"{target_name}.h5ad", target_config)
    logger.info("Source: %d cells, %d genes", adata_source.n_obs, adata_source.n_vars)
    logger.info("Target: %d cells, %d genes", adata_target.n_obs, adata_target.n_vars)

    data_source = _adata_to_graph(adata_source, source_config)
    data_target = _adata_to_graph(adata_target, target_config)

    # Extract embeddings
    logger.info("Extracting embeddings...")
    with torch.no_grad():
        z_source_raw = _extract_pca_embeddings(model.encoder_source, data_source, device)
        z_target_raw = _extract_pca_embeddings(model.encoder_target, data_target, device)

        # Joint projection + L2 normalization
        z_both = model.projection_head(torch.cat([z_source_raw, z_target_raw], dim=0))
        z_both = F.normalize(z_both, dim=1)
        n_s = z_source_raw.shape[0]
        z_source_proj = z_both[:n_s].cpu().numpy()
        z_target_proj = z_both[n_s:].cpu().numpy()
    logger.info("Projected embeddings: source %s, target %s", z_source_proj.shape, z_target_proj.shape)

    # Find label columns in target
    target_labels = {}
    for col in KNOWN_LABEL_COLUMNS:
        if col in adata_target.obs.columns:
            labels = adata_target.obs[col].astype(str).values
            n_classes = len(set(labels))
            if 2 <= n_classes <= 200:
                target_labels[col] = labels

    logger.info("Target labels: %s", {k: len(set(v)) for k, v in target_labels.items()})

    # kNN label transfer (target → source, k=10)
    from sklearn.neighbors import KDTree

    k = 10
    results = {}

    # Subsample for kNN if needed
    max_cells = 30000
    n_total = z_source_proj.shape[0] + z_target_proj.shape[0]
    if n_total > max_cells:
        rng = np.random.default_rng(42)
        frac = max_cells / n_total
        n_s_sub = max(100, int(z_source_proj.shape[0] * frac))
        n_t_sub = max(100, int(z_target_proj.shape[0] * frac))
        idx_s = rng.choice(z_source_proj.shape[0], n_s_sub, replace=False)
        idx_t = rng.choice(z_target_proj.shape[0], n_t_sub, replace=False)
    else:
        idx_s = np.arange(z_source_proj.shape[0])
        idx_t = np.arange(z_target_proj.shape[0])

    z_s_sub = z_source_proj[idx_s]
    z_t_sub = z_target_proj[idx_t]

    # Alignment score
    z_combined = np.concatenate([z_s_sub, z_t_sub], axis=0)
    species_labels = np.array(["source"] * len(z_s_sub) + ["target"] * len(z_t_sub))
    try:
        align_score = alignment_score(z_combined, species_labels, k=10)
        results["alignment_score"] = float(align_score)
        logger.info("Alignment score: %.4f", align_score)
    except Exception as e:
        logger.warning("Alignment score failed: %s", e)

    # Label transfer: target (mouse, labeled) -> source (human, unlabeled)
    # Then validate with Leiden clustering on source expression (same as main eval)
    from rosetta.utils.metrics import label_transfer_with_confidence

    coords_source = np.array(adata_source.obsm["spatial"][:, :2], dtype=np.float32)

    for col, labels_full in target_labels.items():
        logger.info("  --- %s (%d classes) ---", col, len(set(labels_full)))

        # Transfer labels via kNN (full data)
        pred_labels, confidences = label_transfer_with_confidence(
            z_target_proj, z_source_proj, labels_full, k=10,
        )

        results[f"label_transfer_{col}_mean_confidence"] = float(np.mean(confidences))
        logger.info("    Mean confidence: %.3f", np.mean(confidences))

        # Spatial label smoothness
        smoothness = spatial_label_smoothness(coords_source, pred_labels, k=10)
        n_classes = len(set(labels_full))
        random_baseline = 1.0 / n_classes
        results[f"label_transfer_{col}_spatial_smoothness"] = float(smoothness)
        results[f"label_transfer_{col}_smoothness_vs_random"] = float(smoothness / random_baseline)
        logger.info("    Spatial smoothness: %.4f (%.1fx random)", smoothness, smoothness / random_baseline)

        # Save transferred labels
        eval_dir = OUTPUTS_DIR / f"{source_name}_{target_name}" / "ablations" / "pca_encoder" / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        np.savez(eval_dir / f"transferred_labels_{col}.npz", labels=pred_labels, confidences=confidences)

    # Pseudo-label validation: ARI of transferred labels vs Leiden on source expression
    logger.info("Computing pseudo-label validation (Leiden on source expression)...")
    adata_cluster = adata_source.copy()
    sc.pp.pca(adata_cluster, n_comps=min(50, adata_cluster.n_vars - 1))
    sc.pp.neighbors(adata_cluster, n_neighbors=15)

    for col in target_labels:
        eval_dir = OUTPUTS_DIR / f"{source_name}_{target_name}" / "ablations" / "pca_encoder" / "evaluation"
        npz_path = eval_dir / f"transferred_labels_{col}.npz"
        if not npz_path.exists():
            continue
        pred_labels = np.load(npz_path, allow_pickle=True)["labels"]

        best_ari = -1.0
        best_res = None
        for resolution in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
            sc.tl.leiden(adata_cluster, resolution=resolution, key_added=f"leiden_{resolution}")
            leiden_labels = np.array(adata_cluster.obs[f"leiden_{resolution}"].values, dtype=str)
            ari = adjusted_rand_score(pred_labels, leiden_labels)
            nmi = normalized_mutual_info_score(pred_labels, leiden_labels)
            results[f"pseudo_validation_{col}_leiden{resolution}_ari"] = float(ari)
            if ari > best_ari:
                best_ari = ari
                best_res = resolution
            logger.info("    %s vs Leiden(res=%.1f): ARI=%.4f, NMI=%.4f", col, resolution, ari, nmi)

        results[f"pseudo_validation_{col}_best_ari"] = float(best_ari)
        results[f"pseudo_validation_{col}_best_resolution"] = float(best_res)
        logger.info("    %s BEST: ARI=%.4f at resolution=%.1f", col, best_ari, best_res)

    # Save results
    eval_dir = OUTPUTS_DIR / f"{source_name}_{target_name}" / "ablations" / "pca_encoder" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.save(eval_dir / "z_source_proj.npy", z_source_proj)
    np.save(eval_dir / "z_target_proj.npy", z_target_proj)

    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", eval_dir)
    print("\n" + "=" * 60)
    print("PCA Ablation Evaluation Results")
    print("=" * 60)
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
