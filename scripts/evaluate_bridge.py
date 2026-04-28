#!/usr/bin/env python
"""Evaluate BRIDGE cross-species alignment with label transfer.

Computes alignment metrics, transfers mouse cell type labels to human via kNN,
evaluates spatial coherence, compares against baselines, and generates plots.

Usage:
    python scripts/evaluate_bridge.py \
        --source visium_human_brain --target merfish_mouse_brain

Output directory: outputs/<source>_<target>/evaluation/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from rosetta.bridge.shared_space import build_shared_gene_space, _get_gene_lookup
from rosetta.bridge.training import BRIDGELitModule, SharedProjectionHead
from rosetta.chisel.encoders import CHISELEncoder
from rosetta.chisel.graph_construction import build_spatial_graph
from rosetta.data.ortholog_db import load_ortholog_mapping
from rosetta.utils.config import (
    BRIDGEConfig,
    CHISELConfig,
    MultiScaleConfig,
    PreprocessingConfig,
    SpatialGNNConfig,
    TrainingConfig,
)
from rosetta.utils.metrics import (
    alignment_score,
    label_transfer_accuracy,
    label_transfer_with_confidence,
    morans_i,
    spatial_label_smoothness,
)

# Standard Leiden resolution sweep — MUST be identical for BRIDGE and all baselines
LEIDEN_RESOLUTIONS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

# Reuse dataset configs from training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bridge import DATASET_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Default annotation columns to look for (auto-detected per dataset)
KNOWN_LABEL_COLUMNS = [
    "class", "subclass", "neurotransmitter", "supertype", "cluster",
    "Cell_Type", "cell_type", "celltype",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_best_bridge_checkpoint(output_dir: Path) -> Path | None:
    """Find the best BRIDGE checkpoint (highest epoch number)."""
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    # Find all bridge-epoch=*.ckpt files
    ckpts = list(ckpt_dir.glob("bridge-epoch=*.ckpt"))
    if not ckpts:
        last = ckpt_dir / "last.ckpt"
        return last if last.exists() else None

    # Sort by epoch number (pick highest)
    def _epoch(p: Path) -> int:
        name = p.stem  # e.g. "bridge-epoch=149-v1"
        epoch_str = name.split("=")[1].split("-")[0]
        return int(epoch_str)

    ckpts.sort(key=_epoch)
    return ckpts[-1]


def _load_bridge_model(
    ckpt_path: Path,
    source_name: str,
    target_name: str,
    device: torch.device,
) -> BRIDGELitModule:
    """Reconstruct BRIDGELitModule and load checkpoint weights."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    input_dim_source = hparams["input_dim_source"]
    input_dim_target = hparams["input_dim_target"]

    # Infer max_nodes from DiffPool assignment layer dimensions
    state = ckpt["state_dict"]
    src_pool1_dim = state["encoder_source.multi_scale.pool1.assign_gnn.layers.0.bias"].shape[0]
    tgt_pool1_dim = state["encoder_target.multi_scale.pool1.assign_gnn.layers.0.bias"].shape[0]
    pool_ratio = 0.25
    max_nodes_source = int(round(src_pool1_dim / pool_ratio))
    max_nodes_target = int(round(tgt_pool1_dim / pool_ratio))

    hidden_dim = 64
    embed_dim = 64

    def _make_encoder(input_dim: int, max_nodes: int) -> CHISELEncoder:
        cfg = CHISELConfig(
            spatial_gnn=SpatialGNNConfig(
                input_dim=input_dim, hidden_dim=hidden_dim,
                num_layers=2, num_heads=4, dropout=0.1,
            ),
            multi_scale=MultiScaleConfig(
                pool_ratios=[pool_ratio, pool_ratio],
                embed_dim=embed_dim, num_gnn_layers_per_level=2,
            ),
        )
        return CHISELEncoder(cfg, max_nodes=max_nodes)

    encoder_source = _make_encoder(input_dim_source, max_nodes_source)
    encoder_target = _make_encoder(input_dim_target, max_nodes_target)

    # Detect projection_hidden_dim from checkpoint (0 = identity/no projection head)
    if "projection_head.net.0.weight" in state:
        proj_hidden = state["projection_head.net.0.weight"].shape[0]
    else:
        proj_hidden = 0  # Identity projection (no projection head ablation)

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
    return model


def _preprocess_adata(h5ad_path: Path, config: dict) -> ad.AnnData:
    """Load and preprocess h5ad (same pipeline as training, NO subsampling)."""
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
            sc.pp.highly_variable_genes(
                adata_hvg, n_top_genes=pp.n_top_genes, flavor="seurat_v3",
            )
            adata.var["highly_variable"] = adata_hvg.var["highly_variable"]
            adata = adata[:, adata.var["highly_variable"]].copy()
        except Exception as e:
            logger.warning("HVG failed (%s), using top by variance", e)
            X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
            var = np.var(X, axis=0)
            top_idx = np.argsort(var)[-pp.n_top_genes:]
            adata = adata[:, top_idx].copy()

    return adata


def _adata_to_graph(adata: ad.AnnData, config: dict):
    """Build PyG Data from preprocessed AnnData (no subsampling)."""
    spatial_coords = torch.tensor(
        np.array(adata.obsm["spatial"][:, :2], dtype=np.float32)
    )
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
def _extract_spot_embeddings(
    encoder: CHISELEncoder,
    data,
    device: torch.device,
) -> torch.Tensor:
    """Extract spot embeddings via GNN-only path (bypasses DiffPool max_nodes)."""
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    z_sparse = encoder.gnn_encoder(x, edge_index, edge_attr)
    z_spot = encoder.proj_spot(z_sparse)
    return z_spot


@torch.no_grad()
def _project_embeddings(
    projection_head: SharedProjectionHead,
    z_source: torch.Tensor,
    z_target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint projection + L2 normalization (matches training forward pass)."""
    z_both = projection_head(torch.cat([z_source, z_target], dim=0))
    z_both = F.normalize(z_both, dim=1)
    n_s = z_source.shape[0]
    z_s_np = z_both[:n_s].cpu().numpy()
    z_t_np = z_both[n_s:].cpu().numpy()
    return z_s_np, z_t_np


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate BRIDGE alignment")
    parser.add_argument("--source", type=str, required=True,
                        help="Source dataset name (convention: human)")
    parser.add_argument("--target", type=str, required=True,
                        help="Target dataset name (convention: mouse)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific BRIDGE checkpoint (auto-detect if omitted)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for name in [args.source, args.target]:
        if name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {name}")
            print(f"Available: {sorted(DATASET_CONFIGS.keys())}")
            sys.exit(1)

    source_config = DATASET_CONFIGS[args.source]
    target_config = DATASET_CONFIGS[args.target]
    device = torch.device(args.device)

    pair_name = f"{args.source}_{args.target}"
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / pair_name
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # -----------------------------------------------------------------------
    # 1. Load BRIDGE checkpoint
    # -----------------------------------------------------------------------
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = _find_best_bridge_checkpoint(output_dir)

    if ckpt_path is None or not ckpt_path.exists():
        print(f"No BRIDGE checkpoint found in {output_dir}/checkpoints/")
        print("Run scripts/train_bridge.py first.")
        sys.exit(1)

    logger.info("Loading BRIDGE checkpoint: %s", ckpt_path)
    model = _load_bridge_model(ckpt_path, args.source, args.target, device)
    logger.info("Model loaded successfully")

    # -----------------------------------------------------------------------
    # 2. Load full data + build graphs
    # -----------------------------------------------------------------------
    logger.info("Loading and preprocessing datasets (full data, no subsampling)...")
    source_h5ad = PROCESSED_DIR / f"{args.source}.h5ad"
    target_h5ad = PROCESSED_DIR / f"{args.target}.h5ad"

    adata_source = _preprocess_adata(source_h5ad, source_config)
    adata_target = _preprocess_adata(target_h5ad, target_config)
    logger.info("Source (%s): %d cells, %d genes", args.source, adata_source.n_obs, adata_source.n_vars)
    logger.info("Target (%s): %d cells, %d genes", args.target, adata_target.n_obs, adata_target.n_vars)

    data_source = _adata_to_graph(adata_source, source_config)
    data_target = _adata_to_graph(adata_target, target_config)

    # -----------------------------------------------------------------------
    # 3. Extract projected embeddings (spot-only path, all cells)
    # -----------------------------------------------------------------------
    logger.info("Extracting spot embeddings (GNN-only path)...")
    z_source_raw = _extract_spot_embeddings(model.encoder_source, data_source, device)
    z_target_raw = _extract_spot_embeddings(model.encoder_target, data_target, device)
    logger.info("Source embeddings: %s, Target embeddings: %s",
                z_source_raw.shape, z_target_raw.shape)

    z_source_proj, z_target_proj = _project_embeddings(
        model.projection_head, z_source_raw, z_target_raw,
    )
    logger.info("Projected embeddings: source %s, target %s",
                z_source_proj.shape, z_target_proj.shape)

    # Save projected embeddings
    np.save(eval_dir / "z_source_proj.npy", z_source_proj)
    np.save(eval_dir / "z_target_proj.npy", z_target_proj)

    # Spatial coordinates
    coords_source = np.array(adata_source.obsm["spatial"][:, :2], dtype=np.float32)
    coords_target = np.array(adata_target.obsm["spatial"][:, :2], dtype=np.float32)

    # -----------------------------------------------------------------------
    # 4. Alignment metrics (kNN mixing + FOSCTTM)
    # -----------------------------------------------------------------------
    logger.info("Computing alignment metrics...")

    # Subsample for kNN mixing if combined data exceeds 30k cells (cKDTree is O(n^2) in high dims)
    max_cells_for_knn = 30000
    n_total = z_source_proj.shape[0] + z_target_proj.shape[0]
    if n_total > max_cells_for_knn:
        rng_sub = np.random.default_rng(42)
        frac = max_cells_for_knn / n_total
        n_s_sub = max(100, int(z_source_proj.shape[0] * frac))
        n_t_sub = max(100, int(z_target_proj.shape[0] * frac))
        idx_s = rng_sub.choice(z_source_proj.shape[0], n_s_sub, replace=False)
        idx_t = rng_sub.choice(z_target_proj.shape[0], n_t_sub, replace=False)
        z_combined = np.concatenate([z_source_proj[idx_s], z_target_proj[idx_t]], axis=0)
        species_labels = np.array([0] * n_s_sub + [1] * n_t_sub)
        logger.info("  Subsampled %d + %d = %d cells for kNN mixing (from %d total)",
                     n_s_sub, n_t_sub, n_s_sub + n_t_sub, n_total)
    else:
        z_combined = np.concatenate([z_source_proj, z_target_proj], axis=0)
        species_labels = np.array(
            [0] * z_source_proj.shape[0] + [1] * z_target_proj.shape[0]
        )

    for k in [10, 50, 100]:
        k_actual = min(k, z_combined.shape[0] - 1)
        score = alignment_score(z_combined, species_labels, k=k_actual)
        results[f"knn_mixing_k{k}"] = score
        logger.info("  kNN mixing (k=%d): %.4f", k, score)

    # NOTE: FOSCTTM removed — it used the first N cells (not the randomly
    # subsampled training cells) and measured the transport plan against its own
    # argmax, making it self-referential.  All values were ~0.5 (random).

    # -----------------------------------------------------------------------
    # 5. Auto-detect which side has annotations, set up label transfer
    # -----------------------------------------------------------------------
    source_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_source.obs.columns]
    target_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_target.obs.columns]

    if target_label_cols:
        # Target has labels -> transfer target labels to source (e.g., mouse -> human)
        label_cols = target_label_cols
        adata_labeled = adata_target
        adata_unlabeled = adata_source
        z_labeled = z_target_proj
        z_unlabeled = z_source_proj
        coords_unlabeled = coords_source
        labeled_name = args.target
        unlabeled_name = args.source
    elif source_label_cols:
        # Source has labels -> transfer source labels to target (e.g., human -> mouse)
        label_cols = source_label_cols
        adata_labeled = adata_source
        adata_unlabeled = adata_target
        z_labeled = z_source_proj
        z_unlabeled = z_target_proj
        coords_unlabeled = coords_target
        labeled_name = args.source
        unlabeled_name = args.target
    else:
        label_cols = []
        logger.warning("No known annotation columns found in either dataset, skipping label transfer")

    if label_cols:
        logger.info("=" * 60)
        logger.info("LABEL TRANSFER: %s -> %s", labeled_name, unlabeled_name)
        logger.info("  Annotation columns: %s", label_cols)
        logger.info("=" * 60)

    for col in label_cols:
        src_labels = np.array(adata_labeled.obs[col].values, dtype=str)
        n_classes = len(np.unique(src_labels))
        logger.info("  --- %s (%d classes) ---", col, n_classes)

        # Transfer labels via kNN
        pred_labels, confidences = label_transfer_with_confidence(
            z_labeled, z_unlabeled, src_labels, k=10,
        )

        results[f"label_transfer_{col}_n_classes"] = n_classes
        results[f"label_transfer_{col}_mean_confidence"] = float(np.mean(confidences))
        results[f"label_transfer_{col}_median_confidence"] = float(np.median(confidences))
        logger.info("    Mean confidence: %.3f, Median: %.3f",
                     np.mean(confidences), np.median(confidences))

        # Spatial label smoothness of transferred labels on unlabeled tissue
        smoothness = spatial_label_smoothness(coords_unlabeled, pred_labels, k=10)
        random_baseline = 1.0 / n_classes
        results[f"label_transfer_{col}_spatial_smoothness"] = smoothness
        results[f"label_transfer_{col}_random_baseline"] = random_baseline
        results[f"label_transfer_{col}_smoothness_vs_random"] = smoothness / random_baseline
        logger.info("    Spatial smoothness: %.4f (random baseline: %.4f, ratio: %.1fx)",
                     smoothness, random_baseline, smoothness / random_baseline)

        # Moran's I on one-hot transferred labels
        unique_labels = np.unique(pred_labels)
        onehot = np.zeros((len(pred_labels), len(unique_labels)), dtype=np.float32)
        for j, lbl in enumerate(unique_labels):
            onehot[:, j] = (pred_labels == lbl).astype(np.float32)
        mi = morans_i(onehot, coords_unlabeled, k=10)
        results[f"label_transfer_{col}_morans_i"] = mi
        logger.info("    Moran's I (one-hot): %.4f", mi)

        # Save transferred labels
        np.savez(
            eval_dir / f"transferred_labels_{col}.npz",
            labels=pred_labels,
            confidences=confidences,
        )

    # -----------------------------------------------------------------------
    # 6. Pseudo-label validation (ARI/NMI vs Leiden clustering on unlabeled side)
    # -----------------------------------------------------------------------
    if label_cols:
        logger.info("Computing pseudo-label validation (Leiden on unlabeled expression)...")
        adata_cluster = adata_unlabeled.copy()
        sc.pp.pca(adata_cluster, n_comps=min(50, adata_cluster.n_vars - 1))
        sc.pp.neighbors(adata_cluster, n_neighbors=15)

        for col in label_cols:
            npz_path = eval_dir / f"transferred_labels_{col}.npz"
            if not npz_path.exists():
                continue
            pred_labels = np.load(npz_path)["labels"]

            best_ari_col = -1.0
            best_res_col = None
            for resolution in LEIDEN_RESOLUTIONS:
                sc.tl.leiden(adata_cluster, resolution=resolution, key_added=f"leiden_{resolution}", flavor="igraph", directed=False, n_iterations=2)
                leiden_labels = np.array(adata_cluster.obs[f"leiden_{resolution}"].values, dtype=str)
                n_leiden = len(np.unique(leiden_labels))

                ari = adjusted_rand_score(pred_labels, leiden_labels)
                nmi = normalized_mutual_info_score(pred_labels, leiden_labels)
                results[f"pseudo_validation_{col}_leiden{resolution}_ari"] = ari
                results[f"pseudo_validation_{col}_leiden{resolution}_nmi"] = nmi
                results[f"pseudo_validation_{col}_leiden{resolution}_n_clusters"] = n_leiden
                if ari > best_ari_col:
                    best_ari_col = ari
                    best_res_col = resolution
                logger.info("    %s vs Leiden(res=%.1f, %d clusters): ARI=%.4f, NMI=%.4f",
                             col, resolution, n_leiden, ari, nmi)

            results[f"pseudo_validation_{col}_best_ari"] = best_ari_col
            results[f"pseudo_validation_{col}_best_resolution"] = best_res_col
            logger.info("    %s BEST: ARI=%.4f at resolution=%.1f", col, best_ari_col, best_res_col)

    # -----------------------------------------------------------------------
    # 7. Within-species control (upper bound for label transfer quality)
    # -----------------------------------------------------------------------
    if label_cols:
        logger.info("Computing within-species label transfer control (80/20 split)...")
        rng = np.random.default_rng(42)
        n_labeled = z_labeled.shape[0]
        perm = rng.permutation(n_labeled)
        split = int(0.8 * n_labeled)
        train_idx, test_idx = perm[:split], perm[split:]

        for col in label_cols:
            all_labels = np.array(adata_labeled.obs[col].values, dtype=str)
            # BRIDGE embedding-based control
            acc = label_transfer_accuracy(
                z_labeled[train_idx], z_labeled[test_idx],
                all_labels[train_idx], all_labels[test_idx], k=10,
            )
            results[f"within_species_{col}_accuracy"] = acc
            logger.info("  Within-species %s accuracy (BRIDGE emb): %.4f (chance: %.4f)",
                         col, acc, 1.0 / len(np.unique(all_labels)))

        # Expression-only control (PCA kNN as fair upper bound, independent of BRIDGE)
        logger.info("  Computing expression-only within-species control (PCA kNN)...")
        from sklearn.decomposition import PCA as _PCA
        X_labeled_raw = adata_labeled.X
        if sp.issparse(X_labeled_raw):
            X_labeled_raw = X_labeled_raw.toarray()
        X_labeled_raw = np.array(X_labeled_raw, dtype=np.float32)
        n_pcs_ctrl = min(50, X_labeled_raw.shape[1] - 1)
        pca_ctrl = _PCA(n_components=n_pcs_ctrl)
        X_labeled_pca = pca_ctrl.fit_transform(X_labeled_raw)

        for col in label_cols:
            all_labels = np.array(adata_labeled.obs[col].values, dtype=str)
            acc_pca = label_transfer_accuracy(
                X_labeled_pca[train_idx], X_labeled_pca[test_idx],
                all_labels[train_idx], all_labels[test_idx], k=10,
            )
            results[f"within_species_{col}_accuracy_pca"] = acc_pca
            logger.info("  Within-species %s accuracy (PCA): %.4f",
                         col, acc_pca)

    # -----------------------------------------------------------------------
    # 7b. Permutation null for ARI (proves observed ARI >> chance)
    # -----------------------------------------------------------------------
    if label_cols:
        N_PERMUTATIONS = 100
        logger.info("Computing permutation null for ARI (%d permutations)...", N_PERMUTATIONS)
        rng_perm_null = np.random.default_rng(0)

        for col in label_cols:
            npz_path = eval_dir / f"transferred_labels_{col}.npz"
            if not npz_path.exists():
                continue
            pred_labels = np.load(npz_path)["labels"]

            null_aris = []
            for perm_i in range(N_PERMUTATIONS):
                shuffled = rng_perm_null.permutation(pred_labels)
                best_null_ari = -1.0
                for resolution in LEIDEN_RESOLUTIONS:
                    key = f"leiden_{resolution}"
                    if key in adata_cluster.obs.columns:
                        leiden_labels = np.array(adata_cluster.obs[key].values, dtype=str)
                        ari_null = adjusted_rand_score(shuffled, leiden_labels)
                        best_null_ari = max(best_null_ari, ari_null)
                null_aris.append(best_null_ari)

            null_aris = np.array(null_aris)
            observed_ari = results.get(f"pseudo_validation_{col}_best_ari", 0.0)
            p_value = float(np.mean(null_aris >= observed_ari))

            results[f"permutation_null_{col}_mean"] = float(null_aris.mean())
            results[f"permutation_null_{col}_std"] = float(null_aris.std())
            results[f"permutation_null_{col}_95th"] = float(np.percentile(null_aris, 95))
            results[f"permutation_null_{col}_p_value"] = p_value
            logger.info("  %s: null=%.4f+/-%.4f (95th=%.4f), observed=%.4f, p=%.4f",
                         col, null_aris.mean(), null_aris.std(),
                         np.percentile(null_aris, 95), observed_ari, p_value)

    # -----------------------------------------------------------------------
    # 8. Baselines
    # -----------------------------------------------------------------------
    if label_cols:
        logger.info("Computing baselines...")

        # 8a. Random baseline: permute labeled embeddings, transfer, check smoothness
        logger.info("  Random baseline (5 permutations)...")
        n_labeled = z_labeled.shape[0]
        for col in label_cols:
            all_labels = np.array(adata_labeled.obs[col].values, dtype=str)
            random_smoothness_list = []
            for seed in range(5):
                rng_perm = np.random.default_rng(seed)
                z_labeled_perm = z_labeled[rng_perm.permutation(n_labeled)]
                pred_rand, _ = label_transfer_with_confidence(
                    z_labeled_perm, z_unlabeled, all_labels, k=10,
                )
                s = spatial_label_smoothness(coords_unlabeled, pred_rand, k=10)
                random_smoothness_list.append(s)
            mean_random = float(np.mean(random_smoothness_list))
            results[f"baseline_random_{col}_spatial_smoothness"] = mean_random
            logger.info("    Random %s spatial smoothness: %.4f (±%.4f)",
                         col, mean_random, np.std(random_smoothness_list))

        # 8b. Build shared ortholog gene space (used by expression and Harmony baselines)
        source_species = source_config["species"]
        target_species = target_config["species"]
        X_src_shared = X_tgt_shared = None
        shared_genes = []
        adata_labeled_raw = adata_unlabeled_raw = None
        coords_unlabeled_raw = None
        try:
            ortholog_map = load_ortholog_mapping(source_species, target_species)
            adata_src_raw = ad.read_h5ad(source_h5ad)
            adata_tgt_raw = ad.read_h5ad(target_h5ad)
            X_src_shared, X_tgt_shared, shared_genes = build_shared_gene_space(
                adata_src_raw, adata_tgt_raw, ortholog_map,
            )
            logger.info("    Shared ortholog genes: %d", len(shared_genes))
            if len(shared_genes) > 0:
                if target_label_cols:
                    adata_labeled_raw = adata_tgt_raw
                    adata_unlabeled_raw = adata_src_raw
                    coords_unlabeled_raw = np.array(adata_src_raw.obsm["spatial"][:, :2], dtype=np.float32)
                else:
                    adata_labeled_raw = adata_src_raw
                    adata_unlabeled_raw = adata_tgt_raw
                    coords_unlabeled_raw = np.array(adata_tgt_raw.obsm["spatial"][:, :2], dtype=np.float32)
        except Exception as e:
            logger.warning("    Failed to build shared gene space: %s", e)

        # 8c. Expression-only baseline: kNN in shared ortholog gene space
        if len(shared_genes) > 0:
            logger.info("  Expression-only baseline (shared ortholog genes)...")
            try:
                if target_label_cols:
                    X_labeled_shared, X_unlabeled_shared = X_tgt_shared, X_src_shared
                else:
                    X_labeled_shared, X_unlabeled_shared = X_src_shared, X_tgt_shared

                for col in label_cols:
                    if col not in adata_labeled_raw.obs.columns:
                        continue
                    raw_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)
                    pred_expr, _ = label_transfer_with_confidence(
                        X_labeled_shared, X_unlabeled_shared, raw_labels, k=10,
                    )
                    s_expr = spatial_label_smoothness(coords_unlabeled_raw, pred_expr, k=10)
                    results[f"baseline_expression_{col}_spatial_smoothness"] = s_expr
                    logger.info("    Expression-only %s spatial smoothness: %.4f", col, s_expr)

                    # ARI vs Leiden — align cells if count mismatch
                    pred_for_ari = pred_expr
                    if len(pred_expr) != adata_cluster.n_obs:
                        # Subset raw predictions to preprocessed cell indices
                        raw_ids = list(adata_unlabeled_raw.obs_names)
                        preproc_ids = list(adata_cluster.obs_names)
                        idx_map = {name: i for i, name in enumerate(raw_ids)}
                        keep = [idx_map[n] for n in preproc_ids if n in idx_map]
                        if len(keep) == adata_cluster.n_obs:
                            pred_for_ari = pred_expr[keep]
                            logger.info("    Aligned %d raw -> %d preprocessed cells for ARI", len(pred_expr), len(keep))
                        else:
                            logger.info("    Skipping expression ARI: cannot align cells (raw=%d, preprocessed=%d, matched=%d)",
                                         len(pred_expr), adata_cluster.n_obs, len(keep))
                            pred_for_ari = None
                    if pred_for_ari is not None:
                        for resolution in LEIDEN_RESOLUTIONS:
                            key = f"leiden_{resolution}"
                            if key in adata_cluster.obs.columns:
                                leiden_labels_expr = np.array(adata_cluster.obs[key].values, dtype=str)
                                ari_expr = adjusted_rand_score(pred_for_ari, leiden_labels_expr)
                                results[f"baseline_expression_{col}_leiden{resolution}_ari"] = ari_expr
                                logger.info("    Expression-only %s vs Leiden(res=%.1f): ARI=%.4f", col, resolution, ari_expr)
            except Exception as e:
                logger.warning("    Expression baseline failed: %s", e)
        else:
            logger.warning("    No shared genes found, skipping expression baseline")

        # 8d. Harmony baseline: batch correction on shared ortholog gene space
        if len(shared_genes) > 0:
            logger.info("  Harmony baseline (batch correction on shared genes)...")
            try:
                import harmonypy as hm
                import pandas as pd

                X_combined = np.vstack([X_src_shared, X_tgt_shared])
                n_src_h = X_src_shared.shape[0]
                meta = pd.DataFrame({"batch": ["source"] * n_src_h + ["target"] * X_tgt_shared.shape[0]})

                from sklearn.decomposition import PCA
                n_pcs = min(30, X_combined.shape[1] - 1)
                pca = PCA(n_components=n_pcs)
                X_pca = pca.fit_transform(X_combined)

                ho = hm.run_harmony(X_pca, meta, ["batch"], max_iter_harmony=20, device="cpu")
                X_harmony = np.array(ho.Z_corr, dtype=np.float32)

                X_harmony_src = X_harmony[:n_src_h]
                X_harmony_tgt = X_harmony[n_src_h:]

                if target_label_cols:
                    X_harmony_labeled, X_harmony_unlabeled = X_harmony_tgt, X_harmony_src
                else:
                    X_harmony_labeled, X_harmony_unlabeled = X_harmony_src, X_harmony_tgt

                # kNN mixing in Harmony space
                X_harmony_all = np.vstack([X_harmony_labeled, X_harmony_unlabeled])
                n_lab_h = X_harmony_labeled.shape[0]
                species_h = np.array([0] * n_lab_h + [1] * X_harmony_unlabeled.shape[0])
                for k in [10, 50, 100]:
                    k_eff = min(k, X_harmony_all.shape[0] - 1)
                    mix = alignment_score(X_harmony_all, species_h, k=k_eff)
                    results[f"baseline_harmony_knn_mixing_k{k}"] = mix
                    logger.info("    Harmony kNN mixing (k=%d): %.4f", k, mix)

                for col in label_cols:
                    if col not in adata_labeled_raw.obs.columns:
                        continue
                    raw_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)
                    pred_harmony, conf_harmony = label_transfer_with_confidence(
                        X_harmony_labeled, X_harmony_unlabeled, raw_labels, k=10,
                    )
                    s_harmony = spatial_label_smoothness(coords_unlabeled_raw, pred_harmony, k=10)
                    results[f"baseline_harmony_{col}_spatial_smoothness"] = s_harmony
                    results[f"baseline_harmony_{col}_mean_confidence"] = float(np.mean(conf_harmony))
                    logger.info("    Harmony %s spatial smoothness: %.4f, confidence: %.4f",
                                 col, s_harmony, np.mean(conf_harmony))

                    # Harmony ARI vs Leiden — align cells if count mismatch
                    pred_h_ari = pred_harmony
                    if len(pred_harmony) != adata_cluster.n_obs:
                        raw_ids = list(adata_unlabeled_raw.obs_names)
                        preproc_ids = list(adata_cluster.obs_names)
                        idx_map = {name: i for i, name in enumerate(raw_ids)}
                        keep = [idx_map[n] for n in preproc_ids if n in idx_map]
                        if len(keep) == adata_cluster.n_obs:
                            pred_h_ari = pred_harmony[keep]
                            logger.info("    Aligned %d raw -> %d preprocessed cells for Harmony ARI", len(pred_harmony), len(keep))
                        else:
                            pred_h_ari = None
                    if pred_h_ari is not None:
                        for resolution in LEIDEN_RESOLUTIONS:
                            key = f"leiden_{resolution}"
                            if key in adata_cluster.obs.columns:
                                leiden_labels_h = np.array(adata_cluster.obs[key].values, dtype=str)
                                ari_harmony = adjusted_rand_score(pred_h_ari, leiden_labels_h)
                                results[f"baseline_harmony_{col}_leiden{resolution}_ari"] = ari_harmony
                                logger.info("    Harmony %s vs Leiden(res=%.1f): ARI=%.4f", col, resolution, ari_harmony)
            except ImportError:
                logger.warning("    harmonypy not installed, skipping Harmony baseline")
            except Exception as e:
                logger.warning("    Harmony baseline failed: %s", e)
        else:
            logger.warning("    No shared genes found, skipping Harmony baseline")

        # 8e. Scanorama baseline: batch correction on shared ortholog gene space
        if len(shared_genes) > 0:
            logger.info("  Scanorama baseline (batch correction on shared genes)...")
            try:
                import scanorama

                gene_names = [str(g) for g in shared_genes]
                datasets_corrected, _ = scanorama.correct(
                    [X_src_shared, X_tgt_shared],
                    [gene_names, gene_names],
                )
                X_scan_src = np.array(datasets_corrected[0].todense()
                                      if sp.issparse(datasets_corrected[0])
                                      else datasets_corrected[0], dtype=np.float32)
                X_scan_tgt = np.array(datasets_corrected[1].todense()
                                      if sp.issparse(datasets_corrected[1])
                                      else datasets_corrected[1], dtype=np.float32)

                if target_label_cols:
                    X_scan_labeled, X_scan_unlabeled = X_scan_tgt, X_scan_src
                else:
                    X_scan_labeled, X_scan_unlabeled = X_scan_src, X_scan_tgt

                # kNN mixing in Scanorama space
                X_scan_all = np.vstack([X_scan_labeled, X_scan_unlabeled])
                n_lab_sc = X_scan_labeled.shape[0]
                species_sc = np.array([0] * n_lab_sc + [1] * X_scan_unlabeled.shape[0])
                for k in [10, 50, 100]:
                    k_eff = min(k, X_scan_all.shape[0] - 1)
                    mix = alignment_score(X_scan_all, species_sc, k=k_eff)
                    results[f"baseline_scanorama_knn_mixing_k{k}"] = mix
                    logger.info("    Scanorama kNN mixing (k=%d): %.4f", k, mix)

                for col in label_cols:
                    if col not in adata_labeled_raw.obs.columns:
                        continue
                    raw_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)
                    pred_scan, conf_scan = label_transfer_with_confidence(
                        X_scan_labeled, X_scan_unlabeled, raw_labels, k=10,
                    )
                    s_scan = spatial_label_smoothness(coords_unlabeled_raw, pred_scan, k=10)
                    results[f"baseline_scanorama_{col}_spatial_smoothness"] = s_scan
                    results[f"baseline_scanorama_{col}_mean_confidence"] = float(np.mean(conf_scan))
                    logger.info("    Scanorama %s spatial smoothness: %.4f, confidence: %.4f",
                                 col, s_scan, np.mean(conf_scan))

                    # Scanorama ARI vs Leiden — align cells if count mismatch
                    pred_sc_ari = pred_scan
                    if len(pred_scan) != adata_cluster.n_obs:
                        raw_ids = list(adata_unlabeled_raw.obs_names)
                        preproc_ids = list(adata_cluster.obs_names)
                        idx_map = {name: i for i, name in enumerate(raw_ids)}
                        keep = [idx_map[n] for n in preproc_ids if n in idx_map]
                        if len(keep) == adata_cluster.n_obs:
                            pred_sc_ari = pred_scan[keep]
                            logger.info("    Aligned %d raw -> %d preprocessed cells for Scanorama ARI", len(pred_scan), len(keep))
                        else:
                            pred_sc_ari = None
                    if pred_sc_ari is not None:
                        for resolution in LEIDEN_RESOLUTIONS:
                            key = f"leiden_{resolution}"
                            if key in adata_cluster.obs.columns:
                                leiden_labels_sc = np.array(adata_cluster.obs[key].values, dtype=str)
                                ari_scan = adjusted_rand_score(pred_sc_ari, leiden_labels_sc)
                                results[f"baseline_scanorama_{col}_leiden{resolution}_ari"] = ari_scan
                                logger.info("    Scanorama %s vs Leiden(res=%.1f): ARI=%.4f", col, resolution, ari_scan)
            except ImportError:
                logger.warning("    scanorama not installed, skipping Scanorama baseline")
            except Exception as e:
                logger.warning("    Scanorama baseline failed: %s", e)
        else:
            logger.warning("    No shared genes found, skipping Scanorama baseline")

        # 8f. BBKNN baseline: batch correction via graph-based approach
        if len(shared_genes) > 0:
            logger.info("  BBKNN baseline (graph-based batch correction)...")
            try:
                import bbknn

                X_combined_bb = np.vstack([X_src_shared, X_tgt_shared])
                n_src_bb = X_src_shared.shape[0]
                adata_bb = ad.AnnData(X=X_combined_bb)
                adata_bb.obs["batch"] = (["source"] * n_src_bb
                                         + ["target"] * X_tgt_shared.shape[0])

                n_pcs_bb = min(30, X_combined_bb.shape[1] - 1)
                sc.pp.pca(adata_bb, n_comps=n_pcs_bb)
                bbknn.bbknn(adata_bb, batch_key="batch", n_pcs=n_pcs_bb)
                # Use PCA space (30D) for downstream metrics — NOT 2D UMAP
                X_bbknn = adata_bb.obsm["X_pca"]

                X_bbknn_src = X_bbknn[:n_src_bb]
                X_bbknn_tgt = X_bbknn[n_src_bb:]

                if target_label_cols:
                    X_bb_labeled, X_bb_unlabeled = X_bbknn_tgt, X_bbknn_src
                else:
                    X_bb_labeled, X_bb_unlabeled = X_bbknn_src, X_bbknn_tgt

                # kNN mixing in BBKNN space
                X_bb_all = np.vstack([X_bb_labeled, X_bb_unlabeled])
                n_lab_bb = X_bb_labeled.shape[0]
                species_bb = np.array([0] * n_lab_bb + [1] * X_bb_unlabeled.shape[0])
                for k in [10, 50, 100]:
                    k_eff = min(k, X_bb_all.shape[0] - 1)
                    mix = alignment_score(X_bb_all, species_bb, k=k_eff)
                    results[f"baseline_bbknn_knn_mixing_k{k}"] = mix
                    logger.info("    BBKNN kNN mixing (k=%d): %.4f", k, mix)

                for col in label_cols:
                    if col not in adata_labeled_raw.obs.columns:
                        continue
                    raw_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)
                    pred_bb, conf_bb = label_transfer_with_confidence(
                        X_bb_labeled, X_bb_unlabeled, raw_labels, k=10,
                    )
                    s_bb = spatial_label_smoothness(coords_unlabeled_raw, pred_bb, k=10)
                    results[f"baseline_bbknn_{col}_spatial_smoothness"] = s_bb
                    results[f"baseline_bbknn_{col}_mean_confidence"] = float(np.mean(conf_bb))
                    logger.info("    BBKNN %s spatial smoothness: %.4f, confidence: %.4f",
                                 col, s_bb, np.mean(conf_bb))

                    # BBKNN ARI vs Leiden — align cells if count mismatch
                    pred_bb_ari = pred_bb
                    if len(pred_bb) != adata_cluster.n_obs:
                        raw_ids = list(adata_unlabeled_raw.obs_names)
                        preproc_ids = list(adata_cluster.obs_names)
                        idx_map = {name: i for i, name in enumerate(raw_ids)}
                        keep = [idx_map[n] for n in preproc_ids if n in idx_map]
                        if len(keep) == adata_cluster.n_obs:
                            pred_bb_ari = pred_bb[keep]
                            logger.info("    Aligned %d raw -> %d preprocessed cells for BBKNN ARI", len(pred_bb), len(keep))
                        else:
                            pred_bb_ari = None
                    if pred_bb_ari is not None:
                        for resolution in LEIDEN_RESOLUTIONS:
                            key = f"leiden_{resolution}"
                            if key in adata_cluster.obs.columns:
                                leiden_labels_bb = np.array(adata_cluster.obs[key].values, dtype=str)
                                ari_bb = adjusted_rand_score(pred_bb_ari, leiden_labels_bb)
                                results[f"baseline_bbknn_{col}_leiden{resolution}_ari"] = ari_bb
                                logger.info("    BBKNN %s vs Leiden(res=%.1f): ARI=%.4f", col, resolution, ari_bb)
            except ImportError:
                logger.warning("    bbknn not installed, skipping BBKNN baseline")
            except Exception as e:
                logger.warning("    BBKNN baseline failed: %s", e)
        else:
            logger.warning("    No shared genes found, skipping BBKNN baseline")

        # 8g. PASTE baseline: spatial FGW alignment (transport plan → label transfer)
        if len(shared_genes) > 0:
            logger.info("  PASTE baseline (spatial FGW optimal transport)...")
            try:
                import paste

                # Monkey-patch PASTE's line_search to accept POT 0.9+ signature
                # POT 0.9 added df_G as 6th positional arg; PASTE only expects 5
                _orig_fgw = paste.PASTE.my_fused_gromov_wasserstein

                def _patched_fgw(*args, **kwargs):
                    import functools
                    import ot as _ot

                    # Patch ot.optim.cg temporarily to wrap line_search
                    _orig_cg = _ot.optim.cg

                    @functools.wraps(_orig_cg)
                    def _cg_wrapper(p, q, M, alpha, f, df, G0, line_search_fn,
                                    *cg_args, **cg_kwargs):
                        @functools.wraps(line_search_fn)
                        def _ls_compat(cost, G, deltaG, Mi, cost_G, *ls_extra,
                                       **ls_kw):
                            # Drop the extra df_G arg that POT 0.9 passes
                            return line_search_fn(cost, G, deltaG, Mi, cost_G,
                                                  **ls_kw)
                        return _orig_cg(p, q, M, alpha, f, df, G0,
                                        _ls_compat, *cg_args, **cg_kwargs)

                    _ot.optim.cg = _cg_wrapper
                    try:
                        return _orig_fgw(*args, **kwargs)
                    finally:
                        _ot.optim.cg = _orig_cg

                paste.PASTE.my_fused_gromov_wasserstein = _patched_fgw

                # Build AnnData with shared genes + spatial coords for PASTE
                adata_paste_src = ad.AnnData(
                    X=X_src_shared.astype(np.float64),
                    obsm={"spatial": np.array(adata_src_raw.obsm["spatial"][:, :2],
                                              dtype=np.float64)},
                )
                adata_paste_src.var_names = [str(g) for g in shared_genes]

                adata_paste_tgt = ad.AnnData(
                    X=X_tgt_shared.astype(np.float64),
                    obsm={"spatial": np.array(adata_tgt_raw.obsm["spatial"][:, :2],
                                              dtype=np.float64)},
                )
                adata_paste_tgt.var_names = [str(g) for g in shared_genes]

                # Subsample if too large (PASTE FGW is O(n^3))
                max_paste = 2000
                n_src_p = adata_paste_src.n_obs
                n_tgt_p = adata_paste_tgt.n_obs
                paste_src_idx = np.arange(n_src_p)
                paste_tgt_idx = np.arange(n_tgt_p)
                if n_src_p > max_paste:
                    paste_src_idx = np.random.default_rng(42).choice(
                        n_src_p, max_paste, replace=False)
                    adata_paste_src = adata_paste_src[paste_src_idx].copy()
                if n_tgt_p > max_paste:
                    paste_tgt_idx = np.random.default_rng(42).choice(
                        n_tgt_p, max_paste, replace=False)
                    adata_paste_tgt = adata_paste_tgt[paste_tgt_idx].copy()

                # alpha=0.1: 90% expression + 10% spatial structure (FGW)
                # Note: spatial component uses *intra-slice* distances (preserves
                # local neighborhood structure), not inter-slice coordinates
                logger.info("    PASTE: aligning %d x %d cells (alpha=0.1, FGW)...",
                            adata_paste_src.n_obs, adata_paste_tgt.n_obs)
                pi_paste = paste.pairwise_align(
                    adata_paste_src, adata_paste_tgt,
                    alpha=0.1, numItermax=50,
                )
                logger.info("    PASTE transport plan shape: %s", pi_paste.shape)

                # Check if transport plan is degenerate (near-uniform)
                pi_row_norm = pi_paste / (pi_paste.sum(axis=1, keepdims=True) + 1e-20)
                row_entropy = -(pi_row_norm * np.log(pi_row_norm + 1e-20)).sum(axis=1)
                max_entropy = np.log(pi_paste.shape[1])
                mean_entropy_ratio = row_entropy.mean() / max_entropy
                logger.info("    PASTE transport plan entropy ratio: %.4f (1.0 = uniform/degenerate)",
                            mean_entropy_ratio)
                if mean_entropy_ratio > 0.99:
                    logger.warning("    PASTE transport plan is near-uniform (degenerate). "
                                   "PASTE is designed for serial sections, not cross-species alignment.")

                # Label transfer via transport plan
                if target_label_cols:
                    paste_labeled_idx = paste_tgt_idx
                    paste_unlabeled_idx = paste_src_idx
                    paste_labeled_raw = adata_tgt_raw
                    paste_coords_unlabeled = np.array(
                        adata_src_raw.obsm["spatial"][:, :2], dtype=np.float32
                    )[paste_src_idx]
                    # Transport plan: src x tgt, transfer tgt->src
                    T = pi_paste  # (n_src_sub, n_tgt_sub)
                else:
                    paste_labeled_idx = paste_src_idx
                    paste_unlabeled_idx = paste_tgt_idx
                    paste_labeled_raw = adata_src_raw
                    paste_coords_unlabeled = np.array(
                        adata_tgt_raw.obsm["spatial"][:, :2], dtype=np.float32
                    )[paste_tgt_idx]
                    T = pi_paste.T  # Transpose: now (n_tgt_sub, n_src_sub)

                # Normalize transport plan row-wise for label transfer
                T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-10)

                for col in label_cols:
                    if col not in paste_labeled_raw.obs.columns:
                        continue
                    all_labels = np.array(paste_labeled_raw.obs[col].values, dtype=str)
                    labeled_labels = all_labels[paste_labeled_idx]
                    unique_labels = np.unique(labeled_labels)
                    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
                    # One-hot encode labels
                    Y_onehot = np.zeros((len(labeled_labels), len(unique_labels)))
                    for i, l in enumerate(labeled_labels):
                        Y_onehot[i, label_to_idx[l]] = 1.0
                    # Transfer via transport plan
                    pred_probs = T_norm @ Y_onehot
                    pred_paste = unique_labels[pred_probs.argmax(axis=1)]

                    s_paste = spatial_label_smoothness(
                        paste_coords_unlabeled, pred_paste, k=10)
                    results[f"baseline_paste_{col}_spatial_smoothness"] = s_paste
                    logger.info("    PASTE %s spatial smoothness: %.4f", col, s_paste)

                # Restore original function
                paste.PASTE.my_fused_gromov_wasserstein = _orig_fgw

            except ImportError:
                logger.warning("    paste-bio not installed, skipping PASTE baseline")
            except Exception as e:
                logger.warning("    PASTE baseline failed: %s", e)
                import traceback
                traceback.print_exc()
        else:
            logger.warning("    No shared genes found, skipping PASTE baseline")

        # 8h. scVI/scANVI baseline: VAE-based integration
        if len(shared_genes) > 0:
            logger.info("  scVI baseline (VAE-based batch integration)...")
            try:
                import scvi

                # Build raw-count shared gene space (scVI needs counts, not z-scores)
                # Use _get_gene_lookup to handle Ensembl IDs in var_names
                src_gene_lookup = _get_gene_lookup(adata_src_raw)
                tgt_gene_lookup = _get_gene_lookup(adata_tgt_raw)
                ortholog_map_vi = load_ortholog_mapping(
                    source_species, target_species)

                vi_src_cols, vi_tgt_cols, vi_gene_names = [], [], []
                for sg, tg in ortholog_map_vi.forward.items():
                    if sg in src_gene_lookup and tg in tgt_gene_lookup:
                        vi_src_cols.append(src_gene_lookup[sg])
                        vi_tgt_cols.append(tgt_gene_lookup[tg])
                        vi_gene_names.append(sg)

                X_src_raw_vi = adata_src_raw.X[:, vi_src_cols]
                X_tgt_raw_vi = adata_tgt_raw.X[:, vi_tgt_cols]
                if sp.issparse(X_src_raw_vi):
                    X_src_raw_vi = np.array(X_src_raw_vi.todense())
                if sp.issparse(X_tgt_raw_vi):
                    X_tgt_raw_vi = np.array(X_tgt_raw_vi.todense())

                X_joint_vi = np.vstack([X_src_raw_vi, X_tgt_raw_vi]).astype(np.float32)
                n_src_vi = X_src_raw_vi.shape[0]

                # If data has negative values, undo log1p to get approximate counts
                if X_joint_vi.min() < 0:
                    logger.info("    Negative values found, applying expm1 to approximate counts")
                    X_joint_vi = np.expm1(np.abs(X_joint_vi))
                X_joint_vi = np.round(X_joint_vi).astype(np.float32)
                X_joint_vi = np.maximum(X_joint_vi, 0)  # ensure non-negative

                adata_vi = ad.AnnData(X=X_joint_vi)
                adata_vi.var_names = [str(g) for g in vi_gene_names]
                adata_vi.obs["batch"] = (
                    ["source"] * n_src_vi + ["target"] * X_tgt_raw_vi.shape[0]
                )
                adata_vi.obs["batch"] = adata_vi.obs["batch"].astype("category")

                # Subsample if very large (scVI scales well but we want speed)
                max_scvi = 10000
                if adata_vi.n_obs > max_scvi:
                    sc.pp.subsample(adata_vi, n_obs=max_scvi, random_state=42)
                    logger.info("    Subsampled to %d cells for scVI", adata_vi.n_obs)
                    n_src_vi = (adata_vi.obs["batch"] == "source").sum()

                # Setup and train scVI model
                scvi.model.SCVI.setup_anndata(adata_vi, batch_key="batch")
                vae = scvi.model.SCVI(
                    adata_vi,
                    n_latent=30,
                    n_layers=2,
                    gene_likelihood="nb",
                )
                vae.train(
                    max_epochs=100,
                    early_stopping=True,
                    batch_size=256,
                    train_size=0.9,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1,
                )

                # Get latent representation
                Z_vi = vae.get_latent_representation()
                src_mask_vi = adata_vi.obs["batch"] == "source"
                tgt_mask_vi = adata_vi.obs["batch"] == "target"
                Z_vi_src = Z_vi[src_mask_vi.values]
                Z_vi_tgt = Z_vi[tgt_mask_vi.values]

                # Map joint indices back to original dataset indices
                joint_idx = adata_vi.obs.index.astype(int).values
                src_orig_idx = joint_idx[src_mask_vi.values]
                tgt_orig_idx = joint_idx[tgt_mask_vi.values] - n_src_vi

                if target_label_cols:
                    Z_vi_labeled, Z_vi_unlabeled = Z_vi_tgt, Z_vi_src
                    vi_labeled_raw = adata_tgt_raw
                    vi_labeled_orig_idx = tgt_orig_idx
                    vi_coords_unlabeled = np.array(
                        adata_src_raw.obsm["spatial"][:, :2], dtype=np.float32)
                    if adata_vi.n_obs < X_joint_vi.shape[0]:
                        vi_coords_unlabeled = vi_coords_unlabeled[src_orig_idx]
                else:
                    Z_vi_labeled, Z_vi_unlabeled = Z_vi_src, Z_vi_tgt
                    vi_labeled_raw = adata_src_raw
                    vi_labeled_orig_idx = src_orig_idx
                    vi_coords_unlabeled = np.array(
                        adata_tgt_raw.obsm["spatial"][:, :2], dtype=np.float32)
                    if adata_vi.n_obs < X_joint_vi.shape[0]:
                        vi_coords_unlabeled = vi_coords_unlabeled[tgt_orig_idx]

                # kNN mixing in scVI latent space
                Z_vi_all = np.vstack([Z_vi_labeled, Z_vi_unlabeled])
                n_lab_vi = Z_vi_labeled.shape[0]
                species_vi = np.array([0] * n_lab_vi + [1] * Z_vi_unlabeled.shape[0])
                for k in [10, 50, 100]:
                    k_eff = min(k, Z_vi_all.shape[0] - 1)
                    mix = alignment_score(Z_vi_all, species_vi, k=k_eff)
                    results[f"baseline_scvi_knn_mixing_k{k}"] = mix
                    logger.info("    scVI kNN mixing (k=%d): %.4f", k, mix)

                for col in label_cols:
                    if col not in vi_labeled_raw.obs.columns:
                        continue
                    raw_labels = np.array(vi_labeled_raw.obs[col].values, dtype=str)
                    if Z_vi_labeled.shape[0] < len(raw_labels):
                        raw_labels = raw_labels[vi_labeled_orig_idx]

                    pred_vi, conf_vi = label_transfer_with_confidence(
                        Z_vi_labeled, Z_vi_unlabeled, raw_labels, k=10,
                    )
                    # Only compute spatial smoothness if coords match
                    if len(pred_vi) <= len(vi_coords_unlabeled):
                        s_vi = spatial_label_smoothness(
                            vi_coords_unlabeled[:len(pred_vi)], pred_vi, k=10)
                        results[f"baseline_scvi_{col}_spatial_smoothness"] = s_vi
                        logger.info("    scVI %s spatial smoothness: %.4f", col, s_vi)
                    results[f"baseline_scvi_{col}_mean_confidence"] = float(
                        np.mean(conf_vi))
                    logger.info("    scVI %s confidence: %.4f", col, np.mean(conf_vi))

            except ImportError:
                logger.warning("    scvi-tools not installed, skipping scVI baseline")
            except Exception as e:
                logger.warning("    scVI baseline failed: %s", e)
                import traceback
                traceback.print_exc()
        else:
            logger.warning("    No shared genes found, skipping scVI baseline")

    # -----------------------------------------------------------------------
    # 9. Visualizations
    # -----------------------------------------------------------------------
    logger.info("Generating visualizations...")

    # Pick the primary label column for visualization (first available)
    viz_col = label_cols[0] if label_cols else None

    # Determine labeled/unlabeled coords for plots
    if target_label_cols:
        coords_labeled = coords_target
    elif source_label_cols:
        coords_labeled = coords_source
    else:
        coords_labeled = None

    # 9a. Joint UMAP colored by species and by transferred cell type
    try:
        from umap import UMAP

        # Subsample for UMAP if dataset is large
        max_umap = 20000
        n_s_full, n_t_full = z_source_proj.shape[0], z_target_proj.shape[0]
        if n_s_full + n_t_full > max_umap:
            rng_umap = np.random.default_rng(42)
            frac_u = max_umap / (n_s_full + n_t_full)
            n_s_u = max(500, int(n_s_full * frac_u))
            n_t_u = max(500, int(n_t_full * frac_u))
            idx_s_u = rng_umap.choice(n_s_full, n_s_u, replace=False)
            idx_t_u = rng_umap.choice(n_t_full, n_t_u, replace=False)
            z_umap_input = np.concatenate([z_source_proj[idx_s_u], z_target_proj[idx_t_u]])
            n_s_umap = n_s_u
            logger.info("  Subsampled %d + %d cells for UMAP", n_s_u, n_t_u)
        else:
            z_umap_input = np.concatenate([z_source_proj, z_target_proj])
            n_s_umap = n_s_full
            idx_s_u = np.arange(n_s_full)
            idx_t_u = np.arange(n_t_full)

        reducer = UMAP(n_components=2, random_state=42)
        z_umap = reducer.fit_transform(z_umap_input)

        # UMAP by species
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(z_umap[:n_s_umap, 0], z_umap[:n_s_umap, 1],
                   c="tab:blue", alpha=0.3, s=3, label=args.source, rasterized=True)
        ax.scatter(z_umap[n_s_umap:, 0], z_umap[n_s_umap:, 1],
                   c="tab:orange", alpha=0.3, s=3, label=args.target, rasterized=True)
        ax.legend(fontsize=11)
        ax.set_title("Joint UMAP — colored by species")
        ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(eval_dir / "umap_species.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved umap_species.png")

        # UMAP colored by transferred cell type
        if viz_col:
            npz_path = eval_dir / f"transferred_labels_{viz_col}.npz"
            if npz_path.exists():
                pred_labels_full = np.load(npz_path)["labels"]
                orig_labels_full = np.array(adata_labeled.obs[viz_col].values, dtype=str)

                # Subsample labels to match UMAP indices
                if target_label_cols:
                    pred_labels_sub = pred_labels_full[idx_s_u]
                    orig_labels_sub = orig_labels_full[idx_t_u]
                    umap_unlabeled = z_umap[:n_s_umap]
                    umap_labeled = z_umap[n_s_umap:]
                else:
                    pred_labels_sub = pred_labels_full[idx_t_u]
                    orig_labels_sub = orig_labels_full[idx_s_u]
                    umap_unlabeled = z_umap[n_s_umap:]
                    umap_labeled = z_umap[:n_s_umap]

                all_labels_viz = np.concatenate([pred_labels_sub, orig_labels_sub])
                unique_types = np.unique(all_labels_viz)
                cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_types))
                label_to_idx = {l: i for i, l in enumerate(unique_types)}

                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                axes[0].scatter(umap_unlabeled[:, 0], umap_unlabeled[:, 1],
                                c=[label_to_idx[l] for l in pred_labels_sub],
                                cmap=cmap, alpha=0.5, s=3, vmin=0,
                                vmax=len(unique_types) - 1, rasterized=True)
                axes[0].set_title(f"{unlabeled_name} (transferred labels)")
                axes[0].set_xticks([]); axes[0].set_yticks([])
                axes[1].scatter(umap_labeled[:, 0], umap_labeled[:, 1],
                                c=[label_to_idx[l] for l in orig_labels_sub],
                                cmap=cmap, alpha=0.5, s=3, vmin=0,
                                vmax=len(unique_types) - 1, rasterized=True)
                axes[1].set_title(f"{labeled_name} (original labels)")
                axes[1].set_xticks([]); axes[1].set_yticks([])

                handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                                      color=cmap(label_to_idx[l]),
                                      label=l, markersize=6) for l in unique_types]
                fig.legend(handles=handles, loc="center right", fontsize=7,
                           bbox_to_anchor=(1.15, 0.5))
                fig.suptitle(f"Joint UMAP — cell types ({viz_col})", fontsize=14)
                fig.savefig(eval_dir / "umap_celltypes.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info("  Saved umap_celltypes.png")

    except ImportError:
        logger.warning("  UMAP not available, skipping UMAP plots")

    # 9b. Spatial map of transferred labels on unlabeled tissue
    if viz_col:
        npz_path = eval_dir / f"transferred_labels_{viz_col}.npz"
        if npz_path.exists():
            pred_labels_viz = np.load(npz_path)["labels"]
            unique_types = np.unique(pred_labels_viz)
            cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_types))
            label_to_idx = {l: i for i, l in enumerate(unique_types)}

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.scatter(
                coords_unlabeled[:, 0], coords_unlabeled[:, 1],
                c=[label_to_idx[l] for l in pred_labels_viz],
                cmap=cmap, s=8, alpha=0.8, vmin=0,
                vmax=len(unique_types) - 1, rasterized=True,
            )
            ax.set_title(f"{unlabeled_name} — transferred cell types ({viz_col})", fontsize=13)
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_aspect("equal")
            handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                                  color=cmap(label_to_idx[l]),
                                  label=l, markersize=6) for l in unique_types]
            ax.legend(handles=handles, fontsize=7, loc="upper right",
                      bbox_to_anchor=(1.35, 1.0))
            fig.savefig(eval_dir / "spatial_labels_transferred.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved spatial_labels_transferred.png")

    # 9c. Original labels spatial map on labeled tissue
    if viz_col and coords_labeled is not None:
        orig_labels_viz = np.array(adata_labeled.obs[viz_col].values, dtype=str)
        unique_types_m = np.unique(orig_labels_viz)
        cmap_m = plt.colormaps.get_cmap("tab20").resampled(len(unique_types_m))
        label_to_idx_m = {l: i for i, l in enumerate(unique_types_m)}

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(
            coords_labeled[:, 0], coords_labeled[:, 1],
            c=[label_to_idx_m[l] for l in orig_labels_viz],
            cmap=cmap_m, s=1, alpha=0.6, vmin=0,
            vmax=len(unique_types_m) - 1, rasterized=True,
        )
        ax.set_title(f"{labeled_name} — original cell types ({viz_col})", fontsize=13)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_aspect("equal")
        handles_m = [plt.Line2D([0], [0], marker="o", linestyle="",
                                color=cmap_m(label_to_idx_m[l]),
                                label=l, markersize=6) for l in unique_types_m]
        ax.legend(handles=handles_m, fontsize=7, loc="upper right",
                  bbox_to_anchor=(1.35, 1.0))
        fig.savefig(eval_dir / "spatial_labels_original.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved spatial_labels_original.png")

    # 9d. Confusion matrix: transferred labels vs Leiden pseudo-labels
    if viz_col and label_cols:
        npz_path = eval_dir / f"transferred_labels_{viz_col}.npz"
        leiden_key = "leiden_1.0"
        if npz_path.exists() and leiden_key in adata_cluster.obs.columns:
            pred_labels_viz = np.load(npz_path)["labels"]
            leiden_labels = np.array(adata_cluster.obs[leiden_key].values, dtype=str)

            from sklearn.metrics import confusion_matrix

            unique_pred = np.unique(pred_labels_viz)
            unique_leiden = np.unique(leiden_labels)
            cm = confusion_matrix(pred_labels_viz, leiden_labels,
                                  labels=np.concatenate([unique_pred,
                                  np.setdiff1d(unique_leiden, unique_pred)]))

            # Only show rows for transferred labels
            cm = cm[:len(unique_pred), :]
            # Normalize rows
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = cm / row_sums

            fig, ax = plt.subplots(1, 1, figsize=(max(8, len(unique_leiden) * 0.5),
                                                    max(6, len(unique_pred) * 0.4)))
            im = ax.imshow(cm_norm, aspect="auto", cmap="Blues")
            ax.set_xticks(range(cm_norm.shape[1]))
            ax.set_xticklabels([f"L{l}" for l in range(cm_norm.shape[1])],
                               rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(unique_pred)))
            ax.set_yticklabels(unique_pred, fontsize=7)
            ax.set_xlabel("Leiden cluster (res=1.0)")
            ax.set_ylabel(f"Transferred cell type ({viz_col})")
            ax.set_title(f"Confusion: transferred labels ({viz_col}) vs Leiden pseudo-labels")
            fig.colorbar(im, ax=ax, shrink=0.6)
            fig.savefig(eval_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved confusion_matrix.png")

    # -----------------------------------------------------------------------
    # 10. Save results
    # -----------------------------------------------------------------------
    results_path = eval_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("All results saved to %s", results_path)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"BRIDGE Evaluation: {args.source} <-> {args.target}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"{'=' * 70}")

    print("\n--- Alignment Metrics ---")
    for k in sorted(results):
        if k.startswith("knn_"):
            print(f"  {k}: {results[k]:.4f}")

    print("\n--- Label Transfer (spatial smoothness) ---")
    for col in label_cols:
        sk = f"label_transfer_{col}_spatial_smoothness"
        rk = f"label_transfer_{col}_random_baseline"
        if sk in results:
            print(f"  {col}: {results[sk]:.4f} (random: {results[rk]:.4f}, "
                  f"ratio: {results[sk]/results[rk]:.1f}x)")

    print("\n--- Pseudo-Label Validation (ARI) ---")
    for k in sorted(results):
        if "ari" in k:
            print(f"  {k}: {results[k]:.4f}")

    print("\n--- Within-Species Control ---")
    for k in sorted(results):
        if "within_species" in k:
            print(f"  {k}: {results[k]:.4f}")

    print("\n--- Baselines ---")
    for k in sorted(results):
        if "baseline_" in k:
            print(f"  {k}: {results[k]:.4f}")

    print(f"\n{'=' * 70}")
    print(f"Outputs saved to: {eval_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
