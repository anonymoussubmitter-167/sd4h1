#!/usr/bin/env python
"""Compare silhouette scores: CHISEL embeddings vs raw expression PCA.

For each dataset with cell type annotations, load the trained CHISEL
checkpoint, extract embeddings, compute PCA on raw expression, and
compare silhouette scores.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

# Suppress noisy warnings
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

from rosetta.chisel.encoders import CHISELEncoder
from rosetta.chisel.graph_construction import build_spatial_graph
from rosetta.chisel.training import CHISELLitModule
from rosetta.utils.config import (
    CHISELConfig,
    MultiScaleConfig,
    PreprocessingConfig,
    SpatialGNNConfig,
    TrainingConfig,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ------------------------------------------------------------------ #
# Per-dataset configs (must match train_chisel.py exactly)
# ------------------------------------------------------------------ #
DATASET_CONFIGS = {
    "merfish_human_liver": {
        "preprocessing": PreprocessingConfig(
            min_genes=5, min_cells=10, n_top_genes=500, target_sum=100.0,
        ),
        "graph_k_spatial": 10,
        "graph_k_expression": 5,
        "skip_hvg": True,
        "skip_normalize": False,
        "label_col": "Cell_Type",
    },
    "merfish_mouse_brain": {
        "preprocessing": PreprocessingConfig(
            min_genes=5, min_cells=10, n_top_genes=500, target_sum=100.0,
        ),
        "graph_k_spatial": 10,
        "graph_k_expression": 5,
        "skip_hvg": True,
        "skip_normalize": False,
        "label_col": "class",  # 12 classes - good granularity
    },
    "stereoseq_zebrafish_brain": {
        "preprocessing": PreprocessingConfig(
            min_genes=100, min_cells=5, n_top_genes=2000, target_sum=1e4,
        ),
        "graph_k_spatial": 8,
        "graph_k_expression": 3,
        "skip_hvg": False,
        "skip_normalize": False,
        "label_col": "layer_annotation",
    },
}

MAX_NODES = 5000
EMBED_DIM = 64
HIDDEN_DIM = 64


def preprocess_adata(adata, cfg):
    """Reproduce exactly the preprocessing from SpatialGraphDataModule."""
    import scanpy as sc

    adata = adata.copy()
    adata.layers["counts"] = adata.X.copy()

    pp = cfg["preprocessing"]

    sc.pp.filter_cells(adata, min_genes=pp.min_genes)
    sc.pp.filter_genes(adata, min_cells=pp.min_cells)

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("All cells or genes filtered out")

    if not cfg.get("skip_normalize", False):
        sc.pp.normalize_total(adata, target_sum=pp.target_sum)
        sc.pp.log1p(adata)

    if not cfg.get("skip_hvg", False) and adata.n_vars > pp.n_top_genes:
        try:
            adata_hvg = adata.copy()
            adata_hvg.X = adata_hvg.layers["counts"].copy()
            sc.pp.highly_variable_genes(
                adata_hvg, n_top_genes=pp.n_top_genes, flavor="seurat_v3",
            )
            adata.var["highly_variable"] = adata_hvg.var["highly_variable"]
            adata = adata[:, adata.var["highly_variable"]].copy()
        except Exception:
            X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
            var = np.var(X, axis=0)
            top_idx = np.argsort(var)[-pp.n_top_genes:]
            adata = adata[:, top_idx].copy()

    return adata


def evaluate_dataset(dataset_name, cfg):
    """Evaluate one dataset: CHISEL vs PCA silhouette scores."""
    h5ad_path = PROCESSED_DIR / f"{dataset_name}.h5ad"
    ckpt_path = OUTPUTS_DIR / dataset_name / "checkpoints" / "last.ckpt"
    label_col = cfg["label_col"]

    if not h5ad_path.exists():
        return None, f"h5ad not found: {h5ad_path}"
    if not ckpt_path.exists():
        return None, f"checkpoint not found: {ckpt_path}"

    # 1. Load and preprocess
    print(f"  Loading {h5ad_path.name}...")
    adata = ad.read_h5ad(h5ad_path)
    adata = preprocess_adata(adata, cfg)

    # 2. Subsample if needed (same seed as training)
    n_original = adata.n_obs
    if adata.n_obs > MAX_NODES:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, size=MAX_NODES, replace=False)
        idx.sort()
        adata = adata[idx].copy()
        print(f"  Subsampled {n_original} -> {adata.n_obs} nodes")

    n_nodes = adata.n_obs
    print(f"  Nodes: {n_nodes}, Genes: {adata.n_vars}")

    # 3. Get labels
    if label_col not in adata.obs.columns:
        return None, f"Label column '{label_col}' not found after subsample"

    labels_raw = adata.obs[label_col].values
    # Remove NaN labels
    valid_mask = ~(labels_raw == None)  # noqa
    if hasattr(labels_raw, 'isna'):
        valid_mask = ~adata.obs[label_col].isna().values
    else:
        valid_mask = np.array([l is not None and str(l) != 'nan' for l in labels_raw])

    le = LabelEncoder()
    labels_all = le.fit_transform(labels_raw[valid_mask].astype(str))
    n_classes = len(le.classes_)
    print(f"  Label column: '{label_col}', {n_classes} classes, {valid_mask.sum()} valid labels")

    if n_classes < 2:
        return None, "Need at least 2 classes for silhouette score"

    # 4. Build graph (same as training)
    spatial_coords = torch.tensor(
        np.array(adata.obsm["spatial"][:, :2], dtype=np.float32)
    )
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    expression = torch.tensor(np.array(X, dtype=np.float32))

    data = build_spatial_graph(
        spatial_coords=spatial_coords,
        expression=expression,
        k_spatial=cfg.get("graph_k_spatial", 6),
        k_expression=cfg.get("graph_k_expression", 3),
        expression_threshold=0.3,
    )

    input_dim = data.x.shape[1]

    # 5. Build model with same config as training
    chisel_config = CHISELConfig(
        spatial_gnn=SpatialGNNConfig(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        ),
        multi_scale=MultiScaleConfig(
            pool_ratios=[0.25, 0.25],
            embed_dim=EMBED_DIM,
            num_gnn_layers_per_level=2,
        ),
    )

    training_config = TrainingConfig(max_nodes=MAX_NODES)

    model = CHISELLitModule(
        chisel_config=chisel_config,
        training_config=training_config,
        input_dim=input_dim,
        max_nodes=n_nodes,
    )

    # 6. Load checkpoint
    print(f"  Loading checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 7. Get CHISEL embeddings
    print(f"  Computing CHISEL embeddings...")
    with torch.no_grad():
        out = model(data)
        z_chisel = out["z_spot"].cpu().numpy()  # (N, 64)

    # 8. Get PCA embeddings from raw expression
    print(f"  Computing PCA embeddings...")
    X_np = expression.numpy()
    n_components = min(EMBED_DIM, X_np.shape[0] - 1, X_np.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    z_pca = pca.fit_transform(X_np)

    # 9. Compute silhouette scores (on valid labels only)
    z_chisel_valid = z_chisel[valid_mask]
    z_pca_valid = z_pca[valid_mask]

    sil_chisel = silhouette_score(z_chisel_valid, labels_all, metric="euclidean")
    sil_pca = silhouette_score(z_pca_valid, labels_all, metric="euclidean")

    pca_var_explained = pca.explained_variance_ratio_.sum()

    return {
        "dataset": dataset_name,
        "label_col": label_col,
        "n_nodes": n_nodes,
        "n_genes": input_dim,
        "n_classes": n_classes,
        "n_valid_labels": int(valid_mask.sum()),
        "sil_chisel": sil_chisel,
        "sil_pca": sil_pca,
        "pca_var_explained": pca_var_explained,
        "chisel_embed_dim": z_chisel.shape[1],
        "pca_n_components": n_components,
    }, None


def main():
    print("=" * 80)
    print("CHISEL vs PCA Silhouette Score Comparison")
    print("=" * 80)
    print()

    results = []
    for dataset_name, cfg in DATASET_CONFIGS.items():
        print(f"\n--- {dataset_name} ---")
        result, error = evaluate_dataset(dataset_name, cfg)
        if error:
            print(f"  SKIPPED: {error}")
        else:
            results.append(result)
            print(f"  Silhouette (CHISEL): {result['sil_chisel']:.4f}")
            print(f"  Silhouette (PCA):    {result['sil_pca']:.4f}")

    if not results:
        print("\nNo datasets had valid results.")
        return

    # Print comparison table
    print("\n")
    print("=" * 100)
    print("COMPARISON TABLE: Silhouette Scores (CHISEL Embeddings vs Raw Expression PCA)")
    print("=" * 100)

    header = (
        f"{'Dataset':<30s} {'Label Column':<20s} {'Nodes':>6s} {'Genes':>6s} "
        f"{'Classes':>7s} {'CHISEL':>8s} {'PCA':>8s} {'Delta':>8s} {'Winner':>8s}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        delta = r["sil_chisel"] - r["sil_pca"]
        winner = "CHISEL" if delta > 0 else "PCA"
        row = (
            f"{r['dataset']:<30s} {r['label_col']:<20s} {r['n_nodes']:>6d} {r['n_genes']:>6d} "
            f"{r['n_classes']:>7d} {r['sil_chisel']:>8.4f} {r['sil_pca']:>8.4f} "
            f"{delta:>+8.4f} {winner:>8s}"
        )
        print(row)

    print("-" * 100)

    # Summary statistics
    chisel_wins = sum(1 for r in results if r["sil_chisel"] > r["sil_pca"])
    avg_chisel = np.mean([r["sil_chisel"] for r in results])
    avg_pca = np.mean([r["sil_pca"] for r in results])
    avg_delta = avg_chisel - avg_pca

    print(f"\n{'Summary':>30s}")
    print(f"{'CHISEL wins:':<30s} {chisel_wins}/{len(results)} datasets")
    print(f"{'Avg silhouette (CHISEL):':<30s} {avg_chisel:.4f}")
    print(f"{'Avg silhouette (PCA):':<30s} {avg_pca:.4f}")
    print(f"{'Avg delta (CHISEL - PCA):':<30s} {avg_delta:+.4f}")

    print("\n" + "=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)
    for r in results:
        print(f"\n  {r['dataset']}:")
        print(f"    Label column:        {r['label_col']}")
        print(f"    Nodes / Genes:       {r['n_nodes']} / {r['n_genes']}")
        print(f"    Classes:             {r['n_classes']}")
        print(f"    Valid labels:        {r['n_valid_labels']}")
        print(f"    CHISEL embed dim:    {r['chisel_embed_dim']}")
        print(f"    PCA components:      {r['pca_n_components']}")
        print(f"    PCA var explained:   {r['pca_var_explained']:.4f}")
        print(f"    Silhouette (CHISEL): {r['sil_chisel']:.4f}")
        print(f"    Silhouette (PCA):    {r['sil_pca']:.4f}")
        delta = r["sil_chisel"] - r["sil_pca"]
        pct = (delta / abs(r["sil_pca"])) * 100 if r["sil_pca"] != 0 else float("inf")
        print(f"    Delta:               {delta:+.4f} ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
