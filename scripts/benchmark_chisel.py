#!/usr/bin/env python
"""Benchmark CHISEL embeddings vs PCA, scVI, and autoencoder baselines.

Compares embedding quality on labeled mouse brain data using:
- ARI (Adjusted Rand Index) via Leiden clustering
- NMI (Normalized Mutual Information)
- Silhouette Score
- Moran's I (spatial autocorrelation)

Usage:
    python scripts/benchmark_chisel.py --dataset merfish_mouse_brain
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from rosetta.chisel.training import CHISELLitModule, SpatialGraphDataModule
from rosetta.utils.config import (
    CHISELConfig,
    MultiScaleConfig,
    PreprocessingConfig,
    SpatialGNNConfig,
    TrainingConfig,
)
from rosetta.utils.metrics import morans_i

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASET_CONFIGS = {
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
    "stereoseq_zebrafish_brain": {
        "species": "zebrafish",
        "platform": "stereoseq",
        "preprocessing": PreprocessingConfig(
            min_genes=5, min_cells=10, n_top_genes=2000, target_sum=1e4,
        ),
        "graph_k_spatial": 10,
        "graph_k_expression": 5,
        "skip_hvg": False,
        "skip_normalize": False,
    },
}

KNOWN_LABEL_COLUMNS = [
    "class", "subclass", "neurotransmitter", "supertype", "cluster",
    "Cell_Type", "cell_type", "celltype", "cluster_alias",
]


def get_labels(adata, max_nodes=5000):
    """Get available label columns from adata."""
    labels = {}
    for col in KNOWN_LABEL_COLUMNS:
        if col in adata.obs.columns:
            vals = adata.obs[col].values[:max_nodes]
            # Skip if too few unique values
            unique = np.unique([str(v) for v in vals])
            if 2 <= len(unique) <= 500:
                labels[col] = np.array([str(v) for v in vals])
    return labels


def compute_clustering_metrics(Z, labels_dict, n_neighbors=15, resolutions=[0.3, 0.5, 1.0, 2.0]):
    """Compute ARI, NMI, silhouette for embeddings via Leiden clustering."""
    results = {}

    # Build AnnData for scanpy Leiden
    adata_emb = ad.AnnData(X=Z.astype(np.float32))
    sc.pp.neighbors(adata_emb, n_neighbors=n_neighbors, use_rep="X")

    for col, true_labels in labels_dict.items():
        best_ari, best_nmi, best_res = -1, -1, -1
        for res in resolutions:
            sc.tl.leiden(adata_emb, resolution=res, key_added="leiden")
            pred = adata_emb.obs["leiden"].values
            ari = adjusted_rand_score(true_labels, pred)
            nmi = normalized_mutual_info_score(true_labels, pred)
            if ari > best_ari:
                best_ari, best_nmi, best_res = ari, nmi, res

        results[f"{col}_ari"] = best_ari
        results[f"{col}_nmi"] = best_nmi
        results[f"{col}_best_resolution"] = best_res

    # Silhouette (use first label column, sample if large)
    first_col = list(labels_dict.keys())[0]
    first_labels = labels_dict[first_col]
    n_sample = min(5000, len(first_labels))
    if n_sample < len(first_labels):
        idx = np.random.default_rng(42).choice(len(first_labels), n_sample, replace=False)
        Z_sample, labels_sample = Z[idx], first_labels[idx]
    else:
        Z_sample, labels_sample = Z, first_labels
    sil = silhouette_score(Z_sample, labels_sample)
    results[f"{first_col}_silhouette"] = sil

    return results


def get_chisel_embeddings(dataset, max_nodes=5000):
    """Load CHISEL model and extract embeddings."""
    dataset_config = DATASET_CONFIGS[dataset]
    training_config = TrainingConfig(max_nodes=max_nodes)
    h5ad_path = PROCESSED_DIR / f"{dataset}.h5ad"

    data_module = SpatialGraphDataModule(
        h5ad_path=h5ad_path,
        dataset_config=dataset_config,
        training_config=training_config,
        max_nodes=max_nodes,
    )
    data_module.setup()

    ckpt_path = PROJECT_ROOT / "outputs" / dataset / "checkpoints" / "last.ckpt"
    if not ckpt_path.exists():
        logger.warning("CHISEL checkpoint not found: %s", ckpt_path)
        return None, None, None

    chisel_config = CHISELConfig(
        spatial_gnn=SpatialGNNConfig(
            input_dim=data_module.input_dim,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        ),
        multi_scale=MultiScaleConfig(
            pool_ratios=[0.25, 0.25],
            embed_dim=64,
            num_gnn_layers_per_level=2,
        ),
    )

    model = CHISELLitModule.load_from_checkpoint(
        ckpt_path,
        chisel_config=chisel_config,
        training_config=training_config,
        input_dim=data_module.input_dim,
        max_nodes=data_module.n_nodes,
    )
    model.eval()
    model = model.cpu()

    graph_data = data_module._graph_data
    with torch.no_grad():
        out = model.encoder(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        z_chisel = out["z_spot"].cpu().numpy()

    spatial_coords = graph_data.pos.cpu().numpy()
    expr_matrix = graph_data.x.cpu().numpy()

    return z_chisel, spatial_coords, expr_matrix


def get_pca_embeddings(expr_matrix, n_components=64):
    """PCA baseline."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, expr_matrix.shape[1] - 1))
    return pca.fit_transform(expr_matrix)


def get_scvi_embeddings(dataset, max_nodes=5000, max_epochs=100):
    """scVI baseline - VAE-based encoder."""
    import scvi

    h5ad_path = PROCESSED_DIR / f"{dataset}.h5ad"
    adata = ad.read_h5ad(h5ad_path)

    # Subsample to match CHISEL
    if adata.n_obs > max_nodes:
        sc.pp.subsample(adata, n_obs=max_nodes, random_state=42)

    # scVI needs raw counts (non-negative integers)
    import scipy.sparse as sps
    X = adata.X
    if sps.issparse(X):
        X = np.array(X.todense())
    if X.min() < 0:
        X = np.expm1(np.abs(X))
    X = np.round(np.maximum(X, 0)).astype(np.float32)
    adata_vi = ad.AnnData(X=X)
    adata_vi.var_names = [str(v) for v in adata.var_names[:X.shape[1]]]

    scvi.model.SCVI.setup_anndata(adata_vi)
    vae = scvi.model.SCVI(adata_vi, n_latent=64, n_layers=2, gene_likelihood="nb")
    vae.train(max_epochs=max_epochs, early_stopping=True, batch_size=256, train_size=0.9,
              accelerator="cpu", devices=1)

    return vae.get_latent_representation()


def get_autoencoder_embeddings(expr_matrix, latent_dim=64, epochs=100):
    """Simple autoencoder baseline."""
    import torch.nn as nn

    X = torch.tensor(expr_matrix, dtype=torch.float32)
    input_dim = X.shape[1]

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    ae = AE()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    ae.train()
    for epoch in range(epochs):
        for (batch,) in loader:
            recon, _ = ae(batch)
            loss = nn.functional.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ae.eval()
    with torch.no_grad():
        _, z = ae(X)
    return z.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-scvi", action="store_true", help="Skip scVI baseline (slow on CPU)")
    parser.add_argument("--scvi-epochs", type=int, default=100, help="Max epochs for scVI training")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.dataset / "chisel_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"dataset": args.dataset}

    # Load data for labels
    adata = ad.read_h5ad(PROCESSED_DIR / f"{args.dataset}.h5ad")
    labels = get_labels(adata, args.max_nodes)
    logger.info("Label columns: %s", {k: len(np.unique(v)) for k, v in labels.items()})

    if not labels:
        logger.error("No label columns found in dataset")
        sys.exit(1)

    # 1. CHISEL embeddings
    logger.info("=== CHISEL ===")
    z_chisel, spatial_coords, expr_matrix = get_chisel_embeddings(args.dataset, args.max_nodes)
    if z_chisel is not None:
        # Trim labels to match CHISEL node count
        labels_trimmed = {k: v[:z_chisel.shape[0]] for k, v in labels.items()}
        chisel_metrics = compute_clustering_metrics(z_chisel, labels_trimmed)
        mi_chisel = morans_i(z_chisel, spatial_coords, k=6)
        chisel_metrics["morans_i"] = mi_chisel
        for k, v in sorted(chisel_metrics.items()):
            logger.info("  CHISEL %s: %.4f", k, v)
            results[f"chisel_{k}"] = v

    # 2. PCA baseline
    logger.info("=== PCA ===")
    z_pca = get_pca_embeddings(expr_matrix, n_components=64)
    pca_metrics = compute_clustering_metrics(z_pca, labels_trimmed)
    mi_pca = morans_i(z_pca, spatial_coords, k=6)
    pca_metrics["morans_i"] = mi_pca
    for k, v in sorted(pca_metrics.items()):
        logger.info("  PCA %s: %.4f", k, v)
        results[f"pca_{k}"] = v

    # 3. scVI baseline
    if args.skip_scvi:
        logger.info("=== scVI (skipped) ===")
    else:
        logger.info("=== scVI ===")
        try:
            z_scvi = get_scvi_embeddings(args.dataset, args.max_nodes, max_epochs=args.scvi_epochs)
            n_scvi = min(z_scvi.shape[0], z_chisel.shape[0])
            labels_scvi = {k: v[:n_scvi] for k, v in labels_trimmed.items()}
            scvi_metrics = compute_clustering_metrics(z_scvi[:n_scvi], labels_scvi)
            mi_scvi = morans_i(z_scvi[:n_scvi], spatial_coords[:n_scvi], k=6)
            scvi_metrics["morans_i"] = mi_scvi
            for k, v in sorted(scvi_metrics.items()):
                logger.info("  scVI %s: %.4f", k, v)
                results[f"scvi_{k}"] = v
        except Exception as e:
            logger.warning("scVI failed: %s", e)
            import traceback; traceback.print_exc()

    # 4. Autoencoder baseline
    logger.info("=== Autoencoder ===")
    z_ae = get_autoencoder_embeddings(expr_matrix, latent_dim=64, epochs=100)
    ae_metrics = compute_clustering_metrics(z_ae, labels_trimmed)
    mi_ae = morans_i(z_ae, spatial_coords, k=6)
    ae_metrics["morans_i"] = mi_ae
    for k, v in sorted(ae_metrics.items()):
        logger.info("  AE %s: %.4f", k, v)
        results[f"autoencoder_{k}"] = v

    # 5. Random baseline
    logger.info("=== Random ===")
    z_random = np.random.default_rng(42).standard_normal((z_chisel.shape[0], 64)).astype(np.float32)
    random_metrics = compute_clustering_metrics(z_random, labels_trimmed)
    mi_random = morans_i(z_random, spatial_coords, k=6)
    random_metrics["morans_i"] = mi_random
    for k, v in sorted(random_metrics.items()):
        logger.info("  Random %s: %.4f", k, v)
        results[f"random_{k}"] = v

    # Save
    out_file = output_dir / "chisel_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_file)

    # Print summary table
    methods = ["chisel", "pca", "scvi", "autoencoder", "random"]
    first_label = list(labels_trimmed.keys())[0]
    print(f"\n{'Method':12s} | {'ARI':>6s} | {'NMI':>6s} | {'Silhouette':>10s} | {'Moran I':>8s}")
    print("-" * 55)
    for m in methods:
        ari = results.get(f"{m}_{first_label}_ari", "  --  ")
        nmi = results.get(f"{m}_{first_label}_nmi", "  --  ")
        sil = results.get(f"{m}_{first_label}_silhouette", "  --  ")
        mi = results.get(f"{m}_morans_i", "  --  ")
        ari_s = f"{ari:.4f}" if isinstance(ari, float) else ari
        nmi_s = f"{nmi:.4f}" if isinstance(nmi, float) else nmi
        sil_s = f"{sil:.4f}" if isinstance(sil, float) else sil
        mi_s = f"{mi:.4f}" if isinstance(mi, float) else mi
        print(f"{m:12s} | {ari_s:>6s} | {nmi_s:>6s} | {sil_s:>10s} | {mi_s:>8s}")


if __name__ == "__main__":
    main()
