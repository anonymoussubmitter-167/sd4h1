#!/usr/bin/env python
"""Validate real datasets through the full ROSETTA pipeline.

For each downloaded dataset:
1. Loads via rosetta.data.loaders
2. Runs preprocess_pipeline()
3. Builds spatial graph via build_spatial_graph()
4. Runs CHISEL encoder forward pass
5. Prints summary and checks for NaN/Inf
6. Saves a spatial scatter plot

Usage:
    python scripts/validate_real_data.py                     # Validate all
    python scripts/validate_real_data.py --dataset brain      # Validate one
    python scripts/validate_real_data.py --skip-encoder       # Skip CHISEL pass
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from anndata import AnnData

from rosetta.chisel.encoders import CHISELEncoder
from rosetta.chisel.graph_construction import build_spatial_graph
from rosetta.data.preprocessing import preprocess_pipeline
from rosetta.utils.config import (
    CHISELConfig,
    MultiScaleConfig,
    PreprocessingConfig,
    SpatialGNNConfig,
)
from rosetta.utils.visualization import plot_spatial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Dataset-specific preprocessing configs
# ---------------------------------------------------------------------------

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
        "skip_hvg": True,   # Limited gene panel
        "skip_normalize": True,  # Already processed
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


# ---------------------------------------------------------------------------
# Preprocessing with dataset-specific overrides
# ---------------------------------------------------------------------------

def preprocess_dataset(adata: AnnData, config: dict) -> AnnData:
    """Preprocess a dataset with dataset-specific logic.

    Handles cases where standard preprocessing doesn't apply
    (e.g., MERFISH skips HVG, Allen ISH skips normalization).
    """
    import scanpy as sc

    adata = adata.copy()

    # Store raw counts
    adata.layers["counts"] = adata.X.copy()

    pp = config["preprocessing"]

    # Filtering
    sc.pp.filter_cells(adata, min_genes=pp.min_genes)
    sc.pp.filter_genes(adata, min_cells=pp.min_cells)

    if adata.n_obs == 0:
        raise ValueError("All cells filtered out during preprocessing")
    if adata.n_vars == 0:
        raise ValueError("All genes filtered out during preprocessing")

    # Normalization
    if not config.get("skip_normalize", False):
        sc.pp.normalize_total(adata, target_sum=pp.target_sum)
        sc.pp.log1p(adata)

    # HVG selection
    if not config.get("skip_hvg", False) and adata.n_vars > pp.n_top_genes:
        try:
            # seurat_v3 needs raw counts
            adata_hvg = adata.copy()
            adata_hvg.X = adata_hvg.layers["counts"].copy()
            sc.pp.highly_variable_genes(
                adata_hvg, n_top_genes=pp.n_top_genes, flavor="seurat_v3",
            )
            adata.var["highly_variable"] = adata_hvg.var["highly_variable"]
            adata = adata[:, adata.var["highly_variable"]].copy()
        except Exception as e:
            logger.warning("HVG selection failed (%s), using top %d by variance",
                          e, pp.n_top_genes)
            # Fallback: select by variance
            X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
            var = np.var(X, axis=0)
            top_idx = np.argsort(var)[-pp.n_top_genes:]
            adata = adata[:, top_idx].copy()

    return adata


# ---------------------------------------------------------------------------
# Subsample large datasets for validation
# ---------------------------------------------------------------------------

def subsample_if_needed(adata: AnnData, max_cells: int = 5000) -> AnnData:
    """Subsample to max_cells for validation (large datasets would OOM in CHISEL)."""
    if adata.n_obs <= max_cells:
        return adata

    logger.info("Subsampling %d -> %d cells for validation", adata.n_obs, max_cells)
    rng = np.random.default_rng(42)
    idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
    idx.sort()
    return adata[idx].copy()


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------

def validate_dataset(
    name: str,
    skip_encoder: bool = False,
    max_cells: int = 5000,
) -> dict:
    """Run full validation pipeline on one dataset.

    Returns a dict with validation results.
    """
    h5ad_path = PROCESSED_DIR / f"{name}.h5ad"
    if not h5ad_path.exists():
        return {"status": "SKIP", "reason": f"File not found: {h5ad_path}"}

    config = DATASET_CONFIGS.get(name)
    if config is None:
        return {"status": "SKIP", "reason": f"No config for {name}"}

    result = {
        "name": name,
        "species": config["species"],
        "platform": config["platform"],
        "status": "OK",
        "checks": [],
    }

    try:
        t0 = time.time()

        # 1. Load
        logger.info("[%s] Loading h5ad...", name)
        import anndata as ad
        adata = ad.read_h5ad(h5ad_path)
        result["n_cells_raw"] = adata.n_obs
        result["n_genes_raw"] = adata.n_vars

        # Validate spatial coords exist
        if "spatial" not in adata.obsm:
            raise ValueError("No obsm['spatial'] found")
        spatial = adata.obsm["spatial"]
        assert spatial.shape[0] == adata.n_obs
        assert spatial.shape[1] >= 2
        result["checks"].append(("spatial coords present", True))

        # 2. Preprocess
        logger.info("[%s] Preprocessing...", name)
        adata = preprocess_dataset(adata, config)
        result["n_cells_processed"] = adata.n_obs
        result["n_genes_processed"] = adata.n_vars
        result["checks"].append(("preprocessing", True))

        # 3. Subsample for encoder validation
        adata_sub = subsample_if_needed(adata, max_cells=max_cells)

        # 4. Build spatial graph
        logger.info("[%s] Building spatial graph (%d cells)...", name, adata_sub.n_obs)
        spatial_coords = torch.tensor(
            np.array(adata_sub.obsm["spatial"][:, :2], dtype=np.float32)
        )
        X = adata_sub.X
        if sp.issparse(X):
            X = X.toarray()
        expression = torch.tensor(np.array(X, dtype=np.float32))

        data = build_spatial_graph(
            spatial_coords=spatial_coords,
            expression=expression,
            k_spatial=config["graph_k_spatial"],
            k_expression=config["graph_k_expression"],
            expression_threshold=0.3,
        )
        result["n_edges"] = data.edge_index.shape[1]
        result["checks"].append(("graph construction", True))

        # Check edge_attr shape
        assert data.edge_attr.shape[1] == 5
        assert not torch.isnan(data.edge_attr).any()
        result["checks"].append(("edge attrs valid", True))

        # 5. CHISEL encoder forward pass
        if not skip_encoder:
            logger.info("[%s] Running CHISEL encoder...", name)
            n_features = data.x.shape[1]
            n_nodes = data.x.shape[0]

            chisel_config = CHISELConfig(
                spatial_gnn=SpatialGNNConfig(
                    input_dim=n_features,
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

            encoder = CHISELEncoder(chisel_config, max_nodes=n_nodes)
            encoder.eval()

            with torch.no_grad():
                out = encoder(data.x, data.edge_index, data.edge_attr)

            result["z_spot_shape"] = tuple(out["z_spot"].shape)
            result["z_niche_shape"] = tuple(out["z_niche"].shape)
            result["z_region_shape"] = tuple(out["z_region"].shape)

            # Check no NaN/Inf
            for key in ["z_spot", "z_niche", "z_region"]:
                has_nan = torch.isnan(out[key]).any().item()
                has_inf = torch.isinf(out[key]).any().item()
                ok = not has_nan and not has_inf
                result["checks"].append((f"{key} no NaN/Inf", ok))
                if not ok:
                    result["status"] = "WARN"

            # Hierarchical reduction
            result["checks"].append((
                "niche < spot",
                out["z_niche"].shape[1] < out["z_spot"].shape[0],
            ))
            result["checks"].append((
                "region < niche",
                out["z_region"].shape[1] < out["z_niche"].shape[1],
            ))

            result["link_loss"] = out["link_loss"].item()
            result["entropy_loss"] = out["entropy_loss"].item()
            result["checks"].append(("encoder forward pass", True))

        # 6. Save spatial plot
        logger.info("[%s] Saving spatial plot...", name)
        plot_path = PROCESSED_DIR / f"{name}_spatial.png"
        coords = np.array(adata.obsm["spatial"][:, :2])

        # Color by total counts or first gene
        X_full = adata.X
        if sp.issparse(X_full):
            X_full = X_full.toarray()
        total_counts = np.array(X_full.sum(axis=1)).ravel()

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: colored by total counts
        plot_spatial(coords, total_counts, title=f"{name}\n(total counts)", ax=axes[0])

        # Plot 2: colored by first gene expression
        gene_name = adata.var_names[0]
        gene_expr = np.array(X_full[:, 0]).ravel()
        plot_spatial(coords, gene_expr, title=f"{name}\n({gene_name})", ax=axes[1])

        fig.tight_layout()
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        result["plot_path"] = str(plot_path)
        result["checks"].append(("spatial plot saved", True))

        result["time_s"] = time.time() - t0

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = str(e)
        logger.error("[%s] FAILED: %s", name, e, exc_info=True)

    return result


# ---------------------------------------------------------------------------
# CLI and summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    """Print a summary table of all validation results."""
    print(f"\n{'='*90}")
    print("VALIDATION SUMMARY")
    print(f"{'='*90}")

    # Header
    print(f"{'Dataset':<30s} {'Status':>6s} {'Species':<10s} {'Platform':<10s} "
          f"{'Cells':>8s} {'Genes':>7s} {'Edges':>9s}")
    print("-" * 90)

    for r in results:
        name = r.get("name", "?")
        status = r.get("status", "?")
        species = r.get("species", "?")
        platform = r.get("platform", "?")
        n_cells = r.get("n_cells_processed", r.get("n_cells_raw", "?"))
        n_genes = r.get("n_genes_processed", r.get("n_genes_raw", "?"))
        n_edges = r.get("n_edges", "?")

        cells_str = str(n_cells) if isinstance(n_cells, int) else "?"
        genes_str = str(n_genes) if isinstance(n_genes, int) else "?"
        edges_str = str(n_edges) if isinstance(n_edges, int) else "?"

        print(f"  {name:<28s} [{status:>4s}] {species:<10s} {platform:<10s} "
              f"{cells_str:>8s} {genes_str:>7s} {edges_str:>9s}")

        if status == "FAIL":
            print(f"    Error: {r.get('error', '?')}")
        elif status == "SKIP":
            print(f"    Reason: {r.get('reason', '?')}")

    print("-" * 90)

    # Encoder shapes
    print(f"\n{'Embedding Shapes':}")
    for r in results:
        if "z_spot_shape" in r:
            print(f"  {r['name']:<28s}  z_spot={r['z_spot_shape']}  "
                  f"z_niche={r['z_niche_shape']}  z_region={r['z_region_shape']}")

    # Checks
    print(f"\n{'Detailed Checks':}")
    for r in results:
        name = r.get("name", "?")
        checks = r.get("checks", [])
        if checks:
            all_ok = all(ok for _, ok in checks)
            check_str = "ALL PASS" if all_ok else "SOME FAIL"
            print(f"  {name}: {check_str}")
            for check_name, ok in checks:
                marker = "PASS" if ok else "FAIL"
                print(f"    [{marker}] {check_name}")

    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    print(f"\n{n_ok} passed, {n_fail} failed, {n_skip} skipped "
          f"out of {len(results)} datasets")


def main():
    parser = argparse.ArgumentParser(
        description="Validate real datasets through the ROSETTA pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        help="Dataset name(s) to validate. If omitted, validates all.",
    )
    parser.add_argument(
        "--skip-encoder",
        action="store_true",
        help="Skip CHISEL encoder forward pass (faster validation).",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=5000,
        help="Max cells for encoder validation (default: 5000). "
             "Large datasets are subsampled to this.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets for validation:")
        for name in sorted(DATASET_CONFIGS.keys()):
            h5ad = PROCESSED_DIR / f"{name}.h5ad"
            exists = "EXISTS" if h5ad.exists() else "MISSING"
            print(f"  {name:<30s} [{exists}]")
        return

    targets = args.dataset if args.dataset else sorted(DATASET_CONFIGS.keys())

    print("=" * 70)
    print("ROSETTA Real Data Validation")
    print("=" * 70)
    print(f"Datasets: {len(targets)}")
    print(f"Skip encoder: {args.skip_encoder}")
    print(f"Max cells for encoder: {args.max_cells}")
    print()

    results = []
    for name in targets:
        print(f"\n--- Validating: {name} ---")
        r = validate_dataset(
            name,
            skip_encoder=args.skip_encoder,
            max_cells=args.max_cells,
        )
        results.append(r)

    print_summary(results)

    # Exit with error if any failed
    if any(r["status"] == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
