#!/usr/bin/env python
"""Phase 1 end-to-end verification script.

Creates synthetic spatial transcriptomics data, preprocesses it,
builds a spatial graph, runs the CHISEL encoder, and verifies outputs.
"""

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
    RosettaConfig,
    SpatialGNNConfig,
    load_config,
)


def create_synthetic_spatial_adata(n_cells=200, n_genes=500, seed=42):
    """Create a synthetic spatial transcriptomics AnnData."""
    rng = np.random.default_rng(seed)

    # Simulate count data
    X = rng.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(np.float32)
    X = sp.csr_matrix(X)

    adata = AnnData(X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    # Spatial coordinates on a grid with noise
    grid_size = int(np.ceil(np.sqrt(n_cells)))
    coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(coords) < n_cells:
                coords.append([
                    i * 100 + rng.normal(0, 5),
                    j * 100 + rng.normal(0, 5),
                ])
    adata.obsm["spatial"] = np.array(coords, dtype=np.float32)

    return adata


def main():
    print("=" * 60)
    print("ROSETTA Phase 1 Verification")
    print("=" * 60)

    # 1. Config loading
    print("\n[1] Testing config loading...")
    config = load_config("configs/default.yaml")
    print(f"    Config loaded: preprocessing.n_top_genes={config.preprocessing.n_top_genes}")
    print(f"    CHISEL hidden_dim={config.chisel.spatial_gnn.hidden_dim}")

    # 2. Synthetic data creation
    print("\n[2] Creating synthetic spatial data...")
    adata = create_synthetic_spatial_adata(n_cells=200, n_genes=500)
    print(f"    AnnData: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"    Spatial coords: {adata.obsm['spatial'].shape}")

    # 3. Preprocessing
    print("\n[3] Running preprocessing pipeline...")
    preprocess_config = PreprocessingConfig(
        min_genes=10, min_cells=3, n_top_genes=100, target_sum=1e4
    )
    adata_processed = preprocess_pipeline(adata, preprocess_config)
    print(f"    After preprocessing: {adata_processed.n_obs} cells x {adata_processed.n_vars} genes")
    assert "counts" in adata_processed.layers, "Counts layer missing!"

    # 4. Graph construction
    print("\n[4] Building spatial graph...")
    spatial_coords = torch.tensor(
        adata_processed.obsm["spatial"][:, :2], dtype=torch.float32
    )
    X = adata_processed.X
    if sp.issparse(X):
        X = X.toarray()
    expression = torch.tensor(np.array(X), dtype=torch.float32)

    data = build_spatial_graph(
        spatial_coords=spatial_coords,
        expression=expression,
        k_spatial=6,
        k_expression=3,
        expression_threshold=0.3,
    )
    print(f"    Nodes: {data.x.shape[0]}, Features: {data.x.shape[1]}")
    print(f"    Edges: {data.edge_index.shape[1]}")
    print(f"    Edge attr: {data.edge_attr.shape}")
    print(f"    Pos: {data.pos.shape}")

    # 5. CHISEL Encoder
    print("\n[5] Running CHISEL encoder...")
    n_nodes = data.x.shape[0]
    n_features = data.x.shape[1]

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

    print(f"    z_spot:   {out['z_spot'].shape}")
    print(f"    z_niche:  {out['z_niche'].shape}")
    print(f"    z_region: {out['z_region'].shape}")
    print(f"    link_loss:    {out['link_loss'].item():.4f}")
    print(f"    entropy_loss: {out['entropy_loss'].item():.4f}")

    # 6. Verification checks
    print("\n[6] Running verification checks...")

    checks = []

    # No NaN/Inf
    for key in ["z_spot", "z_niche", "z_region"]:
        has_nan = torch.isnan(out[key]).any().item()
        has_inf = torch.isinf(out[key]).any().item()
        ok = not has_nan and not has_inf
        checks.append((f"{key} no NaN/Inf", ok))

    # Correct dimensions
    checks.append(("z_spot dim matches", out["z_spot"].shape[1] == 64))
    checks.append(("z_niche dim matches", out["z_niche"].shape[2] == 64))
    checks.append(("z_region dim matches", out["z_region"].shape[2] == 64))

    # Hierarchical reduction
    checks.append((
        "niche < spot nodes",
        out["z_niche"].shape[1] < out["z_spot"].shape[0],
    ))
    checks.append((
        "region < niche nodes",
        out["z_region"].shape[1] < out["z_niche"].shape[1],
    ))

    # Losses are finite and non-negative
    checks.append(("link_loss >= 0", out["link_loss"].item() >= 0))
    checks.append(("entropy_loss >= 0", out["entropy_loss"].item() >= 0))

    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"    [{status}] {name}")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        raise SystemExit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
