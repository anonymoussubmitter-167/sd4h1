#!/usr/bin/env python3
"""Compute baseline Leiden ARI for brain pair.

This is a targeted script to compute the missing baseline Leiden ARI values
that were skipped in the main evaluation due to cell count mismatch (4910 raw
vs 4906 preprocessed). It aligns cell indices to handle the mismatch.
"""

import json
import numpy as np
import scanpy as sc
import anndata as ad
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rosetta.utils.metrics import label_transfer_with_confidence
from rosetta.bridge.shared_space import build_shared_gene_space as _build_shared
from rosetta.bridge.training import load_ortholog_mapping

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed"
ORTHO = BASE / "data" / "orthologs"



def main():
    # Load preprocessed data (same as evaluate_bridge.py)
    print("Loading preprocessed data...")
    source_h5ad = DATA / "visium_human_brain.h5ad"
    target_h5ad = DATA / "merfish_mouse_brain.h5ad"

    adata_src_raw = ad.read_h5ad(source_h5ad)
    adata_tgt_raw = ad.read_h5ad(target_h5ad)
    print(f"  Source raw: {adata_src_raw.n_obs} cells")
    print(f"  Target raw: {adata_tgt_raw.n_obs} cells")

    # Load orthologs
    ortholog_map = load_ortholog_mapping("human", "mouse")
    print(f"  Orthologs loaded: {type(ortholog_map).__name__}")

    # Build shared gene space (handles Ensembl ID -> gene symbol mapping)
    X_src_shared, X_tgt_shared, shared_genes = _build_shared(
        adata_src_raw, adata_tgt_raw, ortholog_map,
    )
    print(f"  Shared genes: {len(shared_genes)}")

    # Target has labels -> transfer to source (mouse -> human)
    # adata_unlabeled = human (source), adata_labeled = mouse (target)
    adata_labeled_raw = adata_tgt_raw
    adata_unlabeled_raw = adata_src_raw
    X_labeled_shared = X_tgt_shared
    X_unlabeled_shared = X_src_shared

    # Preprocess unlabeled data for Leiden (same as evaluate_bridge.py)
    print("Computing Leiden clustering on preprocessed human data...")
    adata_cluster = adata_unlabeled_raw.copy()
    sc.pp.filter_cells(adata_cluster, min_genes=1)
    sc.pp.filter_genes(adata_cluster, min_cells=1)
    sc.pp.normalize_total(adata_cluster, target_sum=1e4)
    sc.pp.log1p(adata_cluster)
    sc.pp.highly_variable_genes(adata_cluster, n_top_genes=min(2000, adata_cluster.n_vars))
    adata_cluster = adata_cluster[:, adata_cluster.var.highly_variable].copy()
    sc.pp.pca(adata_cluster, n_comps=min(50, adata_cluster.n_vars - 1))
    sc.pp.neighbors(adata_cluster, n_neighbors=15)

    resolutions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    for res in resolutions:
        sc.tl.leiden(adata_cluster, resolution=res, key_added=f"leiden_{res}", flavor="igraph", directed=False, n_iterations=2)
        n_clusters = len(adata_cluster.obs[f"leiden_{res}"].unique())
        print(f"  Leiden res={res}: {n_clusters} clusters")

    print(f"  Preprocessed cells: {adata_cluster.n_obs}")
    print(f"  Raw cells: {adata_unlabeled_raw.n_obs}")

    # Build cell index alignment
    raw_ids = list(adata_unlabeled_raw.obs_names)
    preproc_ids = list(adata_cluster.obs_names)
    idx_map = {name: i for i, name in enumerate(raw_ids)}
    keep = [idx_map[n] for n in preproc_ids if n in idx_map]
    print(f"  Aligned {len(raw_ids)} raw -> {len(keep)} preprocessed cells")

    label_cols = ["class", "subclass", "neurotransmitter", "supertype", "cluster"]
    results = {}

    # --- Expression-only baseline ---
    print("\n=== Expression-only baseline ===")
    for col in label_cols:
        if col not in adata_labeled_raw.obs.columns:
            continue
        raw_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)
        pred, _ = label_transfer_with_confidence(
            X_labeled_shared, X_unlabeled_shared, raw_labels, k=10,
        )
        pred_aligned = pred[keep]
        for res in resolutions:
            leiden = np.array(adata_cluster.obs[f"leiden_{res}"].values, dtype=str)
            ari = adjusted_rand_score(pred_aligned, leiden)
            results[f"baseline_expression_{col}_leiden{res}_ari"] = ari
            print(f"  Expression {col} vs Leiden(res={res}): ARI={ari:.4f}")

    # --- Harmony baseline ---
    print("\n=== Harmony baseline ===")
    try:
        import harmonypy as hm
        import pandas as pd

        X_combined = np.vstack([X_src_shared, X_tgt_shared])
        n_src = X_src_shared.shape[0]
        meta = pd.DataFrame({"batch": ["source"] * n_src + ["target"] * X_tgt_shared.shape[0]})

        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(30, X_combined.shape[1]))
        X_pca = pca.fit_transform(X_combined)

        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        ho = hm.run_harmony(X_pca, meta, "batch", max_iter_harmony=20)
        Z = ho.Z_corr
        if hasattr(Z, 'cpu'):
            Z = Z.cpu().numpy()
        # Z_corr shape: (n_cells, n_pcs) — already correct orientation
        X_harmony = Z

        # labeled=target(mouse), unlabeled=source(human)
        X_harmony_labeled = X_harmony[n_src:]
        X_harmony_unlabeled = X_harmony[:n_src]

        for col in label_cols:
            if col not in adata_labeled_raw.obs.columns:
                continue
            raw_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)
            pred, _ = label_transfer_with_confidence(
                X_harmony_labeled, X_harmony_unlabeled, raw_labels, k=10,
            )
            pred_aligned = pred[keep]
            for res in resolutions:
                leiden = np.array(adata_cluster.obs[f"leiden_{res}"].values, dtype=str)
                ari = adjusted_rand_score(pred_aligned, leiden)
                results[f"baseline_harmony_{col}_leiden{res}_ari"] = ari
                print(f"  Harmony {col} vs Leiden(res={res}): ARI={ari:.4f}")
    except Exception as e:
        print(f"  Harmony failed: {e}")

    # --- Scanorama baseline ---
    print("\n=== Scanorama baseline ===")
    try:
        import scanorama

        datasets = [X_src_shared.copy(), X_tgt_shared.copy()]
        genes_list = [shared_genes, shared_genes]
        corrected, _, _ = scanorama.correct(datasets, genes_list, return_dimred=True)
        X_scan_src = corrected[0]
        X_scan_tgt = corrected[1]

        X_scan_labeled = X_scan_tgt
        X_scan_unlabeled = X_scan_src

        for col in label_cols:
            if col not in adata_labeled_raw.obs.columns:
                continue
            raw_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)
            pred, _ = label_transfer_with_confidence(
                X_scan_labeled, X_scan_unlabeled, raw_labels, k=10,
            )
            pred_aligned = pred[keep]
            for res in resolutions:
                leiden = np.array(adata_cluster.obs[f"leiden_{res}"].values, dtype=str)
                ari = adjusted_rand_score(pred_aligned, leiden)
                results[f"baseline_scanorama_{col}_leiden{res}_ari"] = ari
                print(f"  Scanorama {col} vs Leiden(res={res}): ARI={ari:.4f}")
    except Exception as e:
        print(f"  Scanorama failed: {e}")

    # --- BBKNN baseline ---
    print("\n=== BBKNN baseline ===")
    try:
        import bbknn

        X_combined_bb = np.vstack([X_src_shared, X_tgt_shared])
        n_src_bb = X_src_shared.shape[0]

        adata_bb = ad.AnnData(X=X_combined_bb)
        adata_bb.obs["batch"] = ["source"] * n_src_bb + ["target"] * X_tgt_shared.shape[0]

        from sklearn.decomposition import PCA
        pca_bb = PCA(n_components=min(30, X_combined_bb.shape[1]))
        adata_bb.obsm["X_pca"] = pca_bb.fit_transform(X_combined_bb)

        bbknn.bbknn(adata_bb, batch_key="batch")

        # Use PCA space (30D) for downstream metrics — NOT connectivities or UMAP
        X_bb_reduced = adata_bb.obsm["X_pca"]

        X_bb_labeled = X_bb_reduced[n_src_bb:]
        X_bb_unlabeled = X_bb_reduced[:n_src_bb]

        for col in label_cols:
            if col not in adata_labeled_raw.obs.columns:
                continue
            raw_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)
            pred, _ = label_transfer_with_confidence(
                X_bb_labeled, X_bb_unlabeled, raw_labels, k=10,
            )
            pred_aligned = pred[keep]
            for res in resolutions:
                leiden = np.array(adata_cluster.obs[f"leiden_{res}"].values, dtype=str)
                ari = adjusted_rand_score(pred_aligned, leiden)
                results[f"baseline_bbknn_{col}_leiden{res}_ari"] = ari
                print(f"  BBKNN {col} vs Leiden(res={res}): ARI={ari:.4f}")
    except Exception as e:
        print(f"  BBKNN failed: {e}")

    # Save results
    out_path = BASE / "outputs" / "visium_human_brain_merfish_mouse_brain" / "baseline_ari_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n## Baseline Leiden ARI Summary (Best Resolution)")
    print("| Method | class ARI | subclass ARI | neurotransmitter ARI | supertype ARI |")
    print("|--------|:---------:|:------------:|:--------------------:|:-------------:|")

    for method in ["expression", "harmony", "scanorama", "bbknn"]:
        row = f"| {method.capitalize()} |"
        for col in ["class", "subclass", "neurotransmitter", "supertype"]:
            best_ari = max(
                results.get(f"baseline_{method}_{col}_leiden{res}_ari", 0)
                for res in resolutions
            )
            row += f" {best_ari:.3f} |"
        print(row)


if __name__ == "__main__":
    main()
