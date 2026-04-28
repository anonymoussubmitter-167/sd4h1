#!/usr/bin/env python3
"""Run baselines with 5 bootstrap resamples for variance estimation.

For each of 5 random seeds, resamples 80% of cells (with replacement),
runs Harmony/Scanorama/BBKNN/Expression baselines, and computes best ARI
across the standard Leiden resolution sweep.

This gives baselines mean +/- std, enabling Welch's t-test against BRIDGE.

Output: outputs/visium_human_brain_merfish_mouse_brain/multiseed_baseline_aris.json
"""

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rosetta.bridge.shared_space import build_shared_gene_space
from rosetta.data.ortholog_db import load_ortholog_mapping
from rosetta.utils.metrics import label_transfer_with_confidence

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed"

LEIDEN_RESOLUTIONS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
N_RESAMPLES = 5
RESAMPLE_FRAC = 0.8
LABEL_COLS = ["class", "subclass", "neurotransmitter", "supertype"]


def _best_ari(pred_labels, adata_cluster):
    """Compute best ARI across all Leiden resolutions."""
    best = -1.0
    for res in LEIDEN_RESOLUTIONS:
        key = f"leiden_{res}"
        if key in adata_cluster.obs.columns:
            leiden = np.array(adata_cluster.obs[key].values, dtype=str)
            ari = adjusted_rand_score(pred_labels, leiden)
            best = max(best, ari)
    return best


def run_baselines_once(
    X_src_shared, X_tgt_shared, shared_genes,
    adata_labeled_raw, adata_unlabeled_raw,
    adata_cluster, keep_idx,
    rng, label_cols,
):
    """Run all baselines once with a specific cell subsample, return dict of ARIs."""
    n_src = X_src_shared.shape[0]
    n_tgt = X_tgt_shared.shape[0]

    # Resample 80% with replacement
    src_idx = rng.choice(n_src, int(n_src * RESAMPLE_FRAC), replace=True)
    tgt_idx = rng.choice(n_tgt, int(n_tgt * RESAMPLE_FRAC), replace=True)

    X_src_sub = X_src_shared[src_idx]
    X_tgt_sub = X_tgt_shared[tgt_idx]

    # Target has labels -> transfer to source (mouse -> human)
    X_labeled_sub = X_tgt_sub
    X_unlabeled_sub = X_src_sub
    labeled_idx = tgt_idx
    unlabeled_idx = src_idx

    result = {}

    for col in label_cols:
        if col not in adata_labeled_raw.obs.columns:
            continue
        raw_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)

        # --- Expression baseline ---
        pred_expr, _ = label_transfer_with_confidence(
            X_labeled_sub, X_unlabeled_sub, raw_labels[labeled_idx], k=10,
        )
        # Map back to preprocessed cells
        pred_full = np.full(adata_unlabeled_raw.n_obs, "unknown", dtype=object)
        for i, orig_i in enumerate(unlabeled_idx):
            pred_full[orig_i] = pred_expr[i]
        pred_aligned = pred_full[keep_idx]
        result[f"expression_{col}"] = _best_ari(pred_aligned, adata_cluster)

    # --- Harmony baseline ---
    try:
        import harmonypy as hm
        import pandas as pd
        from sklearn.decomposition import PCA

        X_combined = np.vstack([X_src_sub, X_tgt_sub])
        n_src_h = X_src_sub.shape[0]
        meta = pd.DataFrame({"batch": ["source"] * n_src_h + ["target"] * X_tgt_sub.shape[0]})
        n_pcs = min(30, X_combined.shape[1] - 1)
        pca = PCA(n_components=n_pcs)
        X_pca = pca.fit_transform(X_combined)
        ho = hm.run_harmony(X_pca, meta, ["batch"], max_iter_harmony=20, device="cpu")
        X_harmony = np.array(ho.Z_corr, dtype=np.float32)

        X_h_labeled = X_harmony[n_src_h:]
        X_h_unlabeled = X_harmony[:n_src_h]

        for col in label_cols:
            if col not in adata_labeled_raw.obs.columns:
                continue
            raw_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)
            pred_h, _ = label_transfer_with_confidence(
                X_h_labeled, X_h_unlabeled, raw_labels[labeled_idx], k=10,
            )
            pred_full = np.full(adata_unlabeled_raw.n_obs, "unknown", dtype=object)
            for i, orig_i in enumerate(unlabeled_idx):
                pred_full[orig_i] = pred_h[i]
            pred_aligned = pred_full[keep_idx]
            result[f"harmony_{col}"] = _best_ari(pred_aligned, adata_cluster)
    except Exception as e:
        print(f"  Harmony failed: {e}")

    # --- Scanorama baseline ---
    try:
        import scanorama
        gene_names = [str(g) for g in shared_genes]
        corrected, _ = scanorama.correct(
            [X_src_sub, X_tgt_sub], [gene_names, gene_names],
        )
        import scipy.sparse as sp
        X_sc_src = np.array(corrected[0].todense() if sp.issparse(corrected[0]) else corrected[0], dtype=np.float32)
        X_sc_tgt = np.array(corrected[1].todense() if sp.issparse(corrected[1]) else corrected[1], dtype=np.float32)

        for col in label_cols:
            if col not in adata_labeled_raw.obs.columns:
                continue
            raw_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)
            pred_sc, _ = label_transfer_with_confidence(
                X_sc_tgt, X_sc_src, raw_labels[labeled_idx], k=10,
            )
            pred_full = np.full(adata_unlabeled_raw.n_obs, "unknown", dtype=object)
            for i, orig_i in enumerate(unlabeled_idx):
                pred_full[orig_i] = pred_sc[i]
            pred_aligned = pred_full[keep_idx]
            result[f"scanorama_{col}"] = _best_ari(pred_aligned, adata_cluster)
    except Exception as e:
        print(f"  Scanorama failed: {e}")

    # --- BBKNN baseline ---
    try:
        import bbknn
        X_combined_bb = np.vstack([X_src_sub, X_tgt_sub])
        n_src_bb = X_src_sub.shape[0]
        adata_bb = ad.AnnData(X=X_combined_bb)
        adata_bb.obs["batch"] = ["source"] * n_src_bb + ["target"] * X_tgt_sub.shape[0]

        n_pcs_bb = min(30, X_combined_bb.shape[1] - 1)
        sc.pp.pca(adata_bb, n_comps=n_pcs_bb)
        bbknn.bbknn(adata_bb, batch_key="batch", n_pcs=n_pcs_bb)
        # Use PCA space (30D) — NOT UMAP
        X_bb = adata_bb.obsm["X_pca"]

        X_bb_labeled = X_bb[n_src_bb:]
        X_bb_unlabeled = X_bb[:n_src_bb]

        for col in label_cols:
            if col not in adata_labeled_raw.obs.columns:
                continue
            raw_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)
            pred_bb, _ = label_transfer_with_confidence(
                X_bb_labeled, X_bb_unlabeled, raw_labels[labeled_idx], k=10,
            )
            pred_full = np.full(adata_unlabeled_raw.n_obs, "unknown", dtype=object)
            for i, orig_i in enumerate(unlabeled_idx):
                pred_full[orig_i] = pred_bb[i]
            pred_aligned = pred_full[keep_idx]
            result[f"bbknn_{col}"] = _best_ari(pred_aligned, adata_cluster)
    except Exception as e:
        print(f"  BBKNN failed: {e}")

    return result


def main():
    print("Loading data...")
    adata_src_raw = ad.read_h5ad(DATA / "visium_human_brain.h5ad")
    adata_tgt_raw = ad.read_h5ad(DATA / "merfish_mouse_brain.h5ad")

    ortholog_map = load_ortholog_mapping("human", "mouse")
    X_src_shared, X_tgt_shared, shared_genes = build_shared_gene_space(
        adata_src_raw, adata_tgt_raw, ortholog_map,
    )
    print(f"  Shared genes: {len(shared_genes)}")

    # Preprocess unlabeled (human) data for Leiden clustering
    print("Computing Leiden clustering...")
    adata_cluster = adata_src_raw.copy()
    sc.pp.filter_cells(adata_cluster, min_genes=1)
    sc.pp.filter_genes(adata_cluster, min_cells=1)
    sc.pp.normalize_total(adata_cluster, target_sum=1e4)
    sc.pp.log1p(adata_cluster)
    sc.pp.highly_variable_genes(adata_cluster, n_top_genes=min(2000, adata_cluster.n_vars))
    adata_cluster = adata_cluster[:, adata_cluster.var.highly_variable].copy()
    sc.pp.pca(adata_cluster, n_comps=min(50, adata_cluster.n_vars - 1))
    sc.pp.neighbors(adata_cluster, n_neighbors=15)

    for res in LEIDEN_RESOLUTIONS:
        sc.tl.leiden(adata_cluster, resolution=res, key_added=f"leiden_{res}", flavor="igraph", directed=False, n_iterations=2)

    # Cell index alignment
    raw_ids = list(adata_src_raw.obs_names)
    preproc_ids = list(adata_cluster.obs_names)
    idx_map = {name: i for i, name in enumerate(raw_ids)}
    keep_idx = [idx_map[n] for n in preproc_ids if n in idx_map]
    print(f"  Aligned {len(raw_ids)} raw -> {len(keep_idx)} preprocessed cells")

    # Run N_RESAMPLES bootstrap resamples
    all_results = {}
    for resample_i in range(N_RESAMPLES):
        print(f"\n=== Resample {resample_i} ===")
        rng = np.random.default_rng(resample_i)
        res = run_baselines_once(
            X_src_shared, X_tgt_shared, shared_genes,
            adata_tgt_raw, adata_src_raw,
            adata_cluster, keep_idx, rng, LABEL_COLS,
        )
        for k, v in res.items():
            all_results.setdefault(k, []).append(v)
            print(f"  {k}: {v:.4f}")

    # Summarize
    print("\n" + "=" * 60)
    print("MULTI-SEED BASELINE SUMMARY")
    print("=" * 60)

    summary = {}
    for k, vals in sorted(all_results.items()):
        vals = np.array(vals)
        summary[k] = {
            "values": vals.tolist(),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
        }
        print(f"  {k}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")

    out_path = BASE / "outputs" / "visium_human_brain_merfish_mouse_brain" / "multiseed_baseline_aris.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
