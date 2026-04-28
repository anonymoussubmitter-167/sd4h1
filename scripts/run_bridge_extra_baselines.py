#!/usr/bin/env python
"""Additional BRIDGE baselines: Tangram and integrative NMF (LIGER-style).

Tangram: Maps spatial transcriptomics to reference scRNA-seq using OT.
iNMF:    Integrative NMF on shared gene space (LIGER algorithm).

Usage:
    python scripts/run_bridge_extra_baselines.py \
        --source visium_human_brain --target merfish_mouse_brain
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
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bridge import DATASET_CONFIGS

from rosetta.data.ortholog_db import load_ortholog_mapping
from rosetta.bridge.shared_space import build_shared_gene_space as _build_shared_gene_space
from rosetta.utils.metrics import (
    alignment_score,
    label_transfer_accuracy,
    spatial_label_smoothness,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

KNOWN_LABEL_COLUMNS = [
    "class", "subclass", "neurotransmitter", "supertype", "cluster",
    "Cell_Type", "cell_type", "celltype",
]


def build_shared_gene_space(adata_src, adata_tgt, ortholog_map):
    """Build shared ortholog gene expression matrices.

    Uses the proper shared_space utility that handles Ensembl IDs
    via gene_symbol column in .var.
    """
    X_src, X_tgt, gene_names = _build_shared_gene_space(adata_src, adata_tgt, ortholog_map)
    if len(gene_names) == 0:
        return None, None, []
    return X_src, X_tgt, gene_names


def label_transfer_with_confidence(z_labeled, z_unlabeled, labels, k=10):
    """kNN label transfer with confidence scores."""
    tree = KDTree(z_labeled)
    dists, indices = tree.query(z_unlabeled, k=k)
    weights = 1.0 / (dists + 1e-8)

    unique_labels = np.unique(labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    pred_labels = []
    confidences = []
    for i in range(len(z_unlabeled)):
        votes = np.zeros(len(unique_labels))
        for j in range(k):
            idx = label_to_idx[labels[indices[i, j]]]
            votes[idx] += weights[i, j]
        votes /= votes.sum()
        best = np.argmax(votes)
        pred_labels.append(unique_labels[best])
        confidences.append(votes[best])

    return np.array(pred_labels), np.array(confidences)


def run_tangram_baseline(adata_src, adata_tgt, ortholog_map, source_config, target_config):
    """Run Tangram spatial mapping baseline."""
    import tangram as tg

    # Build shared gene space
    X_src, X_tgt, shared_genes = build_shared_gene_space(adata_src, adata_tgt, ortholog_map)
    if X_src is None or len(shared_genes) < 5:
        logger.warning("Tangram: too few shared genes (%d)", len(shared_genes) if shared_genes else 0)
        return {}

    # Tangram needs: sc_adata (reference) and sp_adata (spatial)
    # We use target (labeled) as "single-cell reference" and source as "spatial"
    target_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_tgt.obs.columns]
    if not target_label_cols:
        # Try the other direction
        target_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_src.obs.columns]
        if target_label_cols:
            adata_ref, adata_sp = adata_src, adata_tgt
            X_ref, X_sp = X_src, X_tgt
        else:
            logger.warning("Tangram: no label columns found")
            return {}
    else:
        adata_ref, adata_sp = adata_tgt, adata_src
        X_ref, X_sp = X_tgt, X_src

    # Subsample for Tangram (memory-intensive)
    max_cells = 5000
    rng = np.random.default_rng(42)
    if X_ref.shape[0] > max_cells:
        idx_ref = rng.choice(X_ref.shape[0], max_cells, replace=False)
        X_ref_sub = X_ref[idx_ref]
        adata_ref_sub = adata_ref[idx_ref].copy()
    else:
        X_ref_sub = X_ref
        adata_ref_sub = adata_ref.copy()
        idx_ref = np.arange(X_ref.shape[0])

    if X_sp.shape[0] > max_cells:
        idx_sp = rng.choice(X_sp.shape[0], max_cells, replace=False)
        X_sp_sub = X_sp[idx_sp]
        adata_sp_sub = adata_sp[idx_sp].copy()
    else:
        X_sp_sub = X_sp
        adata_sp_sub = adata_sp.copy()
        idx_sp = np.arange(X_sp.shape[0])

    # Create AnnData with shared genes for Tangram
    sc_ad = ad.AnnData(X=X_ref_sub)
    sc_ad.var_names = shared_genes
    sc_ad.obs = adata_ref_sub.obs.copy()

    sp_ad = ad.AnnData(X=X_sp_sub)
    sp_ad.var_names = shared_genes
    sp_ad.obs = adata_sp_sub.obs.copy()
    if "spatial" in adata_sp_sub.obsm:
        sp_ad.obsm["spatial"] = np.array(adata_sp_sub.obsm["spatial"][:, :2], dtype=np.float32)

    # Run Tangram
    logger.info("Running Tangram (sc=%d, sp=%d, genes=%d)...",
                sc_ad.n_obs, sp_ad.n_obs, len(shared_genes))

    tg.pp_adatas(sc_ad, sp_ad, genes=shared_genes)

    ad_map = tg.map_cells_to_space(
        sc_ad, sp_ad,
        mode="cells",
        density_prior="rna_count_based",
        num_epochs=500,
        device="cpu",
    )

    # ad_map.X is (n_sp, n_sc) mapping matrix
    mapping_matrix = ad_map.X  # spatial -> single-cell soft assignment
    if sp.issparse(mapping_matrix):
        mapping_matrix = np.array(mapping_matrix.todense())

    results = {}

    # Transfer labels via mapping matrix
    for col in target_label_cols:
        if col not in adata_ref_sub.obs.columns:
            continue
        ref_labels = np.array(adata_ref_sub.obs[col].values, dtype=str)
        unique_labels = np.unique(ref_labels)
        n_classes = len(unique_labels)
        if n_classes < 2 or n_classes > 200:
            continue

        # Build label probability matrix
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        label_onehot = np.zeros((len(ref_labels), n_classes))
        for i, l in enumerate(ref_labels):
            label_onehot[i, label_to_idx[l]] = 1.0

        # Transfer: mapping_matrix @ label_onehot
        transferred_probs = mapping_matrix @ label_onehot
        row_sums = transferred_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        transferred_probs /= row_sums

        pred_labels = np.array([unique_labels[np.argmax(transferred_probs[i])]
                                for i in range(len(transferred_probs))])
        confidences = np.array([transferred_probs[i].max() for i in range(len(transferred_probs))])

        results[f"tangram_{col}_mean_confidence"] = float(np.mean(confidences))

        # Spatial smoothness
        if "spatial" in adata_sp_sub.obsm:
            coords = np.array(adata_sp_sub.obsm["spatial"][:, :2], dtype=np.float32)
            smoothness = spatial_label_smoothness(coords, pred_labels, k=10)
            results[f"tangram_{col}_spatial_smoothness"] = float(smoothness)
            logger.info("  Tangram %s: smoothness=%.4f, confidence=%.4f",
                        col, smoothness, np.mean(confidences))

    return results


def run_inmf_baseline(adata_src, adata_tgt, ortholog_map, source_config, target_config):
    """Run integrative NMF (LIGER-style) baseline.

    Joint NMF factorization: X_src ≈ W_src @ H, X_tgt ≈ W_tgt @ H
    where H (shared factors) aligns the species.
    """
    X_src, X_tgt, shared_genes = build_shared_gene_space(adata_src, adata_tgt, ortholog_map)
    if X_src is None or len(shared_genes) < 5:
        logger.warning("iNMF: too few shared genes (%d)", len(shared_genes) if shared_genes else 0)
        return {}

    # Ensure non-negative
    X_src = np.maximum(X_src, 0)
    X_tgt = np.maximum(X_tgt, 0)

    # Subsample for speed
    max_cells = 10000
    rng = np.random.default_rng(42)
    if X_src.shape[0] > max_cells:
        idx_src = rng.choice(X_src.shape[0], max_cells, replace=False)
        X_src_sub = X_src[idx_src]
    else:
        X_src_sub = X_src
        idx_src = np.arange(X_src.shape[0])

    if X_tgt.shape[0] > max_cells:
        idx_tgt = rng.choice(X_tgt.shape[0], max_cells, replace=False)
        X_tgt_sub = X_tgt[idx_tgt]
    else:
        X_tgt_sub = X_tgt
        idx_tgt = np.arange(X_tgt.shape[0])

    # Joint NMF: stack both datasets, shared H
    n_components = 30
    X_joint = np.vstack([X_src_sub, X_tgt_sub])
    n_src = X_src_sub.shape[0]

    logger.info("Running iNMF (n=%d, genes=%d, k=%d)...",
                X_joint.shape[0], X_joint.shape[1], n_components)

    model = NMF(n_components=n_components, init="nndsvda", max_iter=500, random_state=42)
    W_joint = model.fit_transform(X_joint)

    W_src_sub = W_joint[:n_src]
    W_tgt_sub = W_joint[n_src:]

    # Detect labeled side
    target_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_tgt.obs.columns]
    source_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_src.obs.columns]

    if target_label_cols:
        W_labeled, W_unlabeled = W_tgt_sub, W_src_sub
        adata_labeled = adata_tgt
        adata_unlabeled = adata_src
        label_cols = target_label_cols
        idx_labeled, idx_unlabeled = idx_tgt, idx_src
    elif source_label_cols:
        W_labeled, W_unlabeled = W_src_sub, W_tgt_sub
        adata_labeled = adata_src
        adata_unlabeled = adata_tgt
        label_cols = source_label_cols
        idx_labeled, idx_unlabeled = idx_src, idx_tgt
    else:
        logger.warning("iNMF: no label columns found")
        return {}

    results = {}

    # kNN mixing
    species_labels = np.array([0] * len(W_src_sub) + [1] * len(W_tgt_sub))
    W_all = np.vstack([W_src_sub, W_tgt_sub])
    for k in [10, 50, 100]:
        k_eff = min(k, W_all.shape[0] - 1)
        mix = alignment_score(W_all, species_labels, k=k_eff)
        results[f"inmf_knn_mixing_k{k}"] = float(mix)
        logger.info("  iNMF kNN mixing (k=%d): %.4f", k, mix)

    # Label transfer
    for col in label_cols:
        if col not in adata_labeled.obs.columns:
            continue
        raw_labels = np.array(adata_labeled.obs[col].values[idx_labeled], dtype=str)
        n_classes = len(np.unique(raw_labels))
        if n_classes < 2 or n_classes > 200:
            continue

        pred, conf = label_transfer_with_confidence(W_labeled, W_unlabeled, raw_labels, k=10)
        results[f"inmf_{col}_mean_confidence"] = float(np.mean(conf))

        # Spatial smoothness
        if "spatial" in adata_unlabeled.obsm:
            coords = np.array(adata_unlabeled.obsm["spatial"][idx_unlabeled, :2], dtype=np.float32)
            smoothness = spatial_label_smoothness(coords, pred, k=10)
            results[f"inmf_{col}_spatial_smoothness"] = float(smoothness)
            logger.info("  iNMF %s: smoothness=%.4f, confidence=%.4f",
                        col, smoothness, np.mean(conf))

        # ARI vs Leiden on unlabeled expression
        if len(pred) == len(idx_unlabeled) and "spatial" in adata_unlabeled.obsm:
            adata_cl = adata_unlabeled[idx_unlabeled].copy()
            X_cl = adata_cl.X
            if sp.issparse(X_cl):
                X_cl = np.array(X_cl.todense())
            adata_cl_ad = ad.AnnData(X=X_cl.astype(np.float32))
            sc.pp.pca(adata_cl_ad, n_comps=min(50, adata_cl_ad.n_vars - 1))
            sc.pp.neighbors(adata_cl_ad, n_neighbors=15)

            best_ari = -1.0
            for resolution in [0.1, 0.3, 0.5, 1.0, 2.0]:
                sc.tl.leiden(adata_cl_ad, resolution=resolution, key_added="leiden")
                leiden_labels = np.array(adata_cl_ad.obs["leiden"].values, dtype=str)
                ari = adjusted_rand_score(pred, leiden_labels)
                if ari > best_ari:
                    best_ari = ari
            results[f"inmf_{col}_best_ari"] = float(best_ari)
            logger.info("  iNMF %s best ARI: %.4f", col, best_ari)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-tangram", action="store_true")
    parser.add_argument("--skip-inmf", action="store_true")
    args = parser.parse_args()

    source_config = DATASET_CONFIGS[args.source]
    target_config = DATASET_CONFIGS[args.target]
    source_species = source_config["species"]
    target_species = target_config["species"]

    pair_name = f"{args.source}_{args.target}"
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / pair_name / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    adata_src = ad.read_h5ad(PROCESSED_DIR / f"{args.source}.h5ad")
    adata_tgt = ad.read_h5ad(PROCESSED_DIR / f"{args.target}.h5ad")
    ortholog_map = load_ortholog_mapping(source_species, target_species)
    logger.info("Source: %d cells, Target: %d cells",
                adata_src.n_obs, adata_tgt.n_obs)

    all_results = {}

    # Tangram
    if not args.skip_tangram:
        logger.info("=" * 60)
        logger.info("Running Tangram baseline...")
        logger.info("=" * 60)
        try:
            tangram_results = run_tangram_baseline(
                adata_src, adata_tgt, ortholog_map, source_config, target_config,
            )
            all_results.update(tangram_results)
        except Exception as e:
            logger.error("Tangram failed: %s", e)
            import traceback; traceback.print_exc()

    # iNMF (LIGER-style)
    if not args.skip_inmf:
        logger.info("=" * 60)
        logger.info("Running iNMF (LIGER-style) baseline...")
        logger.info("=" * 60)
        try:
            inmf_results = run_inmf_baseline(
                adata_src, adata_tgt, ortholog_map, source_config, target_config,
            )
            all_results.update(inmf_results)
        except Exception as e:
            logger.error("iNMF failed: %s", e)
            import traceback; traceback.print_exc()

    # Save
    out_file = output_dir / "extra_baselines.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", out_file)

    print(f"\n{'='*60}")
    print(f"Extra Baselines: {args.source} <-> {args.target}")
    print(f"{'='*60}")
    for k, v in sorted(all_results.items()):
        print(f"  {k}: {v:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
