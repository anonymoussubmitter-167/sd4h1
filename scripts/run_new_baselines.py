#!/usr/bin/env python
"""Run PASTE and scVI baselines standalone (faster than full eval)."""

import argparse
import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bridge import DATASET_CONFIGS

from rosetta.bridge.shared_space import _get_gene_lookup, build_shared_gene_space
from rosetta.data.ortholog_db import load_ortholog_mapping
from rosetta.utils.metrics import alignment_score, label_transfer_with_confidence, spatial_label_smoothness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

KNOWN_LABEL_COLUMNS = [
    "class", "subclass", "neurotransmitter", "supertype", "cluster",
    "Cell_Type", "cell_type", "celltype",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    source_config = DATASET_CONFIGS.get(args.source, {})
    target_config = DATASET_CONFIGS.get(args.target, {})
    source_species = source_config.get("species", "human")
    target_species = target_config.get("species", "mouse")

    # Load data
    adata_src_raw = ad.read_h5ad(PROCESSED_DIR / f"{args.source}.h5ad")
    adata_tgt_raw = ad.read_h5ad(PROCESSED_DIR / f"{args.target}.h5ad")
    logger.info("Source: %d cells, Target: %d cells", adata_src_raw.n_obs, adata_tgt_raw.n_obs)

    # Ortholog map + shared genes
    ortholog_map = load_ortholog_mapping(source_species, target_species)
    X_src_shared, X_tgt_shared, shared_genes = build_shared_gene_space(
        adata_src_raw, adata_tgt_raw, ortholog_map)
    logger.info("Shared ortholog genes: %d", len(shared_genes))

    # Find label columns
    target_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_tgt_raw.obs.columns]
    source_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_src_raw.obs.columns]
    if target_label_cols:
        label_cols = target_label_cols
        adata_labeled_raw = adata_tgt_raw
        adata_unlabeled_raw = adata_src_raw
        coords_unlabeled_raw = np.array(adata_src_raw.obsm["spatial"][:, :2], dtype=np.float32)
    else:
        label_cols = source_label_cols
        adata_labeled_raw = adata_src_raw
        adata_unlabeled_raw = adata_tgt_raw
        coords_unlabeled_raw = np.array(adata_tgt_raw.obsm["spatial"][:, :2], dtype=np.float32)
    logger.info("Label columns: %s", label_cols)

    # ---- PASTE baseline ----
    if len(shared_genes) > 0:
        logger.info("PASTE baseline (spatial FGW optimal transport)...")
        try:
            import functools
            import paste
            import paste.PASTE
            import ot

            _orig_fgw = paste.PASTE.my_fused_gromov_wasserstein
            def _patched_fgw(*args, **kwargs):
                _orig_cg = ot.optim.cg
                @functools.wraps(_orig_cg)
                def _cg_wrapper(p, q, M, alpha, f, df, G0, line_search_fn, *cg_args, **cg_kwargs):
                    @functools.wraps(line_search_fn)
                    def _ls_compat(cost, G, deltaG, Mi, cost_G, *ls_extra, **ls_kw):
                        return line_search_fn(cost, G, deltaG, Mi, cost_G, **ls_kw)
                    return _orig_cg(p, q, M, alpha, f, df, G0, _ls_compat, *cg_args, **cg_kwargs)
                ot.optim.cg = _cg_wrapper
                try:
                    return _orig_fgw(*args, **kwargs)
                finally:
                    ot.optim.cg = _orig_cg
            paste.PASTE.my_fused_gromov_wasserstein = _patched_fgw

            adata_paste_src = ad.AnnData(
                X=X_src_shared.astype(np.float64),
                obsm={"spatial": np.array(adata_src_raw.obsm["spatial"][:, :2], dtype=np.float64)})
            adata_paste_src.var_names = [str(g) for g in shared_genes]
            adata_paste_tgt = ad.AnnData(
                X=X_tgt_shared.astype(np.float64),
                obsm={"spatial": np.array(adata_tgt_raw.obsm["spatial"][:, :2], dtype=np.float64)})
            adata_paste_tgt.var_names = [str(g) for g in shared_genes]

            max_paste = 2000
            rng = np.random.default_rng(42)
            paste_src_idx = np.arange(adata_paste_src.n_obs)
            paste_tgt_idx = np.arange(adata_paste_tgt.n_obs)
            if adata_paste_src.n_obs > max_paste:
                paste_src_idx = rng.choice(adata_paste_src.n_obs, max_paste, replace=False)
                adata_paste_src = adata_paste_src[paste_src_idx].copy()
            if adata_paste_tgt.n_obs > max_paste:
                paste_tgt_idx = rng.choice(adata_paste_tgt.n_obs, max_paste, replace=False)
                adata_paste_tgt = adata_paste_tgt[paste_tgt_idx].copy()

            logger.info("PASTE: aligning %d x %d cells (alpha=0.1)...",
                        adata_paste_src.n_obs, adata_paste_tgt.n_obs)
            pi_paste = paste.pairwise_align(adata_paste_src, adata_paste_tgt, alpha=0.1, numItermax=50)

            # Entropy check
            pi_row_norm = pi_paste / (pi_paste.sum(axis=1, keepdims=True) + 1e-20)
            row_entropy = -(pi_row_norm * np.log(pi_row_norm + 1e-20)).sum(axis=1)
            max_entropy = np.log(pi_paste.shape[1])
            entropy_ratio = row_entropy.mean() / max_entropy
            results["paste_transport_entropy_ratio"] = float(entropy_ratio)
            logger.info("PASTE entropy ratio: %.4f (1.0 = degenerate)", entropy_ratio)

            if entropy_ratio > 0.99:
                logger.warning("PASTE transport plan degenerate - method not applicable for cross-species")
                results["paste_status"] = "degenerate"
            else:
                results["paste_status"] = "converged"

            # Label transfer via transport plan
            if target_label_cols:
                T = pi_paste
                paste_labeled_idx = paste_tgt_idx
            else:
                T = pi_paste.T
                paste_labeled_idx = paste_src_idx
            T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-10)

            paste_coords = coords_unlabeled_raw[paste_src_idx if target_label_cols else paste_tgt_idx]

            for col in label_cols:
                if col not in adata_labeled_raw.obs.columns:
                    continue
                all_labels = np.array(adata_labeled_raw.obs[col].values, dtype=str)
                labeled_labels = all_labels[paste_labeled_idx]
                unique_labels = np.unique(labeled_labels)
                Y = np.zeros((len(labeled_labels), len(unique_labels)))
                for i, l in enumerate(labeled_labels):
                    Y[i, list(unique_labels).index(l)] = 1.0
                pred_probs = T_norm @ Y
                pred_paste = unique_labels[pred_probs.argmax(axis=1)]
                s = spatial_label_smoothness(paste_coords, pred_paste, k=10)
                results[f"paste_{col}_spatial_smoothness"] = float(s)
                logger.info("PASTE %s spatial smoothness: %.4f", col, s)

            paste.PASTE.my_fused_gromov_wasserstein = _orig_fgw
        except Exception as e:
            logger.warning("PASTE failed: %s", e)
            import traceback; traceback.print_exc()

    # ---- scVI baseline ----
    if len(shared_genes) > 0:
        logger.info("scVI baseline (VAE-based batch integration)...")
        try:
            import scvi

            src_lookup = _get_gene_lookup(adata_src_raw)
            tgt_lookup = _get_gene_lookup(adata_tgt_raw)
            src_cols, tgt_cols, vi_gene_names = [], [], []
            for sg, tg in ortholog_map.forward.items():
                if sg in src_lookup and tg in tgt_lookup:
                    src_cols.append(src_lookup[sg])
                    tgt_cols.append(tgt_lookup[tg])
                    vi_gene_names.append(sg)

            X_src_raw = adata_src_raw.X[:, src_cols]
            X_tgt_raw = adata_tgt_raw.X[:, tgt_cols]
            if sp.issparse(X_src_raw): X_src_raw = np.array(X_src_raw.todense())
            if sp.issparse(X_tgt_raw): X_tgt_raw = np.array(X_tgt_raw.todense())

            X_joint = np.vstack([X_src_raw, X_tgt_raw]).astype(np.float32)
            n_src_vi = X_src_raw.shape[0]

            if X_joint.min() < 0:
                X_joint = np.expm1(np.abs(X_joint))
            X_joint = np.round(np.maximum(X_joint, 0)).astype(np.float32)

            adata_vi = ad.AnnData(X=X_joint)
            adata_vi.var_names = [str(g) for g in vi_gene_names]
            adata_vi.obs["batch"] = ["source"] * n_src_vi + ["target"] * (X_joint.shape[0] - n_src_vi)
            adata_vi.obs["batch"] = adata_vi.obs["batch"].astype("category")

            max_scvi = 10000
            total_vi = adata_vi.n_obs
            if total_vi > max_scvi:
                sc.pp.subsample(adata_vi, n_obs=max_scvi, random_state=42)
                logger.info("Subsampled to %d cells for scVI", adata_vi.n_obs)

            scvi.model.SCVI.setup_anndata(adata_vi, batch_key="batch")
            vae = scvi.model.SCVI(adata_vi, n_latent=30, n_layers=2, gene_likelihood="nb")
            logger.info("Training scVI (100 epochs)...")
            vae.train(max_epochs=100, early_stopping=True, batch_size=256, train_size=0.9,
                      accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)

            Z_vi = vae.get_latent_representation()
            Z_vi_src = Z_vi[adata_vi.obs["batch"] == "source"]
            Z_vi_tgt = Z_vi[adata_vi.obs["batch"] == "target"]
            logger.info("scVI latent: src=%s, tgt=%s", Z_vi_src.shape, Z_vi_tgt.shape)

            # Track original dataset indices for subsampled cells
            src_mask_vi = adata_vi.obs["batch"] == "source"
            tgt_mask_vi = adata_vi.obs["batch"] == "target"
            # Joint indices -> original dataset indices
            joint_idx = adata_vi.obs.index.astype(int).values
            src_orig_idx = joint_idx[src_mask_vi.values]  # indices into source (0..n_src-1)
            tgt_orig_idx = joint_idx[tgt_mask_vi.values] - n_src_vi  # offset by n_src

            if target_label_cols:
                Z_vi_labeled, Z_vi_unlabeled = Z_vi_tgt, Z_vi_src
                vi_labeled_raw = adata_tgt_raw
                vi_labeled_orig_idx = tgt_orig_idx
                vi_coords = np.array(adata_src_raw.obsm["spatial"][:, :2], dtype=np.float32)
                if adata_vi.n_obs < total_vi:
                    vi_coords = vi_coords[src_orig_idx]
            else:
                Z_vi_labeled, Z_vi_unlabeled = Z_vi_src, Z_vi_tgt
                vi_labeled_raw = adata_src_raw
                vi_labeled_orig_idx = src_orig_idx
                vi_coords = np.array(adata_tgt_raw.obsm["spatial"][:, :2], dtype=np.float32)
                if adata_vi.n_obs < total_vi:
                    vi_coords = vi_coords[tgt_orig_idx]

            # kNN mixing
            Z_all = np.vstack([Z_vi_labeled, Z_vi_unlabeled])
            species = np.array([0] * Z_vi_labeled.shape[0] + [1] * Z_vi_unlabeled.shape[0])
            for k in [10, 50, 100]:
                k_eff = min(k, Z_all.shape[0] - 1)
                mix = alignment_score(Z_all, species, k=k_eff)
                results[f"scvi_knn_mixing_k{k}"] = float(mix)
                logger.info("scVI kNN mixing (k=%d): %.4f", k, mix)

            # Label transfer
            for col in label_cols:
                if col not in vi_labeled_raw.obs.columns:
                    continue
                raw_labels = np.array(vi_labeled_raw.obs[col].values, dtype=str)
                if Z_vi_labeled.shape[0] < len(raw_labels):
                    raw_labels = raw_labels[vi_labeled_orig_idx]

                pred_vi, conf_vi = label_transfer_with_confidence(Z_vi_labeled, Z_vi_unlabeled, raw_labels, k=10)
                if len(pred_vi) <= len(vi_coords):
                    s = spatial_label_smoothness(vi_coords[:len(pred_vi)], pred_vi, k=10)
                    results[f"scvi_{col}_spatial_smoothness"] = float(s)
                    logger.info("scVI %s spatial smoothness: %.4f", col, s)
                results[f"scvi_{col}_mean_confidence"] = float(np.mean(conf_vi))
                logger.info("scVI %s confidence: %.4f", col, np.mean(conf_vi))

        except Exception as e:
            logger.warning("scVI failed: %s", e)
            import traceback; traceback.print_exc()

    # Save results
    out_file = output_dir / "new_baselines.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_file)
    for k, v in sorted(results.items()):
        logger.info("  %s: %.4f" if isinstance(v, float) else "  %s: %s", k, v)


if __name__ == "__main__":
    main()
