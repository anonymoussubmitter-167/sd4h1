#!/usr/bin/env python3
"""Compute SAMap Leiden ARI for the Human-Mouse brain pair.

Re-runs SAMap alignment, does kNN label transfer in SAMap's embedding space
(mouse -> human), runs Leiden clustering on preprocessed human expression,
and computes ARI between transferred labels and Leiden clusters.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sps
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bridge import DATASET_CONFIGS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rosetta.bridge.shared_space import _get_gene_lookup
from rosetta.data.ortholog_db import load_ortholog_mapping
from rosetta.utils.metrics import label_transfer_with_confidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

KNOWN_LABEL_COLUMNS = ["class", "subclass", "neurotransmitter", "supertype", "cluster"]

SPECIES_PREFIX = {
    "human": "hu",
    "mouse": "mo",
}


def build_gnnm(ortholog_map, src_species, tgt_species, adata_src, adata_tgt):
    """Build gene-gene homology network (gnnm) from ortholog mapping."""
    src_prefix = SPECIES_PREFIX[src_species]
    tgt_prefix = SPECIES_PREFIX[tgt_species]

    src_lookup = _get_gene_lookup(adata_src)
    tgt_lookup = _get_gene_lookup(adata_tgt)

    src_genes_in_data = set(src_lookup.keys())
    tgt_genes_in_data = set(tgt_lookup.keys())

    pairs = []
    for sg, tg in ortholog_map.forward.items():
        if sg in src_genes_in_data and tg in tgt_genes_in_data:
            pairs.append((sg, tg))
    logger.info("Found %d ortholog pairs in both datasets", len(pairs))

    src_gene_set = sorted(set(p[0] for p in pairs))
    tgt_gene_set = sorted(set(p[1] for p in pairs))
    all_genes = [f"{src_prefix}_{g}" for g in src_gene_set] + [f"{tgt_prefix}_{g}" for g in tgt_gene_set]

    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    n = len(all_genes)

    rows, cols, vals = [], [], []
    for sg, tg in pairs:
        i = gene_to_idx[f"{src_prefix}_{sg}"]
        j = gene_to_idx[f"{tgt_prefix}_{tg}"]
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([1.0, 1.0])

    gnnm = sps.csr_matrix((vals, (rows, cols)), shape=(n, n))
    gns = np.array(all_genes)
    gns_dict = {
        src_prefix: np.array([f"{src_prefix}_{g}" for g in src_gene_set]),
        tgt_prefix: np.array([f"{tgt_prefix}_{g}" for g in tgt_gene_set]),
    }

    logger.info("Gene network: %s, %d edges", gnnm.shape, gnnm.nnz)
    return gnnm, gns, gns_dict


def prepare_sam_adata(adata, species_prefix, gene_set):
    """Prepare AnnData for SAM: raw counts, gene symbols, subsampled."""
    lookup = _get_gene_lookup(adata)
    gene_indices = [lookup[g] for g in gene_set if g in lookup]
    gene_names = [g for g in gene_set if g in lookup]

    X = adata.X[:, gene_indices]
    if sps.issparse(X):
        X = np.array(X.todense())

    if X.min() < 0:
        X = np.expm1(np.abs(X))
    X = np.maximum(X, 0).astype(np.float32)

    prefixed_names = [f"{species_prefix}_{g}" for g in gene_names]

    adata_sam = ad.AnnData(X=sps.csr_matrix(X))
    adata_sam.var_names = prefixed_names
    adata_sam.obs_names = [f"{species_prefix}_cell_{i}" for i in range(adata_sam.n_obs)]

    return adata_sam


def _apply_samap_compat_patches():
    """Fix SAMap/SAM compatibility with newer scipy/anndata versions."""
    import anndata._core.aligned_mapping as _am

    for cls in (sps.csr_matrix, sps.csc_matrix, sps.coo_matrix, sps.lil_matrix,
                sps.csr_array, sps.csc_array, sps.coo_array, sps.lil_array):
        if not hasattr(cls, 'A'):
            cls.A = property(lambda self: self.toarray())

    _orig_validate = _am.PairwiseArrays._validate_value
    def _safe_validate(self, val, key=None):
        if sps.issparse(val) and not isinstance(val, (sps.csr_matrix, sps.csc_matrix,
                                                        sps.csr_array, sps.csc_array)):
            val = val.tocsr()
        return _orig_validate(self, val, key)
    _am.PairwiseArrays._validate_value = _safe_validate


def main():
    source_name = "visium_human_brain"
    target_name = "merfish_mouse_brain"
    max_cells = 5000

    output_dir = PROJECT_ROOT / "outputs" / f"{source_name}_{target_name}" / "samap_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_config = DATASET_CONFIGS[source_name]
    target_config = DATASET_CONFIGS[target_name]
    source_species = source_config["species"]  # human
    target_species = target_config["species"]  # mouse
    src_prefix = SPECIES_PREFIX[source_species]  # hu
    tgt_prefix = SPECIES_PREFIX[target_species]  # mo

    # Load data
    adata_src_raw = ad.read_h5ad(PROCESSED_DIR / f"{source_name}.h5ad")
    adata_tgt_raw = ad.read_h5ad(PROCESSED_DIR / f"{target_name}.h5ad")
    logger.info("Source (human): %d cells, Target (mouse): %d cells",
                adata_src_raw.n_obs, adata_tgt_raw.n_obs)

    # Ortholog mapping
    ortholog_map = load_ortholog_mapping(source_species, target_species)

    # Build gene homology network
    gnnm, gns, gns_dict = build_gnnm(ortholog_map, source_species, target_species,
                                       adata_src_raw, adata_tgt_raw)

    # Get ortholog gene sets
    src_lookup = _get_gene_lookup(adata_src_raw)
    tgt_lookup = _get_gene_lookup(adata_tgt_raw)
    src_genes = sorted(set(sg for sg, tg in ortholog_map.forward.items()
                           if sg in src_lookup and tg in tgt_lookup))
    tgt_genes = sorted(set(tg for sg, tg in ortholog_map.forward.items()
                           if sg in src_lookup and tg in tgt_lookup))

    # Subsample if needed (same seed as original run)
    rng = np.random.default_rng(42)
    src_idx = np.arange(adata_src_raw.n_obs)
    tgt_idx = np.arange(adata_tgt_raw.n_obs)
    if adata_src_raw.n_obs > max_cells:
        src_idx = rng.choice(adata_src_raw.n_obs, max_cells, replace=False)
        logger.info("Subsampled source (human) to %d cells", max_cells)
    if adata_tgt_raw.n_obs > max_cells:
        tgt_idx = rng.choice(adata_tgt_raw.n_obs, max_cells, replace=False)
        logger.info("Subsampled target (mouse) to %d cells", max_cells)

    adata_src_sub = adata_src_raw[src_idx].copy()
    adata_tgt_sub = adata_tgt_raw[tgt_idx].copy()

    # Prepare AnnData for SAMap
    adata_sam_src = prepare_sam_adata(adata_src_sub, src_prefix, src_genes)
    adata_sam_tgt = prepare_sam_adata(adata_tgt_sub, tgt_prefix, tgt_genes)
    logger.info("SAM AnnData: src=%s, tgt=%s", adata_sam_src.shape, adata_sam_tgt.shape)

    # Run SAMap
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / f"{src_prefix}.h5ad"
        tgt_path = Path(tmpdir) / f"{tgt_prefix}.h5ad"
        adata_sam_src.write_h5ad(src_path)
        adata_sam_tgt.write_h5ad(tgt_path)

        logger.info("Running SAMap alignment...")
        _apply_samap_compat_patches()
        from samap.mapping import SAMAP

        sm = SAMAP(
            sams={src_prefix: str(src_path), tgt_prefix: str(tgt_path)},
            gnnm=(gnnm, gns, gns_dict),
            resolutions={src_prefix: 3.0, tgt_prefix: 3.0},
            save_processed=False,
        )

        samap_result = sm.run(NUMITERS=3, crossK=20, pairwise=True, umap=True)

        adata_combined = samap_result.adata
        logger.info("Combined SAMap result: %s", adata_combined.shape)
        logger.info("obsm keys: %s", list(adata_combined.obsm.keys()))

        # Get SAMap embedding
        if "X_umap" in adata_combined.obsm:
            Z_combined = adata_combined.obsm["X_umap"]
            logger.info("Using UMAP embedding: %s", Z_combined.shape)
        else:
            logger.info("Computing spectral embedding from connectivity graph...")
            from scipy.sparse.linalg import eigsh
            conn = adata_combined.obsp["connectivities"]
            conn_sym = (conn + conn.T) / 2
            degree = np.array(conn_sym.sum(axis=1)).flatten()
            degree[degree == 0] = 1
            D_inv_sqrt = sps.diags(1.0 / np.sqrt(degree))
            L_norm = sps.eye(conn_sym.shape[0]) - D_inv_sqrt @ conn_sym @ D_inv_sqrt
            n_components = 30
            eigenvalues, eigenvectors = eigsh(L_norm, k=n_components + 1, which='SM')
            Z_combined = eigenvectors[:, 1:]
            logger.info("Spectral embedding: %s", Z_combined.shape)

        # Identify species membership
        obs_names = list(adata_combined.obs_names)
        src_mask = np.array([n.startswith(f"{src_prefix}_cell_") for n in obs_names])
        tgt_mask = np.array([n.startswith(f"{tgt_prefix}_cell_") for n in obs_names])
        logger.info("SAMap: %d source (human) cells, %d target (mouse) cells",
                     src_mask.sum(), tgt_mask.sum())

        Z_src = Z_combined[src_mask]  # human embeddings
        Z_tgt = Z_combined[tgt_mask]  # mouse embeddings

        # Cell indices within subsampled arrays
        src_cell_ids = [int(n.split("_")[-1]) for n in np.array(obs_names)[src_mask]]
        tgt_cell_ids = [int(n.split("_")[-1]) for n in np.array(obs_names)[tgt_mask]]

    # Mouse is labeled, human is unlabeled
    Z_labeled = Z_tgt      # mouse
    Z_unlabeled = Z_src     # human
    labeled_cell_ids = tgt_cell_ids
    unlabeled_cell_ids = src_cell_ids

    # ---- Leiden clustering on preprocessed human expression ----
    logger.info("Computing Leiden clustering on preprocessed human expression data...")
    adata_cluster = adata_src_raw.copy()
    sc.pp.filter_cells(adata_cluster, min_genes=1)
    sc.pp.filter_genes(adata_cluster, min_cells=1)
    sc.pp.normalize_total(adata_cluster, target_sum=1e4)
    sc.pp.log1p(adata_cluster)
    sc.pp.highly_variable_genes(adata_cluster, n_top_genes=min(2000, adata_cluster.n_vars))
    adata_cluster = adata_cluster[:, adata_cluster.var.highly_variable].copy()
    sc.pp.pca(adata_cluster, n_comps=min(50, adata_cluster.n_vars - 1))
    sc.pp.neighbors(adata_cluster, n_neighbors=15)

    resolutions = [0.3, 0.5, 1.0]
    for res in resolutions:
        sc.tl.leiden(adata_cluster, resolution=res, key_added=f"leiden_{res}")
        n_clusters = len(adata_cluster.obs[f"leiden_{res}"].unique())
        logger.info("Leiden res=%.1f: %d clusters", res, n_clusters)

    logger.info("Preprocessed human cells: %d, Raw human cells: %d",
                adata_cluster.n_obs, adata_src_raw.n_obs)

    # Build cell index alignment (preprocessed may have fewer cells after filtering)
    raw_ids = list(adata_src_raw.obs_names)
    preproc_ids = list(adata_cluster.obs_names)
    idx_map = {name: i for i, name in enumerate(raw_ids)}
    # keep[i] maps preprocessed cell i -> raw cell index
    keep_raw_indices = [idx_map[n] for n in preproc_ids if n in idx_map]
    # Map from raw index -> preprocessed index
    raw_to_preproc = {}
    for pi, n in enumerate(preproc_ids):
        if n in idx_map:
            raw_to_preproc[idx_map[n]] = pi

    # ---- kNN label transfer in SAMap space ----
    results = {}
    label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_tgt_raw.obs.columns]
    logger.info("Label columns: %s", label_cols)

    for col in label_cols:
        raw_labels = np.array(adata_tgt_raw.obs[col].values, dtype=str)
        # Labels for the subsampled mouse cells present in SAMap output
        labeled_labels = raw_labels[tgt_idx[labeled_cell_ids]]

        # Transfer labels: mouse -> human in SAMap embedding space
        pred, conf = label_transfer_with_confidence(Z_labeled, Z_unlabeled, labeled_labels, k=10)
        logger.info("Label transfer for '%s': %d predictions", col, len(pred))

        # Map predictions to raw human cell indices
        # unlabeled_cell_ids[i] is the index into the subsampled src array
        # src_idx[unlabeled_cell_ids[i]] is the raw cell index
        # We need to align with preprocessed Leiden clusters
        pred_for_preproc = []
        leiden_for_preproc = {res: [] for res in resolutions}
        for i, cell_id in enumerate(unlabeled_cell_ids):
            raw_cell_idx = src_idx[cell_id]
            if raw_cell_idx in raw_to_preproc:
                preproc_idx = raw_to_preproc[raw_cell_idx]
                pred_for_preproc.append(pred[i])
                for res in resolutions:
                    leiden_for_preproc[res].append(
                        adata_cluster.obs[f"leiden_{res}"].values[preproc_idx]
                    )

        pred_aligned = np.array(pred_for_preproc)
        logger.info("  Aligned %d / %d unlabeled cells to preprocessed", len(pred_aligned), len(pred))

        for res in resolutions:
            leiden = np.array(leiden_for_preproc[res], dtype=str)
            ari = adjusted_rand_score(pred_aligned, leiden)
            results[f"samap_{col}_leiden{res}_ari"] = float(ari)
            logger.info("  SAMap %s vs Leiden(res=%.1f): ARI=%.4f", col, res, ari)

    # Find best ARI per label column
    logger.info("\n=== SAMap Leiden ARI Summary (Best Resolution) ===")
    summary = {}
    for col in label_cols:
        best_res = None
        best_ari = -1
        for res in resolutions:
            key = f"samap_{col}_leiden{res}_ari"
            if key in results and results[key] > best_ari:
                best_ari = results[key]
                best_res = res
        summary[f"samap_{col}_best_leiden_ari"] = best_ari
        summary[f"samap_{col}_best_leiden_res"] = best_res
        logger.info("  %s: best ARI=%.4f (res=%.1f)", col, best_ari, best_res)

    results.update(summary)

    # Save results
    out_file = output_dir / "samap_leiden_ari.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_file)


if __name__ == "__main__":
    main()
