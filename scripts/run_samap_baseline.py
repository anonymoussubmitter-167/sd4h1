#!/usr/bin/env python
"""Run SAMap baseline for cross-species alignment benchmarking."""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bridge import DATASET_CONFIGS

from rosetta.bridge.shared_space import _get_gene_lookup
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

SPECIES_PREFIX = {
    "human": "hu",
    "mouse": "mo",
    "zebrafish": "zf",
}


def build_gnnm(ortholog_map, src_species, tgt_species, adata_src, adata_tgt):
    """Build gene-gene homology network (gnnm) from ortholog mapping.

    Returns (gnnm, gns, gns_dict) tuple for SAMap.
    """
    src_prefix = SPECIES_PREFIX[src_species]
    tgt_prefix = SPECIES_PREFIX[tgt_species]

    src_lookup = _get_gene_lookup(adata_src)
    tgt_lookup = _get_gene_lookup(adata_tgt)

    # Get var_names (gene symbols) present in each dataset
    src_genes_in_data = set(src_lookup.keys())
    tgt_genes_in_data = set(tgt_lookup.keys())

    # Build ortholog pairs present in both datasets
    pairs = []
    for sg, tg in ortholog_map.forward.items():
        if sg in src_genes_in_data and tg in tgt_genes_in_data:
            pairs.append((sg, tg))
    logger.info("Found %d ortholog pairs in both datasets", len(pairs))

    # Build gene lists with species prefixes (SAMap convention)
    src_gene_set = sorted(set(p[0] for p in pairs))
    tgt_gene_set = sorted(set(p[1] for p in pairs))
    all_genes = [f"{src_prefix}_{g}" for g in src_gene_set] + [f"{tgt_prefix}_{g}" for g in tgt_gene_set]

    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    n = len(all_genes)

    # Build bipartite adjacency (symmetric)
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

    # SAM/SAMap works best with raw counts or positive expression
    if X.min() < 0:
        X = np.expm1(np.abs(X))
    X = np.maximum(X, 0).astype(np.float32)

    # Prefix gene names for SAMap
    prefixed_names = [f"{species_prefix}_{g}" for g in gene_names]

    adata_sam = ad.AnnData(X=sps.csr_matrix(X))
    adata_sam.var_names = prefixed_names
    adata_sam.obs_names = [f"{species_prefix}_cell_{i}" for i in range(adata_sam.n_obs)]

    return adata_sam


def _apply_samap_compat_patches():
    """Fix SAMap/SAM compatibility with newer scipy/anndata versions.

    1. Re-add `.A` property to sparse matrices (removed in scipy 1.14+)
    2. Auto-convert LIL/COO to CSR when assigning to anndata obsp (anndata 0.10+)
    """
    import anndata._core.aligned_mapping as _am

    # 1. Restore .A property on sparse matrices (dense array shorthand)
    for cls in (sps.csr_matrix, sps.csc_matrix, sps.coo_matrix, sps.lil_matrix,
                sps.csr_array, sps.csc_array, sps.coo_array, sps.lil_array):
        if not hasattr(cls, 'A'):
            cls.A = property(lambda self: self.toarray())

    # 2. Patch anndata obsp setter to accept LIL/COO matrices
    _orig_validate = _am.PairwiseArrays._validate_value
    def _safe_validate(self, val, key=None):
        if sps.issparse(val) and not isinstance(val, (sps.csr_matrix, sps.csc_matrix,
                                                        sps.csr_array, sps.csc_array)):
            val = val.tocsr()
        return _orig_validate(self, val, key)
    _am.PairwiseArrays._validate_value = _safe_validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cells", type=int, default=5000,
                        help="Max cells per species (SAMap is slow on large datasets)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    source_config = DATASET_CONFIGS.get(args.source, {})
    target_config = DATASET_CONFIGS.get(args.target, {})
    source_species = source_config.get("species", "human")
    target_species = target_config.get("species", "mouse")
    src_prefix = SPECIES_PREFIX[source_species]
    tgt_prefix = SPECIES_PREFIX[target_species]

    # Load data
    adata_src_raw = ad.read_h5ad(PROCESSED_DIR / f"{args.source}.h5ad")
    adata_tgt_raw = ad.read_h5ad(PROCESSED_DIR / f"{args.target}.h5ad")
    logger.info("Source: %d cells, Target: %d cells", adata_src_raw.n_obs, adata_tgt_raw.n_obs)

    # Ortholog mapping
    ortholog_map = load_ortholog_mapping(source_species, target_species)

    # Build gene homology network
    gnnm, gns, gns_dict = build_gnnm(ortholog_map, source_species, target_species,
                                       adata_src_raw, adata_tgt_raw)

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

    # Get ortholog gene sets present in data
    src_lookup = _get_gene_lookup(adata_src_raw)
    tgt_lookup = _get_gene_lookup(adata_tgt_raw)
    src_genes = sorted(set(sg for sg, tg in ortholog_map.forward.items()
                           if sg in src_lookup and tg in tgt_lookup))
    tgt_genes = sorted(set(tg for sg, tg in ortholog_map.forward.items()
                           if sg in src_lookup and tg in tgt_lookup))

    # Subsample if needed
    rng = np.random.default_rng(42)
    src_idx = np.arange(adata_src_raw.n_obs)
    tgt_idx = np.arange(adata_tgt_raw.n_obs)
    if adata_src_raw.n_obs > args.max_cells:
        src_idx = rng.choice(adata_src_raw.n_obs, args.max_cells, replace=False)
        logger.info("Subsampled source to %d cells", args.max_cells)
    if adata_tgt_raw.n_obs > args.max_cells:
        tgt_idx = rng.choice(adata_tgt_raw.n_obs, args.max_cells, replace=False)
        logger.info("Subsampled target to %d cells", args.max_cells)

    adata_src_sub = adata_src_raw[src_idx].copy()
    adata_tgt_sub = adata_tgt_raw[tgt_idx].copy()

    # Prepare AnnData for SAMap
    adata_sam_src = prepare_sam_adata(adata_src_sub, src_prefix, src_genes)
    adata_sam_tgt = prepare_sam_adata(adata_tgt_sub, tgt_prefix, tgt_genes)
    logger.info("SAM AnnData: src=%s, tgt=%s", adata_sam_src.shape, adata_sam_tgt.shape)

    # Save to temp h5ad files (SAMap can load from paths)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / f"{src_prefix}.h5ad"
        tgt_path = Path(tmpdir) / f"{tgt_prefix}.h5ad"
        adata_sam_src.write_h5ad(src_path)
        adata_sam_tgt.write_h5ad(tgt_path)

        # Run SAMap
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

        # SAMap alignment lives in the cross-species connectivity graph.
        # Use spectral embedding of the connectivities for kNN-based metrics.
        adata_combined = samap_result.adata
        logger.info("Combined SAMap result: %s", adata_combined.shape)
        logger.info("obsm keys: %s", list(adata_combined.obsm.keys()))

        # Use UMAP embedding if available, else compute spectral embedding from graph
        if "X_umap" in adata_combined.obsm:
            Z_combined = adata_combined.obsm["X_umap"]
            logger.info("Using UMAP embedding: %s", Z_combined.shape)
        else:
            # Compute spectral embedding from cross-species connectivity graph
            logger.info("Computing spectral embedding from connectivity graph...")
            from scipy.sparse.linalg import eigsh
            conn = adata_combined.obsp["connectivities"]
            # Symmetrize and normalize
            conn_sym = (conn + conn.T) / 2
            degree = np.array(conn_sym.sum(axis=1)).flatten()
            degree[degree == 0] = 1
            D_inv_sqrt = sps.diags(1.0 / np.sqrt(degree))
            L_norm = sps.eye(conn_sym.shape[0]) - D_inv_sqrt @ conn_sym @ D_inv_sqrt
            n_components = 30
            eigenvalues, eigenvectors = eigsh(L_norm, k=n_components + 1, which='SM')
            Z_combined = eigenvectors[:, 1:]  # skip first (constant) eigenvector
            logger.info("Spectral embedding: %s", Z_combined.shape)

        # Identify which cells belong to which species
        obs_names = list(adata_combined.obs_names)
        src_mask = np.array([n.startswith(f"{src_prefix}_cell_") for n in obs_names])
        tgt_mask = np.array([n.startswith(f"{tgt_prefix}_cell_") for n in obs_names])
        logger.info("SAMap: %d source cells, %d target cells", src_mask.sum(), tgt_mask.sum())

        Z_src = Z_combined[src_mask]
        Z_tgt = Z_combined[tgt_mask]

        # Map back to original cell indices within subsampled arrays
        src_cell_ids = [int(n.split("_")[-1]) for n in np.array(obs_names)[src_mask]]
        tgt_cell_ids = [int(n.split("_")[-1]) for n in np.array(obs_names)[tgt_mask]]

        # kNN mixing score
        Z_all = np.vstack([Z_src, Z_tgt])
        species = np.array([0] * Z_src.shape[0] + [1] * Z_tgt.shape[0])
        for k in [10, 50, 100]:
            k_eff = min(k, Z_all.shape[0] - 1)
            mix = alignment_score(Z_all, species, k=k_eff)
            results[f"samap_knn_mixing_k{k}"] = float(mix)
            logger.info("SAMap kNN mixing (k=%d): %.4f", k, mix)

        # Label transfer via kNN
        if target_label_cols:
            Z_labeled, Z_unlabeled = Z_tgt, Z_src
            labeled_cell_ids = tgt_cell_ids
            unlabeled_cell_ids = src_cell_ids
            labeled_raw = adata_tgt_raw
            labeled_sub_idx = tgt_idx
            coords = coords_unlabeled_raw[src_idx[unlabeled_cell_ids]]
        else:
            Z_labeled, Z_unlabeled = Z_src, Z_tgt
            labeled_cell_ids = src_cell_ids
            unlabeled_cell_ids = tgt_cell_ids
            labeled_raw = adata_src_raw
            labeled_sub_idx = src_idx
            coords = coords_unlabeled_raw[tgt_idx[unlabeled_cell_ids]]

        for col in label_cols:
            if col not in labeled_raw.obs.columns:
                continue
            raw_labels = np.array(labeled_raw.obs[col].values, dtype=str)
            # Map to subsampled labels
            labeled_labels = raw_labels[labeled_sub_idx[labeled_cell_ids]]

            pred, conf = label_transfer_with_confidence(Z_labeled, Z_unlabeled, labeled_labels, k=10)
            s = spatial_label_smoothness(coords[:len(pred)], pred, k=10)
            results[f"samap_{col}_spatial_smoothness"] = float(s)
            results[f"samap_{col}_mean_confidence"] = float(np.mean(conf))
            logger.info("SAMap %s spatial smoothness: %.4f, confidence: %.4f", col, s, np.mean(conf))

    # Save results
    out_file = output_dir / "samap_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_file)
    for k, v in sorted(results.items()):
        logger.info("  %s: %.4f" if isinstance(v, float) else "  %s: %s", k, v)


if __name__ == "__main__":
    main()
