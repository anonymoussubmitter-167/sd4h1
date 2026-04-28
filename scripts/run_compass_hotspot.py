#!/usr/bin/env python
"""COMPASS baseline: Hotspot spatial gene autocorrelation.

Compares COMPASS joint NMF modules against Hotspot's spatial gene
identification. Hotspot finds genes with significant spatial autocorrelation
independently per species, while COMPASS finds jointly conserved modules.

Usage:
    python scripts/run_compass_hotspot.py \
        --source visium_human_brain --target merfish_mouse_brain
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
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bridge import DATASET_CONFIGS

from rosetta.data.ortholog_db import load_ortholog_mapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def run_hotspot_on_adata(adata, species_name, n_neighbors=30):
    """Run Hotspot spatial autocorrelation analysis on an AnnData."""
    import hotspot

    adata = adata.copy()

    # Ensure counts layer exists
    if "counts" not in adata.layers:
        X = adata.X
        if sp.issparse(X):
            X = np.array(X.todense())
        # If data looks log-transformed, reverse it
        if X.min() >= 0 and X.max() < 20:
            adata.layers["counts"] = np.round(np.expm1(X)).astype(np.float32)
        else:
            adata.layers["counts"] = np.maximum(X, 0).astype(np.float32)

    # Compute total counts for normalization
    counts = adata.layers["counts"]
    if sp.issparse(counts):
        counts = np.array(counts.todense())
    adata.obs["total_counts"] = np.array(counts.sum(axis=1)).flatten()

    # Build spatial neighbors
    coords = np.array(adata.obsm["spatial"][:, :2], dtype=np.float64)

    # Subsample if too large (Hotspot can be slow)
    max_cells = 10000
    if adata.n_obs > max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx].copy()
        coords = coords[idx]

    # Filter zero-variance genes (Hotspot requirement)
    counts_mat = adata.layers["counts"]
    if sp.issparse(counts_mat):
        counts_mat = np.array(counts_mat.todense())
    gene_var = np.var(counts_mat, axis=0)
    nonzero_mask = gene_var > 0
    if nonzero_mask.sum() < adata.n_vars:
        logger.info("  Filtering %d zero-variance genes", adata.n_vars - nonzero_mask.sum())
        adata = adata[:, nonzero_mask].copy()

    logger.info("Running Hotspot on %s (%d cells, %d genes)...",
                species_name, adata.n_obs, adata.n_vars)

    hs = hotspot.Hotspot(
        adata,
        layer_key="counts",
        model="normal",
        latent_obsm_key="spatial",
    )

    hs.create_knn_graph(weighted_graph=False, n_neighbors=n_neighbors)
    hs_results = hs.compute_autocorrelations()

    # Get significant spatially variable genes
    sig_genes = hs_results[hs_results["FDR"] < 0.05].sort_values("Z", ascending=False)
    all_genes = hs_results.sort_values("Z", ascending=False)

    logger.info("  %s: %d/%d genes spatially variable (FDR<0.05)",
                species_name, len(sig_genes), len(all_genes))

    # If var_names are Ensembl IDs, convert to gene symbols for comparison
    ensembl_to_symbol = {}
    if "gene_symbol" in adata.var.columns:
        for ens, sym in zip(adata.var_names, adata.var["gene_symbol"].values):
            if sym:
                ensembl_to_symbol[str(ens)] = str(sym)

    def _to_symbol(gene_list):
        return [ensembl_to_symbol.get(str(g), str(g)) for g in gene_list]

    top_genes_raw = sig_genes.head(50).index.tolist() if len(sig_genes) > 0 else []
    top_genes_sym = _to_symbol(top_genes_raw)

    if len(sig_genes) > 0:
        top10 = top_genes_sym[:10]
        logger.info("  Top 10: %s", ", ".join(str(g) for g in top10))

    return {
        "n_significant": len(sig_genes),
        "n_total": len(all_genes),
        "top_genes": top_genes_sym,
        "top_z_scores": sig_genes.head(50)["Z"].tolist() if len(sig_genes) > 0 else [],
        "all_results": all_genes,
    }


def compare_with_compass(hotspot_src, hotspot_tgt, compass_dir, ortholog_map):
    """Compare Hotspot genes with COMPASS module genes."""
    results = {}

    # Load COMPASS modules
    compass_results_path = compass_dir / "compass_results.json"
    if not compass_results_path.exists():
        logger.warning("COMPASS results not found at %s", compass_results_path)
        return results

    with open(compass_results_path) as f:
        compass_data = json.load(f)

    # Find best k
    best_k = None
    for k_str, data in compass_data.items():
        if k_str == "baselines":
            continue
        if isinstance(data, dict) and "modules" in data:
            if best_k is None or (isinstance(k_str, str) and k_str.isdigit()):
                best_k = k_str

    if best_k is None:
        logger.warning("No COMPASS modules found")
        return results

    modules = compass_data[best_k]["modules"]
    compass_genes = set()
    for mod in modules:
        compass_genes.update(mod["genes"][:10])

    # Hotspot spatially variable genes
    hs_src_genes = set(str(g) for g in hotspot_src["top_genes"][:50])
    hs_tgt_genes_raw = [str(g) for g in hotspot_tgt["top_genes"][:50]]

    # If target genes are Ensembl IDs, try to map via gene_symbol in adata
    # (passed as extra field in hotspot results)
    hs_tgt_genes = set(hs_tgt_genes_raw)

    # Map target genes to source namespace via orthologs
    # Build Ensembl->symbol lookup if target uses Ensembl IDs
    reverse_map = ortholog_map.reverse
    hs_tgt_mapped = set()
    for g in hs_tgt_genes:
        # Try direct lookup first
        if g in reverse_map:
            hs_tgt_mapped.add(reverse_map[g])
        else:
            # Keep original (unmappable)
            hs_tgt_mapped.add(g)

    # Overlap: genes found by both Hotspot (in either species) and COMPASS
    hs_union = hs_src_genes | hs_tgt_mapped
    hs_intersection = hs_src_genes & hs_tgt_mapped

    overlap_compass_union = compass_genes & hs_union
    overlap_compass_intersection = compass_genes & hs_intersection

    results["hotspot_source_n_sig"] = hotspot_src["n_significant"]
    results["hotspot_target_n_sig"] = hotspot_tgt["n_significant"]
    results["hotspot_source_top50"] = list(hs_src_genes)
    results["hotspot_target_top50"] = list(hs_tgt_genes)
    results["hotspot_shared_spatial_genes"] = list(hs_intersection)
    results["compass_genes_top10"] = list(compass_genes)
    results["overlap_compass_hotspot_union"] = list(overlap_compass_union)
    results["overlap_compass_hotspot_intersection"] = list(overlap_compass_intersection)
    results["n_compass_genes"] = len(compass_genes)
    results["n_hotspot_union"] = len(hs_union)
    results["n_hotspot_intersection"] = len(hs_intersection)
    results["n_overlap_union"] = len(overlap_compass_union)
    results["n_overlap_intersection"] = len(overlap_compass_intersection)

    # Jaccard similarity
    if len(compass_genes | hs_union) > 0:
        jaccard = len(overlap_compass_union) / len(compass_genes | hs_union)
        results["jaccard_compass_hotspot"] = float(jaccard)
        logger.info("  COMPASS vs Hotspot Jaccard: %.3f", jaccard)

    logger.info("  COMPASS modules use %d genes (top 10/module)", len(compass_genes))
    logger.info("  Hotspot: %d source, %d target spatially variable",
                hotspot_src["n_significant"], hotspot_tgt["n_significant"])
    logger.info("  Shared spatial genes (both species): %d", len(hs_intersection))
    logger.info("  COMPASS ∩ Hotspot(union): %d/%d", len(overlap_compass_union), len(compass_genes))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    source_config = DATASET_CONFIGS[args.source]
    target_config = DATASET_CONFIGS[args.target]
    source_species = source_config["species"]
    target_species = target_config["species"]

    pair_name = f"{args.source}_{args.target}"
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / pair_name / "compass"
    output_dir.mkdir(parents=True, exist_ok=True)

    adata_src = ad.read_h5ad(PROCESSED_DIR / f"{args.source}.h5ad")
    adata_tgt = ad.read_h5ad(PROCESSED_DIR / f"{args.target}.h5ad")
    ortholog_map = load_ortholog_mapping(source_species, target_species)

    # Run Hotspot per species
    logger.info("=" * 60)
    hotspot_src = run_hotspot_on_adata(adata_src, args.source)
    logger.info("=" * 60)
    hotspot_tgt = run_hotspot_on_adata(adata_tgt, args.target)

    # Compare with COMPASS
    logger.info("=" * 60)
    logger.info("Comparing with COMPASS modules...")
    compass_dir = output_dir  # compass results should be in same dir
    comparison = compare_with_compass(hotspot_src, hotspot_tgt, compass_dir, ortholog_map)

    # Save
    all_results = {
        "source_hotspot": {
            "n_significant": hotspot_src["n_significant"],
            "n_total": hotspot_src["n_total"],
            "top_genes": hotspot_src["top_genes"],
            "top_z_scores": hotspot_src["top_z_scores"],
        },
        "target_hotspot": {
            "n_significant": hotspot_tgt["n_significant"],
            "n_total": hotspot_tgt["n_total"],
            "top_genes": hotspot_tgt["top_genes"],
            "top_z_scores": hotspot_tgt["top_z_scores"],
        },
        "comparison": comparison,
    }

    out_file = output_dir / "hotspot_baseline.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_file)

    print(f"\n{'='*60}")
    print(f"Hotspot Baseline: {args.source} <-> {args.target}")
    print(f"{'='*60}")
    print(f"  Source ({args.source}): {hotspot_src['n_significant']} spatially variable genes")
    print(f"  Target ({args.target}): {hotspot_tgt['n_significant']} spatially variable genes")
    if comparison:
        print(f"  COMPASS genes: {comparison.get('n_compass_genes', 'N/A')}")
        print(f"  Hotspot shared (both species): {comparison.get('n_hotspot_intersection', 'N/A')}")
        print(f"  Overlap COMPASS∩Hotspot: {comparison.get('n_overlap_union', 'N/A')}")
        if "jaccard_compass_hotspot" in comparison:
            print(f"  Jaccard similarity: {comparison['jaccard_compass_hotspot']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
