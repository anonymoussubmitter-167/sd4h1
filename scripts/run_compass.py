#!/usr/bin/env python
"""COMPASS: Conserved Spatial Gene Module Discovery.

Discovers gene modules conserved across species using joint NMF on shared
ortholog gene space, then scores conservation via BRIDGE transport plans
and aligned embeddings.

Usage:
    python scripts/run_compass.py \
        --source visium_human_brain --target merfish_mouse_brain \
        [--k-values 5 8 10 15 20] [--n-runs 10]

Output directory: outputs/<source>_<target>/compass/
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
import matplotlib.pyplot as plt
import numpy as np

from rosetta.compass.conservation import (
    match_independent_modules,
    module_celltype_enrichment,
    module_conservation_knn,
    module_conservation_region,
    module_conservation_transport,
    module_spatial_autocorrelation,
)
from rosetta.compass.enrichment import module_go_enrichment
from rosetta.compass.nmf import (
    build_nonneg_shared_space,
    characterize_modules,
    consensus_nmf,
    run_joint_nmf,
)
from rosetta.data.ortholog_db import load_ortholog_mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bridge import DATASET_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

KNOWN_LABEL_COLUMNS = [
    "class", "subclass", "neurotransmitter", "supertype", "cluster",
    "Cell_Type", "cell_type", "celltype",
]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def plot_module_spatial_maps(
    W_source: np.ndarray,
    W_target: np.ndarray,
    coords_source: np.ndarray,
    coords_target: np.ndarray,
    k: int,
    source_name: str,
    target_name: str,
    out_dir: Path,
) -> None:
    """Plot side-by-side spatial activity maps for each module."""
    n_modules = W_source.shape[1]
    n_cols = 2
    n_rows = min(n_modules, 10)  # cap at 10 modules per figure

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for m in range(n_rows):
        # Source
        sc0 = axes[m, 0].scatter(
            coords_source[:, 0], coords_source[:, 1],
            c=W_source[:, m], cmap="viridis", s=3, alpha=0.8, rasterized=True,
        )
        axes[m, 0].set_title(f"Module {m} — {source_name}", fontsize=9)
        axes[m, 0].set_aspect("equal")
        axes[m, 0].set_xticks([])
        axes[m, 0].set_yticks([])
        fig.colorbar(sc0, ax=axes[m, 0], shrink=0.7)

        # Target
        sc1 = axes[m, 1].scatter(
            coords_target[:, 0], coords_target[:, 1],
            c=W_target[:, m], cmap="viridis", s=1, alpha=0.6, rasterized=True,
        )
        axes[m, 1].set_title(f"Module {m} — {target_name}", fontsize=9)
        axes[m, 1].set_aspect("equal")
        axes[m, 1].set_xticks([])
        axes[m, 1].set_yticks([])
        fig.colorbar(sc1, ax=axes[m, 1], shrink=0.7)

    fig.suptitle(f"Module Spatial Activity (k={k})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"spatial_modules_k{k}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_conservation_bars(
    conservation_transport: np.ndarray,
    conservation_knn: np.ndarray,
    conservation_region: list | np.ndarray | None,
    k: int,
    out_dir: Path,
) -> None:
    """Bar chart of per-module conservation scores."""
    n_modules = len(conservation_transport)
    x = np.arange(n_modules)
    n_bars = 3 if conservation_region is not None else 2
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(6, n_modules * 0.8), 5))
    ax.bar(x - width, conservation_transport, width, label="Transport", color="steelblue")
    ax.bar(x, conservation_knn, width, label="kNN", color="coral")
    if conservation_region is not None:
        region_arr = np.array(conservation_region)
        ax.bar(x + width, region_arr, width, label="Region", color="forestgreen")
    ax.set_xlabel("Module")
    ax.set_ylabel("Conservation Score")
    ax.set_title(f"Module Conservation Scores (k={k})")
    ax.set_xticks(x)
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / f"conservation_k{k}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gene_loading_heatmap(
    modules_info: list[dict],
    k: int,
    out_dir: Path,
    top_n: int = 15,
) -> None:
    """Heatmap of top gene loadings per module."""
    n_modules = len(modules_info)
    # Collect unique top genes across all modules
    all_genes = []
    for info in modules_info:
        all_genes.extend(info["genes"][:top_n])
    unique_genes = list(dict.fromkeys(all_genes))  # preserve order, deduplicate

    # Build loading matrix
    gene_to_col = {g: i for i, g in enumerate(unique_genes)}
    mat = np.zeros((n_modules, len(unique_genes)))
    for info in modules_info:
        m = info["module"]
        for gene, weight in zip(info["genes"][:top_n], info["weights"][:top_n]):
            mat[m, gene_to_col[gene]] = weight

    fig, ax = plt.subplots(figsize=(max(10, len(unique_genes) * 0.4), max(4, n_modules * 0.5)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(n_modules))
    ax.set_yticklabels([f"Module {i}" for i in range(n_modules)], fontsize=8)
    ax.set_xticks(range(len(unique_genes)))
    ax.set_xticklabels(unique_genes, rotation=90, fontsize=6, ha="center")
    ax.set_title(f"Gene Loadings (k={k})")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / f"gene_loadings_k{k}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_celltype_enrichment_heatmap(
    enrichment_df,
    k: int,
    out_dir: Path,
) -> None:
    """Heatmap of mean module activity per cell type."""
    pivot = enrichment_df.pivot_table(
        index="module", columns="cell_type", values="mean_activity",
    )

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.6), max(4, pivot.shape[0] * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"Module {i}" for i in pivot.index], fontsize=8)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Cell Type Enrichment (k={k})")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / f"celltype_enrichment_k{k}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stability_vs_k(all_results: dict, out_dir: Path) -> None:
    """Line plot of stability and mean conservation vs k."""
    k_values = sorted(all_results.keys())
    stabilities = [all_results[k]["stability"] for k in k_values]
    mean_cons_t = [np.mean(all_results[k]["conservation_transport"]) for k in k_values]
    mean_cons_k = [np.mean(all_results[k]["conservation_knn"]) for k in k_values]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(k_values, stabilities, "o-", color="steelblue", label="Stability")
    ax1.set_xlabel("k (number of modules)")
    ax1.set_ylabel("Stability", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(k_values, mean_cons_t, "s--", color="coral", label="Conservation (transport)")
    ax2.plot(k_values, mean_cons_k, "^--", color="green", label="Conservation (kNN)")
    ax2.set_ylabel("Mean Conservation (r)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    ax1.set_title("Stability & Conservation vs Number of Modules")
    fig.tight_layout()
    fig.savefig(out_dir / "stability_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="COMPASS: Conserved Gene Module Discovery")
    parser.add_argument("--source", type=str, required=True,
                        help="Source dataset name (e.g., visium_human_brain)")
    parser.add_argument("--target", type=str, required=True,
                        help="Target dataset name (e.g., merfish_mouse_brain)")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 8, 10, 15, 20],
                        help="Number of modules to try")
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Number of consensus NMF runs per k")
    parser.add_argument("--max-iter", type=int, default=500,
                        help="Max NMF iterations")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip permuted and independent NMF baselines")
    parser.add_argument("--skip-go", action="store_true",
                        help="Skip GO enrichment (requires internet)")
    parser.add_argument("--n-regions", type=int, default=20,
                        help="Number of spatial regions for region-level conservation")
    args = parser.parse_args()

    for name in [args.source, args.target]:
        if name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {name}")
            print(f"Available: {sorted(DATASET_CONFIGS.keys())}")
            sys.exit(1)

    source_config = DATASET_CONFIGS[args.source]
    target_config = DATASET_CONFIGS[args.target]
    source_species = source_config["species"]
    target_species = target_config["species"]

    pair_name = f"{args.source}_{args.target}"
    base_output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / pair_name
    compass_dir = base_output_dir / "compass"
    compass_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------
    logger.info("Loading datasets...")
    source_h5ad = PROCESSED_DIR / f"{args.source}.h5ad"
    target_h5ad = PROCESSED_DIR / f"{args.target}.h5ad"

    adata_source = ad.read_h5ad(source_h5ad)
    adata_target = ad.read_h5ad(target_h5ad)
    logger.info("Source: %d cells, %d genes", adata_source.n_obs, adata_source.n_vars)
    logger.info("Target: %d cells, %d genes", adata_target.n_obs, adata_target.n_vars)

    # Ortholog mapping
    ortholog_map = load_ortholog_mapping(source_species, target_species)

    # Build non-negative shared space
    logger.info("Building non-negative shared gene space...")
    X_source, X_target, gene_names = build_nonneg_shared_space(
        adata_source, adata_target, ortholog_map,
    )
    logger.info("Shared genes: %d, Source: %s, Target: %s",
                len(gene_names), X_source.shape, X_target.shape)

    if len(gene_names) == 0:
        print("No shared ortholog genes found. Cannot run COMPASS.")
        sys.exit(1)

    # Load BRIDGE outputs
    eval_dir = base_output_dir / "evaluation"
    z_source_path = eval_dir / "z_source_proj.npy"
    z_target_path = eval_dir / "z_target_proj.npy"
    T_path = base_output_dir / "transport_plan.npy"

    z_source = np.load(z_source_path) if z_source_path.exists() else None
    z_target = np.load(z_target_path) if z_target_path.exists() else None
    transport_plan = np.load(T_path) if T_path.exists() else None

    if z_source is not None:
        logger.info("Loaded projected embeddings: source %s, target %s",
                     z_source.shape, z_target.shape)
    if transport_plan is not None:
        logger.info("Loaded transport plan: %s", transport_plan.shape)

    # Spatial coordinates
    coords_source = np.array(adata_source.obsm["spatial"][:, :2], dtype=np.float32)
    coords_target = np.array(adata_target.obsm["spatial"][:, :2], dtype=np.float32)

    # Cell type labels (for enrichment analysis)
    target_label_cols = [c for c in KNOWN_LABEL_COLUMNS if c in adata_target.obs.columns]
    target_labels = None
    target_label_col = None
    if target_label_cols:
        target_label_col = target_label_cols[0]
        target_labels = np.array(adata_target.obs[target_label_col].values, dtype=str)
        logger.info("Target labels: %s (%d classes)", target_label_col,
                     len(np.unique(target_labels)))

    # -------------------------------------------------------------------
    # 2. NMF sweep
    # -------------------------------------------------------------------
    all_results: dict[int, dict] = {}

    for k in args.k_values:
        logger.info("=" * 60)
        logger.info("Running consensus NMF with k=%d (n_runs=%d)", k, args.n_runs)
        logger.info("=" * 60)

        W_s, W_t, H, stability = consensus_nmf(
            X_source, X_target, k,
            n_runs=args.n_runs, max_iter=args.max_iter,
        )

        # Characterize modules
        modules_info = characterize_modules(H, gene_names)
        for info in modules_info:
            top5 = ", ".join(info["genes"][:5])
            logger.info("  Module %d top genes: %s", info["module"], top5)

        result = {
            "k": k,
            "stability": stability,
            "modules": modules_info,
        }

        # Conservation via transport plan
        if transport_plan is not None:
            n_s_T, n_t_T = transport_plan.shape
            cons_t = module_conservation_transport(
                W_s[:n_s_T], W_t[:n_t_T], transport_plan,
            )
            result["conservation_transport"] = cons_t.tolist()
            logger.info("  Conservation (transport): %s",
                        ", ".join(f"{c:.3f}" for c in cons_t))

        # Conservation via kNN
        # NMF uses raw adata (may have more cells than preprocessed embeddings)
        if z_source is not None and z_target is not None:
            n_s_emb = z_source.shape[0]
            n_t_emb = z_target.shape[0]
            cons_k = module_conservation_knn(
                W_s[:n_s_emb], W_t[:n_t_emb], z_source, z_target, k=10,
            )
            result["conservation_knn"] = cons_k.tolist()
            logger.info("  Conservation (kNN): %s",
                        ", ".join(f"{c:.3f}" for c in cons_k))

        # Region-level conservation
        if z_source is not None and z_target is not None:
            n_s_emb = z_source.shape[0]
            n_t_emb = z_target.shape[0]
            cons_region = module_conservation_region(
                W_s[:n_s_emb], W_t[:n_t_emb],
                coords_source[:n_s_emb], coords_target[:n_t_emb],
                z_source, z_target,
                n_regions=args.n_regions,
            )
            result["conservation_region"] = cons_region.tolist()
            logger.info("  Conservation (region): %s",
                        ", ".join(f"{c:.3f}" for c in cons_region))

        # Spatial autocorrelation
        morans_source = module_spatial_autocorrelation(W_s, coords_source, k=10)
        morans_target = module_spatial_autocorrelation(W_t, coords_target, k=10)
        result["morans_i_source"] = morans_source.tolist()
        result["morans_i_target"] = morans_target.tolist()
        logger.info("  Moran's I source: %s",
                     ", ".join(f"{m:.3f}" for m in morans_source))
        logger.info("  Moran's I target: %s",
                     ", ".join(f"{m:.3f}" for m in morans_target))

        # Cell type enrichment
        if target_labels is not None:
            enrichment_df = module_celltype_enrichment(W_t, target_labels)
            result["celltype_enrichment"] = enrichment_df.to_dict(orient="records")

        # GO enrichment
        if not args.skip_go:
            logger.info("  Running GO enrichment for k=%d...", k)
            go_results = module_go_enrichment(H, gene_names, top_n=50)
            go_summary = []
            for m, df in enumerate(go_results):
                if len(df) > 0:
                    top_terms = df.head(5)["Term"].tolist()
                    go_summary.append({
                        "module": m,
                        "n_significant": len(df),
                        "top_terms": top_terms,
                    })
                else:
                    go_summary.append({"module": m, "n_significant": 0, "top_terms": []})
            result["go_enrichment"] = go_summary

            # Save full GO results
            for m, df in enumerate(go_results):
                if len(df) > 0:
                    df.to_csv(compass_dir / f"go_module{m}_k{k}.csv", index=False)

        all_results[k] = result

        # Save W matrices
        np.save(compass_dir / f"module_activity_source_k{k}.npy", W_s)
        np.save(compass_dir / f"module_activity_target_k{k}.npy", W_t)

        # Save module info
        with open(compass_dir / f"modules_k{k}.json", "w") as f:
            json.dump(modules_info, f, indent=2)

        # Visualizations for this k
        plot_module_spatial_maps(
            W_s, W_t, coords_source, coords_target, k,
            args.source, args.target, compass_dir,
        )

        if "conservation_transport" in result and "conservation_knn" in result:
            plot_conservation_bars(
                np.array(result["conservation_transport"]),
                np.array(result["conservation_knn"]),
                result.get("conservation_region"),
                k, compass_dir,
            )

        plot_gene_loading_heatmap(modules_info, k, compass_dir)

        if target_labels is not None:
            plot_celltype_enrichment_heatmap(enrichment_df, k, compass_dir)

    # -------------------------------------------------------------------
    # 3. Select best k
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Selecting best k...")

    best_k = None
    best_score = -np.inf
    for k, res in all_results.items():
        # Prefer region-level conservation (more robust), fallback to kNN
        if "conservation_region" in res:
            mean_cons = np.mean(res["conservation_region"])
        elif "conservation_knn" in res:
            mean_cons = np.mean(res["conservation_knn"])
        elif "conservation_transport" in res:
            mean_cons = np.mean(res["conservation_transport"])
        else:
            mean_cons = 0.0
        combined = mean_cons * res["stability"]
        logger.info("  k=%d: stability=%.3f, mean_conservation=%.3f, combined=%.4f",
                     k, res["stability"], mean_cons, combined)
        if combined > best_score:
            best_score = combined
            best_k = k

    logger.info("Best k=%d (combined score=%.4f)", best_k, best_score)

    # -------------------------------------------------------------------
    # 4. Baselines
    # -------------------------------------------------------------------
    if not args.skip_baselines and best_k is not None:
        logger.info("=" * 60)
        logger.info("Running baselines for k=%d...", best_k)

        baseline_results = {}

        # 4a. Permuted baseline: shuffle rows of target expression
        logger.info("  Permuted baseline (5 permutations)...")
        perm_conservation = []
        rng = np.random.default_rng(42)
        for seed in range(5):
            perm_idx = rng.permutation(X_target.shape[0])
            X_target_perm = X_target[perm_idx]
            W_s_perm, W_t_perm, H_perm = run_joint_nmf(
                X_source, X_target_perm, best_k, random_state=seed,
            )
            if z_source is not None and z_target is not None:
                n_s_emb = z_source.shape[0]
                n_t_emb = z_target.shape[0]
                cons_perm = module_conservation_knn(
                    W_s_perm[:n_s_emb], W_t_perm[:n_t_emb],
                    z_source, z_target, k=10,
                )
                perm_conservation.append(np.mean(cons_perm))

        if perm_conservation:
            baseline_results["permuted_mean_conservation"] = float(np.mean(perm_conservation))
            baseline_results["permuted_std_conservation"] = float(np.std(perm_conservation))
            logger.info("  Permuted conservation: %.4f +/- %.4f",
                         np.mean(perm_conservation), np.std(perm_conservation))

        # 4b. Independent NMF + BRIDGE matching baseline
        logger.info("  Independent NMF + BRIDGE matching baseline...")
        from sklearn.decomposition import NMF as _NMF

        model_s = _NMF(n_components=best_k, init="nndsvda", max_iter=500, random_state=42)
        W_s_indep = model_s.fit_transform(X_source)
        H_s_indep = model_s.components_

        model_t = _NMF(n_components=best_k, init="nndsvda", max_iter=500, random_state=42)
        W_t_indep = model_t.fit_transform(X_target)
        H_t_indep = model_t.components_

        # Match modules using the new match_independent_modules function
        if z_source is not None and z_target is not None:
            n_s_emb = z_source.shape[0]
            n_t_emb = z_target.shape[0]
            match_result = match_independent_modules(
                H_s_indep, H_t_indep,
                W_s_indep[:n_s_emb], W_t_indep[:n_t_emb],
                z_source, z_target,
            )
        else:
            match_result = match_independent_modules(H_s_indep, H_t_indep)

        baseline_results["independent_nmf_mean_gene_corr"] = match_result["mean_gene_corr"]
        baseline_results["independent_nmf_gene_corrs"] = [m[2] for m in match_result["matches"]]
        baseline_results["independent_nmf_matches"] = [
            {"source": m[0], "target": m[1], "gene_corr": m[2]}
            for m in match_result["matches"]
        ]
        if match_result["embedding_corr"] is not None:
            baseline_results["independent_nmf_embedding_corr"] = match_result["embedding_corr"]
        logger.info("  Independent NMF gene loading correlation: %.4f",
                     match_result["mean_gene_corr"])
        if match_result["embedding_corr"] is not None:
            logger.info("  Independent NMF embedding correlation: %.4f",
                         match_result["embedding_corr"])

        all_results["baselines"] = baseline_results

    # -------------------------------------------------------------------
    # 5. Stability vs k plot
    # -------------------------------------------------------------------
    numeric_results = {k: v for k, v in all_results.items() if isinstance(k, int)}
    if len(numeric_results) > 1:
        plot_stability_vs_k(numeric_results, compass_dir)

    # -------------------------------------------------------------------
    # 6. Save all results
    # -------------------------------------------------------------------
    # Convert numpy arrays to lists for JSON serialization
    def _make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        return obj

    results_path = compass_dir / "compass_results.json"
    with open(results_path, "w") as f:
        json.dump(_make_serializable(all_results), f, indent=2)
    logger.info("All results saved to %s", results_path)

    # -------------------------------------------------------------------
    # 7. Summary
    # -------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"COMPASS Results: {args.source} <-> {args.target}")
    print(f"{'=' * 70}")
    print(f"Shared ortholog genes: {len(gene_names)}")
    print(f"Best k: {best_k}")
    print()

    for k in sorted(numeric_results.keys()):
        res = numeric_results[k]
        marker = " <-- BEST" if k == best_k else ""
        parts = [f"stability={res['stability']:.3f}"]
        if "conservation_knn" in res:
            parts.append(f"kNN={np.mean(res['conservation_knn']):.3f}")
        if "conservation_region" in res:
            parts.append(f"region={np.mean(res['conservation_region']):.3f}")
        print(f"  k={k}: {', '.join(parts)}{marker}")

    if best_k in numeric_results:
        best_res = numeric_results[best_k]
        print(f"\n--- Best k={best_k} Module Details ---")
        for info in best_res["modules"]:
            top5 = ", ".join(info["genes"][:5])
            print(f"  Module {info['module']}: {top5}")

        if "conservation_region" in best_res:
            print(f"\n  Conservation (region): {', '.join(f'{c:.3f}' for c in best_res['conservation_region'])}")

        if "morans_i_source" in best_res:
            print(f"\n  Moran's I (source): {', '.join(f'{m:.3f}' for m in best_res['morans_i_source'])}")
            print(f"  Moran's I (target): {', '.join(f'{m:.3f}' for m in best_res['morans_i_target'])}")

        if "go_enrichment" in best_res:
            print(f"\n--- GO Enrichment (top term per module) ---")
            for go in best_res["go_enrichment"]:
                if go["top_terms"]:
                    print(f"  Module {go['module']}: {go['top_terms'][0]} ({go['n_significant']} terms)")
                else:
                    print(f"  Module {go['module']}: no significant terms")

    if "baselines" in all_results:
        bl = all_results["baselines"]
        print("\n--- Baselines ---")
        if "permuted_mean_conservation" in bl:
            print(f"  Permuted: {bl['permuted_mean_conservation']:.4f} +/- {bl['permuted_std_conservation']:.4f}")
        if "independent_nmf_mean_gene_corr" in bl:
            print(f"  Independent NMF gene corr: {bl['independent_nmf_mean_gene_corr']:.4f}")

    print(f"\n{'=' * 70}")
    print(f"Outputs saved to: {compass_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
