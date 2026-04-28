#!/usr/bin/env python
"""Biological validation of ROSETTA cross-species label transfer via marker gene enrichment.

For each transferred cell type, checks whether canonical marker genes are
significantly upregulated in the assigned spots, using Wilcoxon rank-sum tests
and log2 fold-change thresholds.

Reports two tiers of enrichment:
  - Strict: p < 0.05 AND log2FC > 0.5 (standard DE criteria)
  - Relaxed: p < 0.05 AND log2FC > 0 (significant upregulation of any magnitude)

The relaxed tier is important for spatial transcriptomics (Visium, Stereo-seq)
where each spot/bin captures a mixture of cell types, attenuating fold-changes
compared to single-cell data.

Validates:
  1. Mouse -> Human brain (MERFISH -> Visium)
  2. Mouse -> Zebrafish brain (MERFISH -> Stereo-seq)

Usage:
    python scripts/biological_validation.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Canonical marker genes per cell type
# ---------------------------------------------------------------------------

# Human gene symbols (uppercase)
MARKERS_HUMAN = {
    "GABAergic": ["GAD1", "GAD2", "SLC32A1"],
    "Glutamatergic": ["SLC17A7", "SLC17A6"],
    "Oligodendrocyte": ["MBP", "MOG", "OPALIN", "SOX10"],
    "OPC": ["PDGFRA", "CSPG4"],
    "Astrocyte": ["GFAP", "AQP4", "SLC1A3"],
    "Vascular": ["CLDN5", "FLT1", "PECAM1"],
    "Immune": ["CX3CR1", "P2RY12", "TMEM119"],
}

# Zebrafish gene symbols (lowercase with paralog suffixes)
MARKERS_ZEBRAFISH = {
    "GABAergic": ["gad1b", "gad2", "slc32a1"],
    "Glutamatergic": ["slc17a6b"],  # slc17a7 not in zebrafish panel
    "Oligodendrocyte": ["mbpa", "mbpb", "sox10"],  # mog, opalin missing
    "OPC": ["pdgfra", "cspg4"],
    "Astrocyte": ["gfap", "aqp4", "slc1a3a", "slc1a3b"],
    "Vascular": ["cldn5a", "cldn5b", "flt1", "pecam1"],
    "Immune": ["p2ry12"],  # cx3cr1, tmem119 missing
}

# Map transferred class labels to marker categories
# The transferred labels use Allen Brain Atlas class names
LABEL_TO_MARKER_CATEGORY = {
    "27 MY GABA": "GABAergic",
    "28 CB GABA": "GABAergic",
    "24 MY Glut": "Glutamatergic",
    "29 CB Glut": "Glutamatergic",
    "31 OPC-Oligo": "Oligodendrocyte",  # mixed OPC + oligo class
    "30 Astro-Epen": "Astrocyte",
    "33 Vascular": "Vascular",
    "34 Immune": "Immune",
}

# For OPC-Oligo class, also check OPC markers
LABEL_TO_EXTRA_CATEGORIES = {
    "31 OPC-Oligo": ["OPC"],
}


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def load_expression_with_labels(
    h5ad_path: Path,
    labels_path: Path,
    min_genes: int = 200,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load expression data and align with transferred labels.

    Returns:
        X: (n_cells, n_genes) expression matrix (log-normalized)
        labels: (n_cells,) transferred cell type labels
        gene_names: list of gene names
        confidences: (n_cells,) confidence scores
    """
    adata = sc.read_h5ad(h5ad_path)

    # Apply same filtering as evaluate_bridge.py
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # Load transferred labels
    npz = np.load(labels_path, allow_pickle=True)
    labels = npz["labels"]
    confidences = npz["confidences"]

    assert len(labels) == adata.n_obs, (
        f"Label count mismatch: {len(labels)} labels vs {adata.n_obs} cells"
    )

    # Get expression matrix
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)

    # Check if data needs log-normalization (raw counts have large max values)
    if X.max() > 50:
        # Raw counts: normalize and log-transform
        # Normalize per cell to target_sum=1e4, then log1p
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X = X / row_sums * 1e4
        X = np.log1p(X)

    gene_names = list(adata.var_names)
    return X, labels, gene_names, confidences


def compute_marker_enrichment(
    X: np.ndarray,
    labels: np.ndarray,
    gene_names: list[str],
    marker_dict: dict[str, list[str]],
    label_to_category: dict[str, str],
    label_to_extra: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Compute differential expression statistics for canonical markers.

    For each transferred cell type, test whether each canonical marker gene
    is upregulated in spots assigned to that type vs all other spots.

    Returns a list of result dicts, one per (cell_type, gene) pair.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    unique_labels = np.unique(labels)
    results = []

    for label in sorted(unique_labels):
        category = label_to_category.get(label)
        if category is None:
            continue

        # Gather all relevant marker categories for this label
        categories = [category]
        if label_to_extra and label in label_to_extra:
            categories.extend(label_to_extra[label])

        # Collect all marker genes for this label
        all_markers = []
        for cat in categories:
            all_markers.extend(marker_dict.get(cat, []))

        mask_in = labels == label
        mask_out = ~mask_in
        n_in = mask_in.sum()
        n_out = mask_out.sum()

        for gene in all_markers:
            if gene not in gene_to_idx:
                results.append({
                    "cell_type": label,
                    "marker_category": category,
                    "gene": gene,
                    "found": False,
                    "n_spots": int(n_in),
                })
                continue

            idx = gene_to_idx[gene]
            expr_in = X[mask_in, idx]
            expr_out = X[mask_out, idx]

            # Mean expression
            mean_in = float(np.mean(expr_in))
            mean_out = float(np.mean(expr_out))

            # Log2 fold-change (add pseudocount for stability)
            pseudo = 1e-4
            log2fc = float(np.log2((mean_in + pseudo) / (mean_out + pseudo)))

            # Fraction expressing (> 0)
            frac_in = float(np.mean(expr_in > 0))
            frac_out = float(np.mean(expr_out > 0))

            # Wilcoxon rank-sum test (one-sided: in > out)
            if n_in >= 3 and n_out >= 3:
                try:
                    stat, pval = mannwhitneyu(
                        expr_in, expr_out, alternative="greater"
                    )
                except ValueError:
                    # All values identical
                    stat, pval = 0.0, 1.0
            else:
                stat, pval = 0.0, 1.0

            results.append({
                "cell_type": label,
                "marker_category": category,
                "gene": gene,
                "found": True,
                "n_spots": int(n_in),
                "mean_in": mean_in,
                "mean_out": mean_out,
                "log2fc": log2fc,
                "frac_in": frac_in,
                "frac_out": frac_out,
                "pval": pval,
            })

    # Apply Benjamini-Hochberg FDR correction across all tested markers
    found_results = [r for r in results if r.get("found", False)]
    if found_results:
        pvals = np.array([r["pval"] for r in found_results])
        reject, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
        for r, pval_adj, rej in zip(found_results, pvals_corrected, reject):
            r["pval_adj"] = float(pval_adj)
            r["significant_strict"] = bool(pval_adj < 0.05 and r["log2fc"] > 0.5)
            r["significant_relaxed"] = bool(pval_adj < 0.05 and r["log2fc"] > 0)
            r["significant"] = r["significant_strict"]  # backward compat

    # Set significance fields for not-found markers
    for r in results:
        if not r.get("found", False):
            r.setdefault("significant_strict", False)
            r.setdefault("significant_relaxed", False)
            r.setdefault("significant", False)

    return results


def compute_enrichment_scores(
    results: list[dict],
) -> dict[str, dict]:
    """Compute per-cell-type enrichment scores at both strict and relaxed thresholds.

    Strict: p < 0.05 AND log2FC > 0.5
    Relaxed: p < 0.05 AND log2FC > 0  (any significant upregulation)
    Directional: fraction of markers with log2FC > 0  (regardless of p-value)
    """
    scores = {}
    for r in results:
        ct = r["cell_type"]
        if ct not in scores:
            scores[ct] = {
                "n_markers_total": 0,
                "n_markers_found": 0,
                "n_markers_significant_strict": 0,
                "n_markers_significant_relaxed": 0,
                "n_markers_directional": 0,
                "marker_category": r["marker_category"],
            }
        scores[ct]["n_markers_total"] += 1
        if r["found"]:
            scores[ct]["n_markers_found"] += 1
            if r.get("significant_strict", False):
                scores[ct]["n_markers_significant_strict"] += 1
            if r.get("significant_relaxed", False):
                scores[ct]["n_markers_significant_relaxed"] += 1
            if r.get("log2fc", 0) > 0:
                scores[ct]["n_markers_directional"] += 1

    for ct in scores:
        n_found = scores[ct]["n_markers_found"]
        n_strict = scores[ct]["n_markers_significant_strict"]
        n_relaxed = scores[ct]["n_markers_significant_relaxed"]
        n_dir = scores[ct]["n_markers_directional"]
        scores[ct]["enrichment_strict"] = n_strict / n_found if n_found > 0 else 0.0
        scores[ct]["enrichment_relaxed"] = n_relaxed / n_found if n_found > 0 else 0.0
        scores[ct]["enrichment_directional"] = n_dir / n_found if n_found > 0 else 0.0
        # Keep backward compat
        scores[ct]["enrichment_score"] = scores[ct]["enrichment_strict"]
        scores[ct]["n_markers_significant"] = n_strict

    return scores


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_marker_table(results: list[dict], title: str) -> None:
    """Print a markdown table of per-gene enrichment results."""
    print(f"\n### {title}\n")
    print(
        "| Cell Type | Category | Gene | n_spots | log2FC | "
        "p-value | p-adj (BH) | Frac(in) | Frac(out) | Strict | Relaxed |"
    )
    print(
        "|-----------|----------|------|---------|--------|"
        "---------|------------|----------|-----------|--------|---------|"
    )

    for r in results:
        ct = r["cell_type"]
        cat = r["marker_category"]
        gene = r["gene"]

        if r["found"]:
            n_spots = str(r["n_spots"])
            log2fc = f"{r['log2fc']:+.3f}"
            pval = f"{r['pval']:.2e}" if r["pval"] < 0.01 else f"{r['pval']:.4f}"
            pval_adj = r.get("pval_adj", r["pval"])
            pval_adj_str = f"{pval_adj:.2e}" if pval_adj < 0.01 else f"{pval_adj:.4f}"
            frac_in = f"{r['frac_in']:.3f}"
            frac_out = f"{r['frac_out']:.3f}"
            strict = "YES" if r.get("significant_strict", False) else "no"
            relaxed = "YES" if r.get("significant_relaxed", False) else "no"
        else:
            n_spots = str(r["n_spots"])
            log2fc = pval = pval_adj_str = frac_in = frac_out = "N/A"
            strict = relaxed = "N/A (missing)"

        print(
            f"| {ct} | {cat} | {gene} | {n_spots} | "
            f"{log2fc} | {pval} | {pval_adj_str} | {frac_in} | {frac_out} | {strict} | {relaxed} |"
        )


def print_enrichment_summary(scores: dict[str, dict], title: str) -> None:
    """Print a summary table of enrichment scores per cell type."""
    print(f"\n### {title} -- Enrichment Summary\n")
    print(
        "| Cell Type | Category | Found | "
        "Strict (p<.05, FC>0.5) | Relaxed (p<.05, FC>0) | Directional (FC>0) |"
    )
    print(
        "|-----------|----------|-------|"
        "------------------------|------------------------|---------------------|"
    )

    total_found = 0
    total_strict = 0
    total_relaxed = 0
    total_dir = 0

    for ct in sorted(scores):
        s = scores[ct]
        nf = s["n_markers_found"]
        total_found += nf
        total_strict += s["n_markers_significant_strict"]
        total_relaxed += s["n_markers_significant_relaxed"]
        total_dir += s["n_markers_directional"]
        print(
            f"| {ct} | {s['marker_category']} | "
            f"{nf}/{s['n_markers_total']} | "
            f"{s['n_markers_significant_strict']}/{nf} "
            f"({s['enrichment_strict']:.0%}) | "
            f"{s['n_markers_significant_relaxed']}/{nf} "
            f"({s['enrichment_relaxed']:.0%}) | "
            f"{s['n_markers_directional']}/{nf} "
            f"({s['enrichment_directional']:.0%}) |"
        )

    os = total_strict / total_found if total_found > 0 else 0.0
    ors = total_relaxed / total_found if total_found > 0 else 0.0
    od = total_dir / total_found if total_found > 0 else 0.0
    print(
        f"| **Overall** | -- | {total_found} | "
        f"{total_strict} ({os:.0%}) | "
        f"{total_relaxed} ({ors:.0%}) | "
        f"{total_dir} ({od:.0%}) |"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_validation_pair(
    expression_path: Path,
    labels_path: Path,
    marker_dict: dict[str, list[str]],
    pair_name: str,
    min_genes: int = 200,
) -> tuple[list[dict], dict[str, dict]]:
    """Run full biological validation for one species pair."""
    print(f"\n{'=' * 72}")
    print(f"  BIOLOGICAL VALIDATION: {pair_name}")
    print(f"{'=' * 72}")
    print(f"  Expression: {expression_path}")
    print(f"  Labels: {labels_path}")

    X, labels, gene_names, confidences = load_expression_with_labels(
        expression_path, labels_path, min_genes=min_genes,
    )

    print(f"  Cells: {X.shape[0]}, Genes: {X.shape[1]}")
    print(f"  Unique transferred labels: {np.unique(labels)}")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")

    # Check which markers are present
    available = set(gene_names)
    all_markers = set()
    for genes in marker_dict.values():
        all_markers.update(genes)
    found_markers = all_markers & available
    missing_markers = all_markers - available
    print(f"  Markers in gene panel: {len(found_markers)}/{len(all_markers)}")
    if missing_markers:
        print(f"  Missing markers: {sorted(missing_markers)}")

    results = compute_marker_enrichment(
        X, labels, gene_names, marker_dict,
        LABEL_TO_MARKER_CATEGORY, LABEL_TO_EXTRA_CATEGORIES,
    )
    scores = compute_enrichment_scores(results)

    print_marker_table(results, pair_name)
    print_enrichment_summary(scores, pair_name)

    return results, scores


def main():
    print("# ROSETTA Biological Validation: Marker Gene Enrichment Analysis\n")

    all_scores = {}

    # -----------------------------------------------------------------------
    # 1. Mouse -> Human brain
    # -----------------------------------------------------------------------
    human_expr = PROCESSED_DIR / "visium_human_brain.h5ad"
    human_labels = (
        OUTPUTS_DIR
        / "visium_human_brain_merfish_mouse_brain"
        / "seeds"
        / "seed0"
        / "evaluation"
        / "transferred_labels_class.npz"
    )

    if human_expr.exists() and human_labels.exists():
        results_h, scores_h = run_validation_pair(
            human_expr, human_labels, MARKERS_HUMAN,
            "Mouse -> Human Brain (MERFISH -> Visium)",
            min_genes=200,
        )
        all_scores["human"] = scores_h
    else:
        print("\n[SKIP] Mouse -> Human brain: missing files")
        if not human_expr.exists():
            print(f"  Missing: {human_expr}")
        if not human_labels.exists():
            print(f"  Missing: {human_labels}")

    # -----------------------------------------------------------------------
    # 2. Mouse -> Zebrafish brain
    # -----------------------------------------------------------------------
    zf_expr = PROCESSED_DIR / "stereoseq_zebrafish_brain.h5ad"
    zf_labels = (
        OUTPUTS_DIR
        / "merfish_mouse_brain_stereoseq_zebrafish_brain"
        / "evaluation"
        / "transferred_labels_class.npz"
    )

    if zf_expr.exists() and zf_labels.exists():
        results_z, scores_z = run_validation_pair(
            zf_expr, zf_labels, MARKERS_ZEBRAFISH,
            "Mouse -> Zebrafish Brain (MERFISH -> Stereo-seq)",
            min_genes=0,  # zebrafish data is already preprocessed
        )
        all_scores["zebrafish"] = scores_z
    else:
        print("\n[SKIP] Mouse -> Zebrafish brain: missing files")
        if not zf_expr.exists():
            print(f"  Missing: {zf_expr}")
        if not zf_labels.exists():
            print(f"  Missing: {zf_labels}")

    # -----------------------------------------------------------------------
    # 3. Overall summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("  OVERALL BIOLOGICAL VALIDATION SUMMARY")
    print(f"{'=' * 72}")

    print("\n  NOTE: Visium and Stereo-seq are spatial transcriptomics platforms")
    print("  where each spot/bin captures a MIXTURE of cell types. This attenuates")
    print("  fold-changes vs single-cell data. The 'relaxed' and 'directional'")
    print("  metrics are more appropriate for evaluating spatial label transfer.")

    for species, scores in all_scores.items():
        total_found = sum(s["n_markers_found"] for s in scores.values())
        total_strict = sum(s["n_markers_significant_strict"] for s in scores.values())
        total_relaxed = sum(s["n_markers_significant_relaxed"] for s in scores.values())
        total_dir = sum(s["n_markers_directional"] for s in scores.values())
        n_types = len(scores)

        os = total_strict / total_found if total_found > 0 else 0.0
        ors = total_relaxed / total_found if total_found > 0 else 0.0
        od = total_dir / total_found if total_found > 0 else 0.0

        n_types_strict = sum(
            1 for s in scores.values() if s["enrichment_strict"] > 0.5
        )
        n_types_relaxed = sum(
            1 for s in scores.values() if s["enrichment_relaxed"] > 0.5
        )
        n_types_dir = sum(
            1 for s in scores.values() if s["enrichment_directional"] > 0.5
        )

        print(f"\n  {species.upper()}:")
        print(f"    Strict enrichment  (p<.05, FC>0.5): {os:.3f} "
              f"({total_strict}/{total_found} markers)")
        print(f"    Relaxed enrichment (p<.05, FC>0):   {ors:.3f} "
              f"({total_relaxed}/{total_found} markers)")
        print(f"    Directional        (FC>0):          {od:.3f} "
              f"({total_dir}/{total_found} markers)")
        print(f"    Cell types >50% enriched:  strict={n_types_strict}/{n_types}, "
              f"relaxed={n_types_relaxed}/{n_types}, "
              f"directional={n_types_dir}/{n_types}")

    # Compute combined score
    if all_scores:
        all_found = sum(
            sum(s["n_markers_found"] for s in scores.values())
            for scores in all_scores.values()
        )
        all_strict = sum(
            sum(s["n_markers_significant_strict"] for s in scores.values())
            for scores in all_scores.values()
        )
        all_relaxed = sum(
            sum(s["n_markers_significant_relaxed"] for s in scores.values())
            for scores in all_scores.values()
        )
        all_dir = sum(
            sum(s["n_markers_directional"] for s in scores.values())
            for scores in all_scores.values()
        )
        cs = all_strict / all_found if all_found > 0 else 0.0
        cr = all_relaxed / all_found if all_found > 0 else 0.0
        cd = all_dir / all_found if all_found > 0 else 0.0
        print(f"\n  COMBINED scores:")
        print(f"    Strict:      {cs:.3f} ({all_strict}/{all_found})")
        print(f"    Relaxed:     {cr:.3f} ({all_relaxed}/{all_found})")
        print(f"    Directional: {cd:.3f} ({all_dir}/{all_found})")

    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    main()
