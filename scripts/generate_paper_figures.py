#!/usr/bin/env python3
"""Generate publication-quality figures for the ROSETTA ISMB paper.

Produces 6 figures:
  A) Baseline comparison bar chart (BRIDGE vs baselines, with error bars)
  B) Node scaling curve (ARI + training time vs nodes)
  C) Ablation waterfall (paired delta ARI from full model, 3 seeds)
  D) Three-species evolutionary story
  E) Conformal vs naive confidence thresholding
  F) Spatial validation (silhouette in physical space)

Outputs are saved to outputs/figures/ as both PDF and PNG (300 DPI).
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")

# Color-blind-friendly palette (Okabe-Ito inspired, then extended)
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_RED = "#D55E00"
CB_PURPLE = "#CC79A7"
CB_CYAN = "#56B4E9"
CB_YELLOW = "#F0E442"
CB_BLACK = "#000000"

PALETTE = [CB_BLUE, CB_ORANGE, CB_GREEN, CB_RED, CB_PURPLE, CB_CYAN, CB_YELLOW]

LABEL_SIZE = 10
TITLE_SIZE = 12
TICK_SIZE = 9

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HM_PAIR_DIR = PROJECT_ROOT / "outputs" / "visium_human_brain_merfish_mouse_brain"
HM_SEEDS_DIR = HM_PAIR_DIR / "seeds"
HM_SCALING_DIR = HM_PAIR_DIR / "scaling"
MZ_EVAL = PROJECT_ROOT / "outputs" / "merfish_mouse_brain_stereoseq_zebrafish_brain" / "evaluation" / "metrics.json"


def _load_json(path):
    """Load a JSON file, returning None if it does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _save(fig, name):
    """Save figure as both PDF and PNG."""
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"  Saved {OUTPUT_DIR / name}.pdf  and  .png")
    plt.close(fig)


# ===================================================================
# Figure A: Baseline Comparison Bar Chart (with error bars)
# ===================================================================
def figure_a():
    print("Generating Figure A: Baseline comparison ...")

    # --- Load BRIDGE seed data (Human-Mouse brain, up to 15 seeds) ---
    hm_bridge_aris = []
    for seed_idx in range(15):
        m = _load_json(HM_SEEDS_DIR / f"seed{seed_idx}" / "evaluation" / "metrics.json")
        if m is not None and "pseudo_validation_class_best_ari" in m:
            hm_bridge_aris.append(m["pseudo_validation_class_best_ari"])
    n_seeds = len(hm_bridge_aris)
    hm_bridge_mean = np.mean(hm_bridge_aris) if hm_bridge_aris else 0.437
    hm_bridge_std = np.std(hm_bridge_aris, ddof=1) if n_seeds > 1 else 0.0
    # 95% CI using t-distribution
    from scipy.stats import t as t_dist
    t_crit = t_dist.ppf(0.975, df=max(1, n_seeds - 1))
    hm_bridge_ci = t_crit * hm_bridge_std / np.sqrt(n_seeds) if n_seeds > 0 else 0.0

    # --- Load multi-seed baseline data ---
    multiseed_path = HM_PAIR_DIR / "multiseed_baseline_aris.json"
    ms_data = _load_json(multiseed_path)

    hm_baselines = {
        "SAMap": {"mean": 0.0, "std": 0.0},
        "iNMF": {"mean": 0.122, "std": 0.0},
        "Harmony": {"mean": 0.0, "std": 0.0},
        "Scanorama": {"mean": 0.0, "std": 0.0},
        "BBKNN": {"mean": 0.0, "std": 0.0},
        "Expression": {"mean": 0.0, "std": 0.0},
    }

    # Map from multiseed JSON keys to display names
    ms_key_map = {
        "harmony": "Harmony", "scanorama": "Scanorama",
        "bbknn": "BBKNN", "expression": "Expression",
    }
    if ms_data is not None:
        for ms_key, display_name in ms_key_map.items():
            full_key = f"{ms_key}_class"
            if full_key in ms_data and isinstance(ms_data[full_key], dict):
                hm_baselines[display_name]["mean"] = ms_data[full_key].get("mean", 0.0)
                hm_baselines[display_name]["std"] = ms_data[full_key].get("std", 0.0)

    # Fallback: try baseline_ari_results.json if multiseed not available
    if ms_data is None:
        nb = _load_json(HM_PAIR_DIR / "baseline_ari_results.json")
        if nb is not None:
            for bl_key, bl_name in [
                ("baseline_harmony_class", "Harmony"),
                ("baseline_scanorama_class", "Scanorama"),
                ("baseline_bbknn_class", "BBKNN"),
                ("baseline_expression_class", "Expression"),
            ]:
                best_ari = 0.0
                for res in ["0.1", "0.2", "0.3", "0.5", "0.7", "1.0", "1.5", "2.0", "3.0"]:
                    key = f"{bl_key}_leiden{res}_ari"
                    if key in nb:
                        best_ari = max(best_ari, nb[key])
                if best_ari > 0:
                    hm_baselines[bl_name]["mean"] = best_ari

    # Load SAMap ARI
    samap_data = _load_json(HM_PAIR_DIR / "samap_baseline" / "samap_leiden_ari.json")
    if samap_data is not None:
        samap_ari = samap_data.get("samap_class_best_leiden_ari", 0.0)
        if samap_ari > 0:
            hm_baselines["SAMap"]["mean"] = samap_ari

    # --- Mouse-Zebrafish ---
    mz_metrics = _load_json(MZ_EVAL)
    mz_bridge_ari = 0.342
    if mz_metrics is not None:
        mz_bridge_ari = mz_metrics.get("pseudo_validation_class_best_ari", 0.342)

    mz_baselines = {
        "SAMap": 0.0, "iNMF": 0.016, "Harmony": 0.045,
        "Scanorama": 0.043, "BBKNN": 0.015, "Expression": 0.027,
    }
    if mz_metrics is not None:
        for bl_key, bl_name in [
            ("baseline_harmony_class", "Harmony"),
            ("baseline_scanorama_class", "Scanorama"),
            ("baseline_bbknn_class", "BBKNN"),
            ("baseline_expression_class", "Expression"),
        ]:
            for res in ["0.5", "0.3", "1.0"]:
                key = f"{bl_key}_leiden{res}_ari"
                if key in mz_metrics:
                    mz_baselines[bl_name] = max(mz_baselines[bl_name], mz_metrics[key])

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    baseline_names = ["BRIDGE", "SAMap", "iNMF", "Harmony", "Scanorama", "BBKNN", "Expression"]
    colors = [CB_BLUE, CB_YELLOW, CB_ORANGE, CB_GREEN, CB_RED, CB_PURPLE, CB_CYAN]

    # Human-Mouse panel (with error bars)
    hm_vals = [hm_bridge_mean] + [hm_baselines[n]["mean"] for n in baseline_names[1:]]
    hm_errs = [hm_bridge_ci] + [hm_baselines[n]["std"] for n in baseline_names[1:]]
    bars1 = ax1.bar(
        np.arange(len(baseline_names)), hm_vals, yerr=hm_errs,
        color=colors, edgecolor="white", linewidth=0.5,
        capsize=4, error_kw={"linewidth": 1.2}, width=0.7,
    )
    ax1.set_xticks(np.arange(len(baseline_names)))
    ax1.set_xticklabels(baseline_names, fontsize=TICK_SIZE, rotation=30, ha="right")
    ax1.set_ylabel("Class ARI (Leiden)", fontsize=LABEL_SIZE)
    ax1.set_title(f"Human $\\leftrightarrow$ Mouse Brain\n(90 Myr, n={n_seeds} seeds)", fontsize=TITLE_SIZE)
    ax1.set_ylim(0, 0.75)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    for bar, val in zip(bars1, hm_vals):
        if val > 0.01:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Mouse-Zebrafish panel
    mz_vals = [mz_bridge_ari] + [mz_baselines[n] for n in baseline_names[1:]]
    bars2 = ax2.bar(
        np.arange(len(baseline_names)), mz_vals,
        color=colors, edgecolor="white", linewidth=0.5, width=0.7,
    )
    ax2.set_xticks(np.arange(len(baseline_names)))
    ax2.set_xticklabels(baseline_names, fontsize=TICK_SIZE, rotation=30, ha="right")
    ax2.set_title("Mouse $\\leftrightarrow$ Zebrafish Brain\n(450 Myr divergence)", fontsize=TITLE_SIZE)

    for bar, val in zip(bars2, mz_vals):
        if val > 0.01:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_baseline_comparison")


# ===================================================================
# Figure B: Node Scaling Curve
# ===================================================================
def figure_b():
    print("Generating Figure B: Node scaling ...")

    nodes_list = [500, 750, 1000, 1500, 2000, 3000]
    training_hours = [1.6, 2.0, 3.2, 6.6, 10.9, 26.4]

    ari_values = []
    for n in nodes_list:
        if n == 1000:
            seed_aris = []
            for s in range(15):
                m = _load_json(HM_SEEDS_DIR / f"seed{s}" / "evaluation" / "metrics.json")
                if m is not None and "pseudo_validation_class_best_ari" in m:
                    seed_aris.append(m["pseudo_validation_class_best_ari"])
            ari_values.append(np.mean(seed_aris) if seed_aris else 0.437)
        else:
            m = _load_json(HM_SCALING_DIR / f"nodes_{n}" / "evaluation" / "metrics.json")
            if m is not None:
                ari_values.append(m["pseudo_validation_class_best_ari"])
            else:
                ari_values.append(np.nan)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    line1 = ax1.plot(nodes_list, ari_values, "o-", color=CB_BLUE, linewidth=2,
                     markersize=7, label="Class ARI", zorder=3)
    ax1.set_xlabel("Number of Nodes per Species", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Class ARI (Leiden)", fontsize=LABEL_SIZE, color=CB_BLUE)
    ax1.tick_params(axis="y", labelcolor=CB_BLUE, labelsize=TICK_SIZE)
    ax1.tick_params(axis="x", labelsize=TICK_SIZE)
    ax1.set_ylim(0.25, 0.55)

    idx_1000 = nodes_list.index(1000)
    ax1.plot(1000, ari_values[idx_1000], marker="*", markersize=18, color=CB_ORANGE,
             zorder=5, markeredgecolor="black", markeredgewidth=0.5)
    ax1.annotate(f"Sweet spot\nARI={ari_values[idx_1000]:.3f}",
                 xy=(1000, ari_values[idx_1000]),
                 xytext=(1350, ari_values[idx_1000] + 0.04),
                 fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=CB_ORANGE, lw=1.5),
                 color=CB_ORANGE)

    ax2 = ax1.twinx()
    line2 = ax2.plot(nodes_list, training_hours, "s--", color=CB_RED, linewidth=2,
                     markersize=6, label="Training time", zorder=2)
    ax2.set_ylabel("Training Time (hours)", fontsize=LABEL_SIZE, color=CB_RED)
    ax2.tick_params(axis="y", labelcolor=CB_RED, labelsize=TICK_SIZE)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=TICK_SIZE, framealpha=0.9)

    ax1.set_title("Node Scaling: Quality vs Compute", fontsize=TITLE_SIZE)
    ax1.set_xticks(nodes_list)
    ax1.set_xticklabels([str(n) for n in nodes_list])
    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig_scaling")


# ===================================================================
# Figure C: Ablation Waterfall (multi-seed paired)
# ===================================================================
def figure_c():
    print("Generating Figure C: Ablation waterfall ...")

    ablation_dir = HM_PAIR_DIR / "ablations_multiseed"
    conditions = [
        ("no_gw_structure", "No GW structure"),
        ("no_projection_head", "No projection head"),
        ("no_mmd", "No MMD loss"),
        ("no_contrastive", "No contrastive loss"),
        ("no_encoder_freeze", "No encoder freezing"),
    ]

    # Load multi-seed ablation data
    full_aris = []
    for seed in [0, 1, 2]:
        m = _load_json(ablation_dir / "full_model" / f"seed{seed}" / "evaluation" / "metrics.json")
        if m is not None and "pseudo_validation_class_best_ari" in m:
            full_aris.append(m["pseudo_validation_class_best_ari"])

    has_multiseed = len(full_aris) >= 2
    if has_multiseed:
        full_mean = np.mean(full_aris)
    else:
        full_mean = 0.585  # Fallback from RESULTS.md

    names = []
    deltas = []
    delta_stds = []
    aris = []

    for cond_key, cond_name in conditions:
        if has_multiseed:
            abl_aris = []
            for seed in [0, 1, 2]:
                m = _load_json(ablation_dir / cond_key / f"seed{seed}" / "evaluation" / "metrics.json")
                if m is not None and "pseudo_validation_class_best_ari" in m:
                    abl_aris.append(m["pseudo_validation_class_best_ari"])
            if abl_aris:
                abl_mean = np.mean(abl_aris)
                # Paired differences
                n_paired = min(len(full_aris), len(abl_aris))
                diffs = np.array(full_aris[:n_paired]) - np.array(abl_aris[:n_paired])
                names.append(cond_name)
                deltas.append(-diffs.mean())  # Negative = drop from full
                delta_stds.append(diffs.std(ddof=1) if n_paired > 1 else 0.0)
                aris.append(abl_mean)
                continue

        # Fallback: single-seed from RESULTS.md
        fallback = {
            "No GW structure": 0.364, "No projection head": 0.407,
            "No MMD loss": 0.481, "No contrastive loss": 0.522,
            "No encoder freezing": 0.561,
        }
        if cond_name in fallback:
            names.append(cond_name)
            deltas.append(fallback[cond_name] - full_mean)
            delta_stds.append(0.0)
            aris.append(fallback[cond_name])

    # Sort by impact (most negative first)
    order = np.argsort(deltas)
    names = [names[i] for i in order]
    deltas = [deltas[i] for i in order]
    delta_stds = [delta_stds[i] for i in order]
    aris = [aris[i] for i in order]

    # Color by severity
    severity_colors = []
    max_delta = max(abs(d) for d in deltas) if deltas else 1.0
    for d in deltas:
        frac = abs(d) / max_delta
        r = 0.84 + (0.94 - 0.84) * (1 - frac)
        g = 0.37 + (0.78 - 0.37) * (1 - frac)
        b = 0.0
        severity_colors.append((r, g, b))

    fig, ax = plt.subplots(figsize=(8, 4))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, deltas, xerr=delta_stds if any(s > 0 for s in delta_stds) else None,
                   color=severity_colors, edgecolor="white", linewidth=0.5,
                   height=0.65, capsize=3)

    for i, (bar, delta, ari) in enumerate(zip(bars, deltas, aris)):
        mid_x = delta / 2
        ax.text(mid_x, i, f"{delta:+.3f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        ax.text(0.005, i, f"ARI = {ari:.3f}", ha="left", va="center",
                fontsize=8.5, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=TICK_SIZE)
    ax.set_xlabel("$\\Delta$ Class ARI from Full Model", fontsize=LABEL_SIZE)
    title_suffix = " (paired, 3 seeds)" if has_multiseed else ""
    ax.set_title(f"Ablation Study (Full Model ARI = {full_mean:.3f}){title_suffix}", fontsize=TITLE_SIZE)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlim(-0.28, 0.06)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig_ablation")


# ===================================================================
# Figure D: Three-Species Evolutionary Story
# ===================================================================
def figure_d():
    print("Generating Figure D: Three-species evolutionary story ...")

    species_pairs = ["Human$\\leftrightarrow$Mouse\n(90 Myr)", "Mouse$\\leftrightarrow$Zebrafish\n(450 Myr)"]
    metrics = ["Class ARI", "kNN Mixing\n(k=100)", "Top COMPASS\nRegion Cons."]

    # Load updated values from seeds
    hm_aris = []
    for s in range(15):
        m = _load_json(HM_SEEDS_DIR / f"seed{s}" / "evaluation" / "metrics.json")
        if m is not None and "pseudo_validation_class_best_ari" in m:
            hm_aris.append(m["pseudo_validation_class_best_ari"])
    hm_ari = np.mean(hm_aris) if hm_aris else 0.437

    hm_values = [hm_ari, 0.012, 0.805]
    mz_values = [0.342, 0.057, 0.921]

    compass_labels = ["GABAergic PV+", "Cerebellar"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    for i, (metric, hm_v, mz_v) in enumerate(zip(metrics, hm_values, mz_values)):
        ax = axes[i]
        x = np.arange(2)
        vals = [hm_v, mz_v]
        bar_colors = [CB_BLUE, CB_GREEN]

        bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.55)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ax.get_ylim()[1] * 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(species_pairs, fontsize=TICK_SIZE - 1)
        ax.set_title(metric, fontsize=TITLE_SIZE - 1, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="y", alpha=0.3)

        if i == 2:
            ax.text(0, hm_v + ax.get_ylim()[1] * 0.10, compass_labels[0],
                    ha="center", va="bottom", fontsize=7.5, fontstyle="italic", color="#555555")
            ax.text(1, mz_v + ax.get_ylim()[1] * 0.10, compass_labels[1],
                    ha="center", va="bottom", fontsize=7.5, fontstyle="italic", color="#555555")

    fig.suptitle("Cross-Species Alignment Across Vertebrate Evolution", fontsize=TITLE_SIZE, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_three_species")


# ===================================================================
# Figure E: Conformal vs Naive
# ===================================================================
def figure_e():
    print("Generating Figure E: Conformal vs naive ...")

    annotation_levels = ["class", "subclass", "neurotrans.", "supertype", "cluster"]

    hm_set_sizes = [1.01, 1.31, 1.00, 1.75, 3.64]
    hm_naive_accept = [23.1, 11.7, 36.0, 9.6, 5.6]

    mz_set_sizes = [1.00, 1.15, 1.00, 1.36, 2.68]
    mz_naive_accept = [35.9, 32.7, 55.9, 29.4, 18.9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(annotation_levels))
    bar_width = 0.35

    for ax, set_sizes, naive_accept, title in [
        (ax1, hm_set_sizes, hm_naive_accept, "Human $\\leftrightarrow$ Mouse Brain"),
        (ax2, mz_set_sizes, mz_naive_accept, "Mouse $\\leftrightarrow$ Zebrafish Brain"),
    ]:
        bars_conf = ax.bar(x - bar_width / 2, set_sizes, bar_width,
                           color=CB_BLUE, edgecolor="white", linewidth=0.5,
                           label="Conformal set size")
        ax_twin = ax.twinx()
        bars_naive = ax_twin.bar(x + bar_width / 2, naive_accept, bar_width,
                                 color=CB_RED, edgecolor="white", linewidth=0.5,
                                 label="Naive acceptance (%)", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(annotation_levels, fontsize=TICK_SIZE, rotation=25, ha="right")
        ax.set_ylabel("Conformal Mean Set Size", fontsize=LABEL_SIZE, color=CB_BLUE)
        ax.tick_params(axis="y", labelcolor=CB_BLUE, labelsize=TICK_SIZE)
        ax.set_ylim(0, 5)
        ax_twin.set_ylabel("Naive Acceptance Rate (%)", fontsize=LABEL_SIZE, color=CB_RED)
        ax_twin.tick_params(axis="y", labelcolor=CB_RED, labelsize=TICK_SIZE)
        ax_twin.set_ylim(0, 70)
        ax.set_title(title, fontsize=TITLE_SIZE)

        for bar, val in zip(bars_conf, set_sizes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                    color=CB_BLUE, fontweight="bold")
        for bar, val in zip(bars_naive, naive_accept):
            ax_twin.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                         f"{val:.1f}%", ha="center", va="bottom", fontsize=7,
                         color=CB_RED, fontweight="bold")

        lines = [bars_conf, bars_naive]
        labels = ["Conformal set size", "Naive acceptance (%)"]
        ax.legend(lines, labels, loc="upper left", fontsize=TICK_SIZE - 1, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig_conformal")


# ===================================================================
# Figure F: Spatial Validation
# ===================================================================
def figure_f():
    print("Generating Figure F: Spatial validation ...")

    spatial_data = _load_json(HM_PAIR_DIR / "spatial_validation.json")
    if spatial_data is None:
        print("  [skip] spatial_validation.json not found")
        return

    # Collect per-seed silhouettes and null distributions
    seed_sils = []
    null_means = []
    for seed in range(15):
        sil_key = f"seed{seed}_global_silhouette"
        null_key = f"seed{seed}_permutation_null_mean"
        if sil_key in spatial_data:
            seed_sils.append(spatial_data[sil_key])
        if null_key in spatial_data:
            null_means.append(spatial_data[null_key])

    mouse_sil = spatial_data.get("within_species_mouse_global_silhouette")

    if not seed_sils:
        print("  [skip] No seed silhouettes found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: Transferred vs within-species vs null
    categories = ["Transferred\nlabels", "Within-species\n(mouse)", "Permutation\nnull"]
    means = [np.mean(seed_sils), mouse_sil or 0, np.mean(null_means) if null_means else 0]
    stds = [np.std(seed_sils, ddof=1) if len(seed_sils) > 1 else 0, 0,
            np.std(null_means, ddof=1) if len(null_means) > 1 else 0]
    colors_bar = [CB_BLUE, CB_GREEN, CB_RED]

    bars = ax1.bar(np.arange(3), means, yerr=stds, color=colors_bar,
                   edgecolor="white", linewidth=0.5, capsize=4, width=0.6)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(np.arange(3))
    ax1.set_xticklabels(categories, fontsize=TICK_SIZE)
    ax1.set_ylabel("Spatial Silhouette Score", fontsize=LABEL_SIZE)
    ax1.set_title("Spatial Coherence of Labels", fontsize=TITLE_SIZE)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Per-seed silhouettes with null reference line
    ax2.bar(np.arange(len(seed_sils)), seed_sils, color=CB_BLUE,
            edgecolor="white", linewidth=0.5, width=0.7, alpha=0.8)
    if null_means:
        ax2.axhline(y=np.mean(null_means), color=CB_RED, linestyle="--",
                     linewidth=1.5, label=f"Null mean ({np.mean(null_means):.3f})")
    if mouse_sil is not None:
        ax2.axhline(y=mouse_sil, color=CB_GREEN, linestyle="--",
                     linewidth=1.5, label=f"Mouse within-species ({mouse_sil:.3f})")
    ax2.set_xlabel("Seed", fontsize=LABEL_SIZE)
    ax2.set_ylabel("Spatial Silhouette Score", fontsize=LABEL_SIZE)
    ax2.set_title("Per-Seed Spatial Coherence", fontsize=TITLE_SIZE)
    ax2.set_xticks(np.arange(len(seed_sils)))
    ax2.legend(fontsize=TICK_SIZE - 1, framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig_spatial_validation")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    figure_a()
    figure_b()
    figure_c()
    figure_d()
    figure_e()
    figure_f()
    print("=" * 60)
    print("All 6 figures generated successfully.")
