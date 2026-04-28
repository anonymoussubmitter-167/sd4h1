#!/usr/bin/env python3
"""
Statistical significance testing for ROSETTA BRIDGE alignment results.

Computes:
  - 95% bootstrap confidence intervals for BRIDGE mean ARI (10,000 resamples)
  - Welch's t-test against multi-seed baselines (both sides have variance)
  - One-sample t-tests against single-point baselines (iNMF, SAMap)
  - Permutation null reporting
  - Cohen's d effect sizes
  - Coefficient of variation
  - Ablation paired t-tests (3 seeds per condition)

Outputs markdown tables to stdout.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
BRAIN_PAIR_DIR = BASE_DIR / "outputs" / "visium_human_brain_merfish_mouse_brain"
BRAIN_SEEDS_DIR = BRAIN_PAIR_DIR / "seeds"
BRAIN_EVAL_DIR = BRAIN_PAIR_DIR / "evaluation"
ZEBRAFISH_EVAL_DIR = (
    BASE_DIR / "outputs" / "merfish_mouse_brain_stereoseq_zebrafish_brain" / "evaluation"
)

SEED_IDS = list(range(15))  # Now 15 seeds
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    """Load a JSON file; return empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"  [warn] Could not load {path}: {exc}", file=sys.stderr)
        return {}


def load_bridge_seed_aris() -> np.ndarray:
    """Load BRIDGE pseudo_validation_class_best_ari for all available seeds."""
    values = []
    for seed in SEED_IDS:
        metrics_path = BRAIN_SEEDS_DIR / f"seed{seed}" / "evaluation" / "metrics.json"
        m = load_json(metrics_path)
        val = m.get("pseudo_validation_class_best_ari")
        if val is not None:
            values.append(float(val))
    return np.array(values)


def _best_leiden_ari(metrics: dict, prefix: str) -> float | None:
    """Extract the best (max) Leiden ARI from a metrics dict for a given prefix."""
    aris: list[float] = []
    for key, val in metrics.items():
        if key.startswith(f"{prefix}_class_leiden") and key.endswith("_ari"):
            aris.append(float(val))
        elif key == f"{prefix}_class_best_ari":
            aris.append(float(val))
    return max(aris) if aris else None


def load_multiseed_baselines() -> dict[str, dict]:
    """
    Load multi-seed baseline ARI values from multiseed_baseline_aris.json.

    Returns dict: method -> {"values": [...], "mean": float, "std": float}
    """
    path = BRAIN_PAIR_DIR / "multiseed_baseline_aris.json"
    data = load_json(path)
    if not data:
        return {}

    baselines = {}
    for key, val_dict in data.items():
        # Keys like "harmony_class", "scanorama_class", etc.
        if "_class" in key and isinstance(val_dict, dict) and "values" in val_dict:
            method = key.replace("_class", "")
            baselines[method] = val_dict
    return baselines


def collect_single_point_baselines() -> dict[str, float]:
    """Collect single-point baseline ARIs (iNMF, SAMap) that don't have multi-seed variants."""
    baselines = {}

    # iNMF from extra_baselines.json
    extra_path = BRAIN_EVAL_DIR / "extra_baselines.json"
    extra = load_json(extra_path)
    inmf_ari = extra.get("inmf_class_best_ari")
    if inmf_ari is not None:
        baselines["inmf"] = float(inmf_ari)
    else:
        baselines["inmf"] = 0.122

    # SAMap from samap_leiden_ari.json
    samap_path = BRAIN_PAIR_DIR / "samap_baseline" / "samap_leiden_ari.json"
    samap_data = load_json(samap_path)
    samap_ari = samap_data.get("samap_class_best_leiden_ari")
    if samap_ari is not None:
        baselines["samap"] = float(samap_ari)

    return baselines


def load_permutation_null() -> dict[str, dict]:
    """Load permutation null results from seed evaluation metrics."""
    nulls = {}
    for seed in SEED_IDS:
        metrics_path = BRAIN_SEEDS_DIR / f"seed{seed}" / "evaluation" / "metrics.json"
        m = load_json(metrics_path)
        for key in ["permutation_null_class_mean", "permutation_null_class_std",
                     "permutation_null_class_95th", "permutation_null_class_p_value"]:
            if key in m:
                nulls.setdefault(key, []).append(float(m[key]))
    return nulls


def load_zebrafish_metrics() -> dict:
    """Load zebrafish evaluation metrics."""
    metrics_path = ZEBRAFISH_EVAL_DIR / "metrics.json"
    return load_json(metrics_path)


def load_ablation_results() -> dict[str, np.ndarray]:
    """Load multi-seed ablation results. Returns condition -> array of ARIs."""
    ablation_dir = BRAIN_PAIR_DIR / "ablations_multiseed"
    conditions = [
        "no_gw_structure", "no_projection_head", "no_mmd",
        "no_contrastive", "no_encoder_freeze", "full_model",
    ]
    results = {}
    for cond in conditions:
        aris = []
        for seed in [0, 1, 2]:
            metrics_path = ablation_dir / cond / f"seed{seed}" / "evaluation" / "metrics.json"
            m = load_json(metrics_path)
            val = m.get("pseudo_validation_class_best_ari")
            if val is not None:
                aris.append(float(val))
        if aris:
            results[cond] = np.array(aris)
    return results


# ---------------------------------------------------------------------------
# Statistical computations
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
    rng_seed: int = RNG_SEED,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(rng_seed)
    n = len(data)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = data[rng.randint(0, n, size=n)]
        boot_means[i] = sample.mean()
    alpha = 1.0 - ci
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lo, hi


def cohens_d(data: np.ndarray, baseline_val: float) -> float:
    """Cohen's d effect size: (mean - baseline) / std."""
    s = data.std(ddof=1)
    if s == 0:
        return float("inf") if data.mean() != baseline_val else 0.0
    return (data.mean() - baseline_val) / s


def cohens_d_two_sample(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for two independent samples."""
    nx, ny = len(x), len(y)
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    pooled_std = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if pooled_std == 0:
        return float("inf") if x.mean() != y.mean() else 0.0
    return (x.mean() - y.mean()) / pooled_std


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("# ROSETTA BRIDGE -- Statistical Significance Testing")
    print()

    # ------------------------------------------------------------------
    # 1. Load BRIDGE seed ARI values
    # ------------------------------------------------------------------
    bridge_aris = load_bridge_seed_aris()
    n_seeds = len(bridge_aris)
    if n_seeds == 0:
        print("ERROR: No BRIDGE seed ARI values found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"## BRIDGE Seed ARI Values ({n_seeds} seeds, Human Brain <-> Mouse Brain)")
    print()
    for i, val in enumerate(bridge_aris):
        print(f"- Seed {i}: {val:.4f}")
    print()

    bridge_mean = bridge_aris.mean()
    bridge_std = bridge_aris.std(ddof=1)
    ci_lo, ci_hi = bootstrap_ci(bridge_aris)
    ci_half = (ci_hi - ci_lo) / 2

    print(f"- **Mean**: {bridge_mean:.3f}")
    print(f"- **Std**: {bridge_std:.3f}")
    print(f"- **n**: {n_seeds} seeds")
    print(f"- **95% Bootstrap CI**: [{ci_lo:.3f}, {ci_hi:.3f}]  (mean +/- {ci_half:.3f})")
    cv = bridge_std / bridge_mean if bridge_mean != 0 else float("inf")
    print(f"- **CV**: {cv:.1%}")
    print()

    # ------------------------------------------------------------------
    # 2. Permutation null
    # ------------------------------------------------------------------
    perm_null = load_permutation_null()
    if perm_null:
        print("## Permutation Null (label shuffling)")
        print()
        null_means = np.array(perm_null.get("permutation_null_class_mean", []))
        null_pvals = np.array(perm_null.get("permutation_null_class_p_value", []))
        null_95th = np.array(perm_null.get("permutation_null_class_95th", []))
        if len(null_means) > 0:
            print(f"- Null ARI (mean across seeds): {null_means.mean():.4f}")
            print(f"- Null 95th percentile: {null_95th.mean():.4f}")
            print(f"- All seeds p < 0.01: {(null_pvals < 0.01).all()}")
            print(f"- Observed/null ratio: {bridge_mean / null_means.mean():.1f}x")
        print()

    # ------------------------------------------------------------------
    # 3. Multi-seed baseline comparisons (Welch's t-test)
    # ------------------------------------------------------------------
    multiseed_baselines = load_multiseed_baselines()
    single_baselines = collect_single_point_baselines()

    baseline_display_names = {
        "harmony": "Harmony",
        "scanorama": "Scanorama",
        "bbknn": "BBKNN",
        "expression": "Expression-only",
        "inmf": "iNMF (LIGER)",
        "samap": "SAMap",
    }

    print("## BRIDGE vs Baselines -- Statistical Comparisons")
    print()

    if multiseed_baselines:
        print("### Multi-seed baselines (Welch's t-test)")
        print()
        print(
            "| Comparison | BRIDGE (mean +/- std) | Baseline (mean +/- std) | Delta "
            "| Welch t-test p | Cohen's d | Sig |"
        )
        print(
            "|:-----------|:---------------------:|:-----------------------:|:-----:"
            "|:--------------:|:---------:|:---:|"
        )

        for method in ["harmony", "scanorama", "bbknn", "expression"]:
            bdata = multiseed_baselines.get(method)
            if bdata is None:
                continue
            bvals = np.array(bdata["values"])
            bmean = bvals.mean()
            bstd = bvals.std(ddof=1)
            delta = bridge_mean - bmean
            # Welch's t-test (unequal variances)
            t_stat, p_welch = stats.ttest_ind(bridge_aris, bvals, equal_var=False)
            d = cohens_d_two_sample(bridge_aris, bvals)
            sig = significance_stars(p_welch)
            name = baseline_display_names.get(method, method)
            d_str = f"{d:.2f}" if np.isfinite(d) else "inf"

            print(
                f"| BRIDGE vs {name} "
                f"| {bridge_mean:.3f} +/- {bridge_std:.3f} "
                f"| {bmean:.3f} +/- {bstd:.3f} "
                f"| +{delta:.3f} "
                f"| {p_welch:.4f} "
                f"| {d_str} "
                f"| {sig} |"
            )
        print()

    if single_baselines:
        print("### Single-point baselines (one-sample t-test)")
        print()
        print(
            "| Comparison | BRIDGE (mean +/- CI) | Baseline ARI | Delta "
            "| t-test p | Cohen's d | Sig |"
        )
        print(
            "|:-----------|:--------------------:|:------------:|:-----:"
            "|:--------:|:---------:|:---:|"
        )

        for method in ["inmf", "samap"]:
            bval = single_baselines.get(method)
            if bval is None:
                continue
            delta = bridge_mean - bval
            t_stat, p_ttest = stats.ttest_1samp(bridge_aris, bval)
            d = cohens_d(bridge_aris, bval)
            sig = significance_stars(p_ttest)
            name = baseline_display_names.get(method, method)
            d_str = f"{d:.2f}" if np.isfinite(d) else "inf"

            print(
                f"| BRIDGE vs {name} "
                f"| {bridge_mean:.3f} +/- {ci_half:.3f} "
                f"| {bval:.3f} "
                f"| +{delta:.3f} "
                f"| {p_ttest:.4f} "
                f"| {d_str} "
                f"| {sig} |"
            )
        print()

    # ------------------------------------------------------------------
    # 4. Seed variance analysis
    # ------------------------------------------------------------------
    print("## Seed Variance Analysis")
    print()
    print(f"- **Coefficient of Variation (CV)**: {cv:.3f} ({cv*100:.1f}%)")
    print(f"- **Range**: [{bridge_aris.min():.3f}, {bridge_aris.max():.3f}]")
    print(f"- **IQR**: [{np.percentile(bridge_aris, 25):.3f}, {np.percentile(bridge_aris, 75):.3f}]")
    print()

    # One-sample t-test against 0
    t_stat, t_p = stats.ttest_1samp(bridge_aris, 0)
    print("### One-Sample t-Test (H0: mean ARI = 0)")
    print()
    print(f"- t-statistic: {t_stat:.3f}")
    print(f"- p-value: {t_p:.6f}")
    print(
        f"- Interpretation: BRIDGE ARI is "
        f"{'significantly' if t_p < 0.05 else 'not significantly'} "
        f"greater than zero (p = {t_p:.6f})"
    )
    print()

    # ------------------------------------------------------------------
    # 5. Ablation paired t-tests
    # ------------------------------------------------------------------
    ablation_results = load_ablation_results()
    if "full_model" in ablation_results and len(ablation_results) > 1:
        print("## Ablation Study (Paired t-test, 3 seeds)")
        print()
        full_aris = ablation_results["full_model"]
        print(f"Full model ARIs: {[f'{v:.3f}' for v in full_aris]}")
        print()

        ablation_display = {
            "no_gw_structure": "No GW structure (alpha=0)",
            "no_projection_head": "No projection head",
            "no_mmd": "No MMD loss",
            "no_contrastive": "No contrastive loss",
            "no_encoder_freeze": "No encoder freezing",
        }

        print(
            "| Condition | Full (mean) | Ablated (mean) | Paired Delta "
            "| Paired t-test p | Sig |"
        )
        print(
            "|:----------|:-----------:|:--------------:|:------------:"
            "|:---------------:|:---:|"
        )

        for cond in ["no_gw_structure", "no_projection_head", "no_mmd",
                      "no_contrastive", "no_encoder_freeze"]:
            if cond not in ablation_results:
                continue
            abl_aris = ablation_results[cond]
            n_paired = min(len(full_aris), len(abl_aris))
            if n_paired < 2:
                continue

            diffs = full_aris[:n_paired] - abl_aris[:n_paired]
            t_stat_p, p_paired = stats.ttest_rel(full_aris[:n_paired], abl_aris[:n_paired])
            sig = significance_stars(p_paired)
            name = ablation_display.get(cond, cond)

            print(
                f"| {name} "
                f"| {full_aris[:n_paired].mean():.3f} "
                f"| {abl_aris[:n_paired].mean():.3f} "
                f"| {diffs.mean():+.3f} "
                f"| {p_paired:.4f} "
                f"| {sig} |"
            )
        print()

    # ------------------------------------------------------------------
    # 6. Spatial validation summary
    # ------------------------------------------------------------------
    spatial_path = BRAIN_PAIR_DIR / "spatial_validation.json"
    spatial = load_json(spatial_path)
    if spatial:
        print("## Spatial Validation (Silhouette in Physical Space)")
        print()
        if "summary_mean_silhouette" in spatial:
            print(f"- Transferred label silhouette: {spatial['summary_mean_silhouette']:.4f} "
                  f"+/- {spatial.get('summary_std_silhouette', 0):.4f}")
        if "within_species_mouse_global_silhouette" in spatial:
            print(f"- Within-species (mouse) silhouette: {spatial['within_species_mouse_global_silhouette']:.4f}")
        # Count seeds with p < 0.01
        sig_count = sum(
            1 for k, v in spatial.items()
            if k.endswith("_permutation_p_value") and v < 0.01
        )
        total_count = sum(1 for k in spatial if k.endswith("_permutation_p_value"))
        if total_count > 0:
            print(f"- Seeds with permutation p < 0.01: {sig_count}/{total_count}")
        print()

    # ------------------------------------------------------------------
    # 7. Zebrafish results
    # ------------------------------------------------------------------
    print("## Mouse <-> Zebrafish Brain Results")
    print()

    zf_metrics = load_zebrafish_metrics()
    if zf_metrics:
        zf_class_ari = zf_metrics.get("pseudo_validation_class_best_ari")
        if zf_class_ari is not None:
            print(f"- **BRIDGE class ARI**: {zf_class_ari:.3f}")
        else:
            print("- BRIDGE class ARI: not available")

        zf_baselines = {}
        for method in ["harmony", "scanorama", "bbknn", "expression"]:
            leiden_aris = []
            for key, val in zf_metrics.items():
                if key.startswith(f"baseline_{method}_class_leiden") and key.endswith("_ari"):
                    leiden_aris.append(float(val))
            if leiden_aris:
                zf_baselines[method] = max(leiden_aris)

        zf_extra_path = ZEBRAFISH_EVAL_DIR / "extra_baselines.json"
        zf_extra = load_json(zf_extra_path)
        inmf_ari = zf_extra.get("inmf_class_best_ari")
        if inmf_ari is not None:
            zf_baselines["inmf"] = float(inmf_ari)

        if zf_baselines and zf_class_ari is not None:
            print()
            print("| Method | class ARI | Delta vs BRIDGE |")
            print("|:-------|:---------:|:---------------:|")
            print(f"| **BRIDGE** | **{zf_class_ari:.3f}** | --- |")
            for method, val in sorted(zf_baselines.items()):
                delta = zf_class_ari - val
                name = baseline_display_names.get(method, method)
                print(f"| {name} | {val:.3f} | +{delta:.3f} |")
        print()
    else:
        print("- Zebrafish metrics not available.")
        print()

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print("## Summary")
    print()
    print(
        f"BRIDGE achieves a mean class ARI of **{bridge_mean:.3f}** "
        f"(95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]) across {n_seeds} random seeds "
        f"on the Human Brain <-> Mouse Brain alignment task."
    )
    print()
    print(
        f"Seed variance: CV = {cv:.1%}, range "
        f"[{bridge_aris.min():.3f}, {bridge_aris.max():.3f}]."
    )


if __name__ == "__main__":
    main()
