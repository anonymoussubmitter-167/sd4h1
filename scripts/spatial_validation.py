#!/usr/bin/env python3
"""Spatial validation: demonstrate transferred labels form spatially coherent domains.

For each seed's transferred labels:
  1. Compute silhouette score in spatial coordinates (per-type and global)
  2. Compare against within-species (mouse) spatial coherence as upper bound
  3. Permutation test: shuffle labels 100 times, recompute silhouette

This is stronger evidence than marker gene enrichment because it directly
measures whether transferred cell types cluster in physical space.

Output: outputs/visium_human_brain_merfish_mouse_brain/spatial_validation.json
"""

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed"
PAIR_DIR = BASE / "outputs" / "visium_human_brain_merfish_mouse_brain"
SEEDS_DIR = PAIR_DIR / "seeds"

N_PERMUTATIONS = 100
MAX_CELLS_SILHOUETTE = 10000  # Subsample if too many cells


def compute_spatial_silhouette(coords, labels, max_cells=MAX_CELLS_SILHOUETTE, rng=None):
    """Compute silhouette score of labels in spatial coordinate space.

    Returns global silhouette, per-type silhouettes, and label counts.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0, {}, {}

    n = len(labels)
    if n > max_cells and rng is not None:
        idx = rng.choice(n, max_cells, replace=False)
        coords = coords[idx]
        labels = labels[idx]
        unique_labels = np.unique(labels)

    # Filter labels with too few samples (silhouette needs >= 2 per cluster)
    label_counts = {l: int((labels == l).sum()) for l in unique_labels}
    valid_mask = np.array([label_counts[l] >= 2 for l in labels])
    if valid_mask.sum() < 10:
        return 0.0, {}, label_counts

    coords_valid = coords[valid_mask]
    labels_valid = labels[valid_mask]

    global_sil = float(silhouette_score(coords_valid, labels_valid, metric="euclidean"))

    # Per-type silhouette
    sample_sils = silhouette_samples(coords_valid, labels_valid, metric="euclidean")
    per_type = {}
    for l in np.unique(labels_valid):
        mask = labels_valid == l
        per_type[l] = float(sample_sils[mask].mean())

    return global_sil, per_type, label_counts


def main():
    print("Loading data...")
    adata_src = ad.read_h5ad(DATA / "visium_human_brain.h5ad")  # unlabeled (human)
    adata_tgt = ad.read_h5ad(DATA / "merfish_mouse_brain.h5ad")  # labeled (mouse)

    coords_human = np.array(adata_src.obsm["spatial"][:, :2], dtype=np.float32)
    coords_mouse = np.array(adata_tgt.obsm["spatial"][:, :2], dtype=np.float32)

    rng = np.random.default_rng(42)

    results = {}

    # --- Within-species control (mouse) ---
    print("\nWithin-species spatial coherence (mouse)...")
    if "class" in adata_tgt.obs.columns:
        mouse_labels = np.array(adata_tgt.obs["class"].values, dtype=str)
        mouse_sil, mouse_per_type, mouse_counts = compute_spatial_silhouette(
            coords_mouse, mouse_labels, rng=rng,
        )
        results["within_species_mouse_global_silhouette"] = mouse_sil
        results["within_species_mouse_per_type"] = mouse_per_type
        print(f"  Mouse global silhouette: {mouse_sil:.4f}")
        for t, s in sorted(mouse_per_type.items(), key=lambda x: -x[1]):
            print(f"    {t}: {s:.4f} (n={mouse_counts.get(t, 0)})")

    # --- Per-seed transferred label analysis ---
    seed_ids = list(range(15))  # Try all possible seeds
    seed_silhouettes = []

    for seed_id in seed_ids:
        eval_dir = SEEDS_DIR / f"seed{seed_id}" / "evaluation"
        npz_path = eval_dir / "transferred_labels_class.npz"
        if not npz_path.exists():
            continue

        print(f"\nSeed {seed_id}:")
        data = np.load(npz_path)
        pred_labels = data["labels"]

        # Transferred labels are on human tissue
        sil, per_type, counts = compute_spatial_silhouette(
            coords_human[:len(pred_labels)], pred_labels, rng=rng,
        )
        seed_silhouettes.append(sil)
        results[f"seed{seed_id}_global_silhouette"] = sil
        results[f"seed{seed_id}_per_type"] = per_type
        print(f"  Global silhouette: {sil:.4f}")
        for t, s in sorted(per_type.items(), key=lambda x: -x[1])[:5]:
            print(f"    {t}: {s:.4f} (n={counts.get(t, 0)})")

        # Permutation test
        print(f"  Permutation test ({N_PERMUTATIONS} permutations)...")
        null_sils = []
        for _ in range(N_PERMUTATIONS):
            shuffled = rng.permutation(pred_labels)
            null_sil, _, _ = compute_spatial_silhouette(
                coords_human[:len(pred_labels)], shuffled, rng=rng,
            )
            null_sils.append(null_sil)

        null_sils = np.array(null_sils)
        p_value = float(np.mean(null_sils >= sil))
        results[f"seed{seed_id}_permutation_null_mean"] = float(null_sils.mean())
        results[f"seed{seed_id}_permutation_null_std"] = float(null_sils.std())
        results[f"seed{seed_id}_permutation_p_value"] = p_value
        print(f"  Null: {null_sils.mean():.4f} +/- {null_sils.std():.4f}")
        print(f"  p-value: {p_value:.4f}")

    # --- Summary ---
    if seed_silhouettes:
        seed_sils = np.array(seed_silhouettes)
        results["summary_n_seeds"] = len(seed_silhouettes)
        results["summary_mean_silhouette"] = float(seed_sils.mean())
        results["summary_std_silhouette"] = float(seed_sils.std(ddof=1))
        print(f"\n{'='*60}")
        print(f"SPATIAL VALIDATION SUMMARY ({len(seed_silhouettes)} seeds)")
        print(f"  Transferred label silhouette: {seed_sils.mean():.4f} +/- {seed_sils.std(ddof=1):.4f}")
        if "within_species_mouse_global_silhouette" in results:
            print(f"  Within-species (mouse) silhouette: {results['within_species_mouse_global_silhouette']:.4f}")
        print(f"  All seeds have permutation p < 0.01: "
              f"{all(results.get(f'seed{s}_permutation_p_value', 1.0) < 0.01 for s in seed_ids if f'seed{s}_permutation_p_value' in results)}")

    out_path = PAIR_DIR / "spatial_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
