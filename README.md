# ROSETTA: Reconstruction Of Spatial Expression Topologies Through Alignment

Anonymous code submission for review.

---

## Overview

ROSETTA is a framework for cross-species spatial transcriptomics alignment. It frames inter-species integration as a structured optimal transport problem — Fused Gromov-Wasserstein (FGW) transport simultaneously matches gene expression similarity and spatial tissue topology, exploiting the conservation of cell-type spatial organization across species even where individual gene expression diverges.

The framework has four components:

| Module | Role |
|--------|------|
| **CHISEL** | Spatial-expression hybrid GNN encoder with DiffPool hierarchical pooling |
| **BRIDGE** | Cross-species FGW alignment with joint BatchNorm and MMD regularization |
| **COMPASS** | Consensus NMF discovery of conserved gene programs |
| **ATLAS** | Conformal prediction for annotation transfer with coverage guarantees |

---

## Repository Structure

```
rosetta/          # Main package
  chisel/         # CHISEL encoder (GNN, graph construction, multi-scale pooling)
  bridge/         # BRIDGE alignment (FGW transport, shared projection, losses)
  compass/        # COMPASS module discovery (NMF, conservation scoring)
  atlas/          # ATLAS conformal annotation transfer
  data/           # Data loaders, preprocessing, ortholog mapping
  utils/          # Metrics, visualization, config

scripts/          # Training and evaluation scripts
  train_chisel.py           # Pretrain CHISEL encoders
  train_bridge.py           # Train BRIDGE alignment
  evaluate_bridge.py        # Evaluate alignment (ARI, NMI, silhouette)
  compute_baseline_ari.py   # Baseline evaluation (Harmony, Scanorama, BBKNN)
  run_compass.py            # Run COMPASS gene module discovery
  run_conformal_eval.py     # ATLAS conformal evaluation
  statistical_testing.py    # Multi-seed statistical tests
  generate_paper_figures.py # Reproduce all paper figures

configs/          # Dataset-specific configs (Visium, MERFISH, Stereo-seq, ISH)
data/orthologs/   # Human-mouse, human-zebrafish, mouse-zebrafish ortholog maps
tests/            # Unit and integration tests (pytest)
```

---

## Installation

```bash
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.4, CUDA-capable GPU recommended.

Full dependencies are listed in `pyproject.toml`.

---

## Data

Raw datasets are available from the original sources; `scripts/download_data.py` automates retrieval for publicly available datasets. Processed `.h5ad` files follow the AnnData format.

Species pairs evaluated:

| Source | Target | Technology |
|--------|--------|------------|
| Visium human brain | MERFISH mouse brain | Visium / MERFISH |
| Visium human intestine | Visium mouse intestine | Visium / Visium |
| Visium human brain | Stereo-seq zebrafish brain | Visium / Stereo-seq |

---

## Reproducing Results

**Step 1 — Pretrain CHISEL encoders:**
```bash
python scripts/train_chisel.py --dataset visium_human_brain --config configs/visium.yaml
python scripts/train_chisel.py --dataset merfish_mouse_brain --config configs/merfish.yaml
```

**Step 2 — Train BRIDGE alignment (15 seeds for full CI):**
```bash
bash scripts/train_seeds_parallel.sh   # launches seeds 0-14 across GPUs
```

**Step 3 — Evaluate:**
```bash
python scripts/evaluate_bridge.py --source visium_human_brain --target merfish_mouse_brain
python scripts/compute_baseline_ari.py --source visium_human_brain --target merfish_mouse_brain
```

**Step 4 — COMPASS gene modules:**
```bash
python scripts/run_compass.py --source visium_human_brain --target merfish_mouse_brain
```

**Step 5 — Reproduce figures:**
```bash
python scripts/generate_paper_figures.py
```

---

## Testing

```bash
pytest tests/ -v
```

---

## License

MIT License. See `LICENSE`.
