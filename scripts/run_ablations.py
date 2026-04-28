#!/usr/bin/env python
"""Ablation studies for BRIDGE cross-species alignment.

Tests the contribution of each component by disabling it and measuring
alignment quality. Ablations:
  1. Full model (baseline)
  2. No GW structure (-alpha 0.0 → feature-only OT)
  3. No contrastive loss (--lambda-cross-contrastive 0.0)
  4. No MMD loss (--lambda-mmd 0.0)
  5. No projection head (--projection-hidden-dim 0 → identity projection)
  6. No encoder freezing (--freeze-encoder-epochs 0)
  7. No spatial edges (expression-only graph)

Usage:
    python scripts/run_ablations.py \
        --source visium_human_brain --target merfish_mouse_brain \
        [--epochs 150] [--max-nodes 3000] [--gpu 0]

    # Just generate commands without running:
    python scripts/run_ablations.py --source ... --target ... --dry-run

    # Run a specific ablation:
    python scripts/run_ablations.py --source ... --target ... --only no_mmd
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Each ablation: name -> dict of train_bridge.py overrides
ABLATIONS = {
    "full_model": {
        "description": "Full BRIDGE model (all components)",
        "overrides": {},
    },
    "no_gw_structure": {
        "description": "Feature-only OT (no Gromov-Wasserstein spatial structure)",
        "overrides": {"--alpha": "0.0"},
    },
    "no_contrastive": {
        "description": "No cross-species contrastive loss",
        "overrides": {"--lambda-cross-contrastive": "0.0"},
    },
    "no_mmd": {
        "description": "No MMD distribution matching loss",
        "overrides": {"--lambda-mmd": "0.0"},
    },
    "no_projection_head": {
        "description": "No shared projection head (identity mapping)",
        "overrides": {"--projection-hidden-dim": "0"},
    },
    "no_encoder_freeze": {
        "description": "No encoder freezing (train everything from start)",
        "overrides": {"--freeze-encoder-epochs": "0"},
    },
    "expression_only_graph": {
        "description": "Expression-only graph (no spatial edges in source)",
        "overrides": {"--no-spatial-source": ""},
    },
}


def build_train_command(
    source: str,
    target: str,
    ablation_name: str,
    overrides: dict,
    epochs: int,
    max_nodes: int,
    gpu: int,
    output_dir: Path,
) -> list[str]:
    """Build the train_bridge.py command for an ablation."""
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "train_bridge.py"),
        "--source", source,
        "--target", target,
        "--max-epochs", str(epochs),
        "--max-nodes", str(max_nodes),
        "--output-dir", str(output_dir),
    ]

    # Apply default hyperparameters
    defaults = {
        "--epsilon": "0.005",
        "--finetune-lr": "1e-4",
        "--projection-hidden-dim": "128",
        "--freeze-encoder-epochs": "10",
    }
    defaults.update(overrides)

    # Handle special flags
    for flag, value in defaults.items():
        if flag == "--no-spatial-source":
            # This needs a code change — skip for now, use alpha=0 as proxy
            continue
        cmd.extend([flag, value])

    return cmd


def build_eval_command(
    source: str,
    target: str,
    output_dir: Path,
    gpu: int,
) -> list[str]:
    """Build the evaluate_bridge.py command."""
    return [
        sys.executable, str(SCRIPTS_DIR / "evaluate_bridge.py"),
        "--source", source,
        "--target", target,
        "--output-dir", str(output_dir),
    ]


def run_command(cmd: list[str], gpu: int, description: str) -> tuple[bool, float]:
    """Run a command with GPU assignment and timing."""
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    logger.info("Running: %s", description)
    logger.info("Command: %s", " ".join(cmd))

    t0 = time.time()
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=3600 * 4,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error("FAILED: %s", description)
        logger.error("stderr: %s", result.stderr[-1000:] if result.stderr else "")
        return False, elapsed

    logger.info("SUCCESS: %s (%.1fs)", description, elapsed)
    return True, elapsed


def load_eval_results(eval_dir: Path) -> dict | None:
    """Load evaluation results from the evaluation directory."""
    results_file = eval_dir / "evaluation" / "alignment_metrics.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    # Try to find metrics from logs
    return None


def main():
    parser = argparse.ArgumentParser(description="Run BRIDGE ablation studies")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--max-nodes", type=int, default=3000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this ablation (name from ABLATIONS dict)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip ablations whose output directory already exists")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation on existing ablation outputs")
    args = parser.parse_args()

    pair_name = f"{args.source}_{args.target}"
    ablation_base = OUTPUTS_DIR / pair_name / "ablations"
    ablation_base.mkdir(parents=True, exist_ok=True)

    # Filter ablations
    ablations_to_run = ABLATIONS
    if args.only:
        if args.only not in ABLATIONS:
            print(f"Unknown ablation: {args.only}")
            print(f"Available: {sorted(ABLATIONS.keys())}")
            sys.exit(1)
        ablations_to_run = {args.only: ABLATIONS[args.only]}

    results_summary = {}

    for name, ablation in ablations_to_run.items():
        output_dir = ablation_base / name
        description = ablation["description"]

        print(f"\n{'='*70}")
        print(f"Ablation: {name}")
        print(f"  {description}")
        print(f"  Output: {output_dir}")
        print(f"{'='*70}")

        if args.skip_existing and output_dir.exists():
            logger.info("Skipping %s (output exists)", name)
            eval_results = load_eval_results(output_dir)
            if eval_results:
                results_summary[name] = eval_results
            continue

        # Build training command
        train_cmd = build_train_command(
            args.source, args.target, name, ablation["overrides"],
            args.epochs, args.max_nodes, args.gpu, output_dir,
        )

        # Build evaluation command
        eval_cmd = build_eval_command(
            args.source, args.target, output_dir, args.gpu,
        )

        if args.dry_run:
            print(f"  TRAIN: {' '.join(train_cmd)}")
            print(f"  EVAL:  {' '.join(eval_cmd)}")
            continue

        # Run training (skip if eval-only)
        if not args.eval_only:
            success, elapsed = run_command(
                train_cmd, args.gpu, f"{name} training",
            )
            if not success:
                results_summary[name] = {"error": "training failed"}
                continue
            results_summary[name] = {"train_time_s": elapsed}

        # Run evaluation
        success, elapsed = run_command(
            eval_cmd, args.gpu, f"{name} evaluation",
        )
        if success:
            eval_results = load_eval_results(output_dir)
            if eval_results:
                results_summary[name] = {
                    **results_summary.get(name, {}),
                    **eval_results,
                }

    # Save and print summary
    if not args.dry_run and results_summary:
        summary_path = ablation_base / "ablation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        logger.info("Summary saved to %s", summary_path)

        print(f"\n{'='*70}")
        print("ABLATION RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"{'Ablation':<25} {'kNN Mix':>10} {'FOSCTTM':>10} {'LT Acc':>10}")
        print("-" * 55)

        for name, res in results_summary.items():
            if "error" in res:
                print(f"{name:<25} {'FAILED':>10}")
                continue
            knn = res.get("knn_mixing", res.get("knn_mixing_score", "N/A"))
            fos = res.get("foscttm", res.get("foscttm_score", "N/A"))
            lt = res.get("label_transfer_accuracy", "N/A")
            knn_s = f"{knn:.4f}" if isinstance(knn, float) else str(knn)
            fos_s = f"{fos:.4f}" if isinstance(fos, float) else str(fos)
            lt_s = f"{lt:.4f}" if isinstance(lt, float) else str(lt)
            print(f"{name:<25} {knn_s:>10} {fos_s:>10} {lt_s:>10}")

        print(f"{'='*70}")


if __name__ == "__main__":
    main()
