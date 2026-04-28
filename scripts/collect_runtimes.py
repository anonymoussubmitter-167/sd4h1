#!/usr/bin/env python3
"""Collect and summarize BRIDGE training runtimes from train.log files.

Parses epoch time from the last epoch progress bar line in each log, then
prints markdown tables for scaling behaviour and seed reproducibility.
"""

import os
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

BRAIN_PAIR = "visium_human_brain_merfish_mouse_brain"
ZEBRAFISH_PAIR = "merfish_mouse_brain_stereoseq_zebrafish_brain"

SEED_DIR = OUTPUTS_DIR / BRAIN_PAIR / "seeds"
SCALING_DIR = OUTPUTS_DIR / BRAIN_PAIR / "scaling"
ABLATION_DIR = OUTPUTS_DIR / BRAIN_PAIR / "ablations"
ZEBRAFISH_LOG = OUTPUTS_DIR / ZEBRAFISH_PAIR / "train.log"

NUM_EPOCHS = 150  # all runs use 150 epochs
REFERENCE_NODES = 500  # baseline for O(n) scaling estimate

# Node counts for scaling runs (the 1000-node point comes from seeds)
SCALING_NODE_COUNTS = [500, 750, 1500, 2000, 3000]
SEED_NODES = 1000
NUM_SEEDS = 5

# Regex for the Rich/Lightning progress-bar epoch line.
# Example:  Epoch 149/149 ━━━━━━━━━━━━ 1/1 0:01:17 • 0:00:00 0.00it/s ...
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+"  # epoch current / max
    r"[━\s]+"                   # progress bar characters
    r"\d+/\d+\s+"               # batch fraction (1/1)
    r"(\d+):(\d+):(\d+)"       # H:MM:SS elapsed time
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_epoch_time(log_path: Path) -> Optional[float]:
    """Return per-epoch wall time in seconds from the *last* epoch line.

    The epoch line is emitted once at the end of training by Lightning's
    progress bar.  The first H:MM:SS group after the batch fraction is the
    cumulative elapsed time for that epoch.
    """
    if not log_path.is_file():
        return None

    last_match = None
    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            m = EPOCH_RE.search(line)
            if m:
                last_match = m

    if last_match is None:
        return None

    hours = int(last_match.group(3))
    minutes = int(last_match.group(4))
    seconds = int(last_match.group(5))
    return hours * 3600 + minutes * 60 + seconds


def fmt_time(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h}:{m:02d}:{s:02d}"


def fmt_total(seconds: float, epochs: int) -> str:
    """Format total training time (epoch_time * epochs) as Hh MMm."""
    total = seconds * epochs
    h = int(total) // 3600
    m = (int(total) % 3600) // 60
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_seed_runtimes() -> list[tuple[int, float]]:
    """Return [(seed_idx, epoch_seconds), ...] for available seed runs."""
    results = []
    for i in range(NUM_SEEDS):
        log = SEED_DIR / f"seed{i}" / "train.log"
        t = parse_epoch_time(log)
        if t is not None:
            results.append((i, t))
    return results


def collect_scaling_runtimes(
    seed_runtimes: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Return [(nodes, epoch_seconds), ...] sorted by node count.

    The 1000-node data point is the mean of seed runs.
    """
    results = []
    for n in SCALING_NODE_COUNTS:
        log = SCALING_DIR / f"nodes_{n}" / "train.log"
        t = parse_epoch_time(log)
        if t is not None:
            results.append((n, t))

    # Add the 1000-node point from seeds (mean)
    if seed_runtimes:
        mean_t = sum(t for _, t in seed_runtimes) / len(seed_runtimes)
        results.append((SEED_NODES, mean_t))

    results.sort(key=lambda x: x[0])
    return results


def collect_ablation_runtimes() -> list[tuple[str, float]]:
    """Return [(ablation_name, epoch_seconds), ...] for ablation runs."""
    results = []
    if not ABLATION_DIR.is_dir():
        return results
    for subdir in sorted(ABLATION_DIR.iterdir()):
        if subdir.is_dir():
            log = subdir / "train.log"
            t = parse_epoch_time(log)
            if t is not None:
                results.append((subdir.name, t))
    return results


def collect_zebrafish_runtime() -> Optional[float]:
    """Return epoch seconds for the zebrafish pair, or None."""
    return parse_epoch_time(ZEBRAFISH_LOG)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_scaling_table(
    scaling: list[tuple[int, float]], epochs: int
) -> None:
    """Print the scaling runtime table in markdown."""
    if not scaling:
        print("No scaling data found.\n")
        return

    # Use smallest node count as baseline for O(n) ratio
    baseline_nodes, baseline_time = scaling[0]

    print("### Scaling Runtime (brain pair, 150 epochs)")
    print()
    print("| Nodes | Epoch Time | Total Time (150 ep) | Scaling vs {0} nodes |".format(
        baseline_nodes
    ))
    print("|------:|-----------:|--------------------:|---------------------:|")

    for nodes, t in scaling:
        ratio = t / baseline_time if baseline_time > 0 else float("nan")
        linear_expected = nodes / baseline_nodes
        scaling_label = f"{ratio:.2f}x (linear: {linear_expected:.1f}x)"
        print(
            f"| {nodes:>5} | {fmt_time(t):>9} | {fmt_total(t, epochs):>19} | {scaling_label:>20} |"
        )
    print()


def print_seed_table(seeds: list[tuple[int, float]], epochs: int) -> None:
    """Print the seed runtime table in markdown."""
    if not seeds:
        print("No seed data found.\n")
        return

    times = [t for _, t in seeds]
    mean_t = sum(times) / len(times)

    print("### Seed Runtimes (brain pair, 1000 nodes, 150 epochs)")
    print()
    print("| Seed | Epoch Time | Total Time (150 ep) |")
    print("|-----:|-----------:|--------------------:|")

    for seed, t in seeds:
        print(
            f"| {seed:>4} | {fmt_time(t):>9} | {fmt_total(t, epochs):>19} |"
        )

    print(
        f"| {'mean':>4} | {fmt_time(mean_t):>9} | {fmt_total(mean_t, epochs):>19} |"
    )
    print()


def print_ablation_table(
    ablations: list[tuple[str, float]], epochs: int
) -> None:
    """Print the ablation runtime table in markdown."""
    if not ablations:
        return

    print("### Ablation Runtimes (brain pair, 150 epochs)")
    print()
    print("| Ablation | Epoch Time | Total Time (150 ep) |")
    print("|----------|-----------:|--------------------:|")

    for name, t in ablations:
        print(
            f"| {name:<8} | {fmt_time(t):>9} | {fmt_total(t, epochs):>19} |"
        )
    print()


def print_zebrafish_runtime(t: Optional[float], epochs: int) -> None:
    """Print the zebrafish pair runtime."""
    if t is None:
        return

    print("### Zebrafish Pair Runtime")
    print()
    print(f"- Epoch time: {fmt_time(t)}")
    print(f"- Total time ({epochs} epochs): {fmt_total(t, epochs)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("# ROSETTA BRIDGE Training Runtimes")
    print()

    seed_runtimes = collect_seed_runtimes()
    scaling_runtimes = collect_scaling_runtimes(seed_runtimes)
    ablation_runtimes = collect_ablation_runtimes()
    zebrafish_time = collect_zebrafish_runtime()

    print_scaling_table(scaling_runtimes, NUM_EPOCHS)
    print_seed_table(seed_runtimes, NUM_EPOCHS)
    print_ablation_table(ablation_runtimes, NUM_EPOCHS)
    print_zebrafish_runtime(zebrafish_time, NUM_EPOCHS)

    # Summary
    if scaling_runtimes:
        smallest_n, smallest_t = scaling_runtimes[0]
        largest_n, largest_t = scaling_runtimes[-1]
        ratio = largest_t / smallest_t if smallest_t > 0 else float("nan")
        linear = largest_n / smallest_n
        print("### Summary")
        print()
        print(
            f"- Scaling from {smallest_n} to {largest_n} nodes: "
            f"{ratio:.2f}x wall time (linear prediction: {linear:.1f}x)"
        )
    if seed_runtimes:
        times = [t for _, t in seed_runtimes]
        mean_t = sum(times) / len(times)
        std_t = (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5
        print(
            f"- Seed variability (n={len(seed_runtimes)}): "
            f"{fmt_time(mean_t)} +/- {std_t:.1f}s per epoch"
        )
    print()


if __name__ == "__main__":
    main()
