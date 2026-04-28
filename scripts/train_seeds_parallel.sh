#!/usr/bin/env bash
# Train BRIDGE seeds 5-14 in parallel across GPUs 1-9.
# Round 1: seeds 5-13 on GPUs 1-9 (~3h46m)
# Round 2: seed 14 on GPU 1 (~3h46m)
#
# Usage: bash scripts/train_seeds_parallel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$SCRIPT_DIR/train_bridge.py"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_bridge.py"
OUTPUT_BASE="$PROJECT_ROOT/outputs/visium_human_brain_merfish_mouse_brain/seeds"
LOG_DIR="$OUTPUT_BASE/logs"
mkdir -p "$LOG_DIR"

SOURCE="visium_human_brain"
TARGET="merfish_mouse_brain"
MAX_NODES=1000
MAX_EPOCHS=150

train_seed() {
    local SEED=$1
    local GPU=$2
    local SEED_DIR="$OUTPUT_BASE/seed${SEED}"
    local LOG_FILE="$LOG_DIR/seed${SEED}.log"

    echo "[$(date '+%H:%M:%S')] Starting seed $SEED on GPU $GPU ..."
    CUDA_VISIBLE_DEVICES=$GPU python "$TRAIN_SCRIPT" \
        --source "$SOURCE" --target "$TARGET" \
        --max-nodes "$MAX_NODES" --max-epochs "$MAX_EPOCHS" \
        --seed "$SEED" \
        --output-dir "$SEED_DIR" \
        > "$LOG_FILE" 2>&1

    echo "[$(date '+%H:%M:%S')] Training done for seed $SEED. Running evaluation..."
    CUDA_VISIBLE_DEVICES=$GPU python "$EVAL_SCRIPT" \
        --source "$SOURCE" --target "$TARGET" \
        --output-dir "$SEED_DIR" \
        --device cuda \
        >> "$LOG_FILE" 2>&1

    echo "[$(date '+%H:%M:%S')] Seed $SEED complete."
}

echo "=========================================="
echo "BRIDGE Multi-Seed Training (seeds 5-14)"
echo "=========================================="
echo "Output: $OUTPUT_BASE"
echo ""

# Round 1: seeds 5-13 on GPUs 1-9
echo "--- Round 1: seeds 5-13 on GPUs 1-9 ---"
PIDS=()
for i in $(seq 0 8); do
    SEED=$((5 + i))
    GPU=$((1 + i))
    train_seed "$SEED" "$GPU" &
    PIDS+=($!)
done

echo "Waiting for Round 1 (${#PIDS[@]} jobs)..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "WARNING: PID $pid exited with error"
done
echo "[$(date '+%H:%M:%S')] Round 1 complete."

# Round 2: seed 14 on GPU 1
echo ""
echo "--- Round 2: seed 14 on GPU 1 ---"
train_seed 14 1
echo "[$(date '+%H:%M:%S')] Round 2 complete."

echo ""
echo "=========================================="
echo "All seeds 5-14 trained and evaluated."
echo "=========================================="
