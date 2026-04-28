#!/usr/bin/env bash
# Run after all seeds 5-14 are trained:
# 1. Evaluate seeds 5-14
# 2. Run spatial validation
# 3. Run statistical testing + figures
#
# Usage: bash scripts/run_pipeline_completion.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_SCRIPT="$PROJECT_ROOT/scripts/evaluate_bridge.py"
OUTPUT_BASE="$PROJECT_ROOT/outputs/visium_human_brain_merfish_mouse_brain/seeds"
LOG_DIR="$OUTPUT_BASE/logs_reeval"
mkdir -p "$LOG_DIR"

SOURCE="visium_human_brain"
TARGET="merfish_mouse_brain"

echo "=========================================="
echo "Evaluating seeds 5-14"
echo "=========================================="

# Evaluate 5 at a time across GPUs 1-9
eval_batch() {
    local SEEDS=("$@")
    local PIDS=()
    local GPU=1
    for SEED in "${SEEDS[@]}"; do
        SEED_DIR="$OUTPUT_BASE/seed${SEED}"
        LOG="$LOG_DIR/seed${SEED}_final.log"
        CUDA_VISIBLE_DEVICES=$GPU python "$EVAL_SCRIPT" \
            --source "$SOURCE" --target "$TARGET" \
            --output-dir "$SEED_DIR" --device cuda \
            > "$LOG" 2>&1 &
        PIDS+=($!)
        echo "[$(date '+%H:%M')] Eval seed $SEED on GPU $GPU (PID $!)"
        GPU=$((GPU + 1))
    done
    echo "Waiting for batch..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" || echo "WARNING: PID $pid exited non-zero"
    done
    echo "[$(date '+%H:%M')] Batch complete."
}

# Batch 1: seeds 5-13
eval_batch 5 6 7 8 9 10 11 12 13
# Batch 2: seed 14
SEED_DIR="$OUTPUT_BASE/seed14"
LOG="$LOG_DIR/seed14_final.log"
CUDA_VISIBLE_DEVICES=1 python "$EVAL_SCRIPT" \
    --source "$SOURCE" --target "$TARGET" \
    --output-dir "$SEED_DIR" --device cuda \
    > "$LOG" 2>&1
echo "[$(date '+%H:%M')] seed 14 eval complete."

echo ""
echo "=========================================="
echo "Running spatial validation"
echo "=========================================="
python "$PROJECT_ROOT/scripts/spatial_validation.py" 2>&1

echo ""
echo "=========================================="
echo "Running statistical testing"
echo "=========================================="
python "$PROJECT_ROOT/scripts/statistical_testing.py" 2>&1

echo ""
echo "=========================================="
echo "Generating paper figures"
echo "=========================================="
python "$PROJECT_ROOT/scripts/generate_paper_figures.py" 2>&1

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
