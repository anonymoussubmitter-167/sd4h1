#!/usr/bin/env bash
# Train ablation conditions with 3 seeds each for paired statistical testing.
#
# 6 conditions x 3 seeds = 18 jobs total.
# Round 1: 9 jobs on GPUs 1-9 (~3h46m)
# Round 2: 9 jobs on GPUs 1-9 (~3h46m)
#
# Seeds 0-2 of full_model already exist as seed0/seed1/seed2 in the seeds dir.
#
# Usage: bash scripts/train_ablations_multiseed.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$SCRIPT_DIR/train_bridge.py"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_bridge.py"
OUTPUT_BASE="$PROJECT_ROOT/outputs/visium_human_brain_merfish_mouse_brain/ablations_multiseed"
LOG_DIR="$OUTPUT_BASE/logs"
mkdir -p "$LOG_DIR"

SOURCE="visium_human_brain"
TARGET="merfish_mouse_brain"
MAX_NODES=1000
MAX_EPOCHS=150

train_ablation() {
    local CONDITION=$1
    local SEED=$2
    local GPU=$3
    local EXTRA_ARGS=$4
    local OUT_DIR="$OUTPUT_BASE/${CONDITION}/seed${SEED}"
    local LOG_FILE="$LOG_DIR/${CONDITION}_seed${SEED}.log"

    echo "[$(date '+%H:%M:%S')] Starting $CONDITION seed=$SEED on GPU $GPU ..."
    CUDA_VISIBLE_DEVICES=$GPU python "$TRAIN_SCRIPT" \
        --source "$SOURCE" --target "$TARGET" \
        --max-nodes "$MAX_NODES" --max-epochs "$MAX_EPOCHS" \
        --seed "$SEED" \
        --output-dir "$OUT_DIR" \
        $EXTRA_ARGS \
        > "$LOG_FILE" 2>&1

    echo "[$(date '+%H:%M:%S')] Training done: $CONDITION seed=$SEED. Evaluating..."
    CUDA_VISIBLE_DEVICES=$GPU python "$EVAL_SCRIPT" \
        --source "$SOURCE" --target "$TARGET" \
        --output-dir "$OUT_DIR" \
        --device cuda \
        >> "$LOG_FILE" 2>&1

    echo "[$(date '+%H:%M:%S')] Done: $CONDITION seed=$SEED"
}

echo "=========================================="
echo "Multi-Seed Ablation Study"
echo "6 conditions x 3 seeds = 18 jobs"
echo "=========================================="
echo ""

# Define ablation conditions and their extra args
# full_model uses default args (seeds 0-2 already trained; retrain for consistency)
declare -A ABLATION_ARGS
ABLATION_ARGS[no_gw_structure]="--alpha 0"
ABLATION_ARGS[no_projection_head]="--projection-hidden-dim 0"
ABLATION_ARGS[no_mmd]="--lambda-mmd 0"
ABLATION_ARGS[no_contrastive]="--lambda-cross-contrastive 0"
ABLATION_ARGS[no_encoder_freeze]="--freeze-encoder-epochs 0"
ABLATION_ARGS[full_model]=""

# Build job list: condition, seed, extra_args
JOBS=()
for CONDITION in no_gw_structure no_projection_head no_mmd no_contrastive no_encoder_freeze full_model; do
    for SEED in 0 1 2; do
        JOBS+=("${CONDITION}|${SEED}|${ABLATION_ARGS[$CONDITION]}")
    done
done

# Dispatch in rounds of 9 (GPUs 1-9)
ROUND=0
for ((i=0; i < ${#JOBS[@]}; i+=9)); do
    ROUND=$((ROUND + 1))
    echo "--- Round $ROUND ---"
    PIDS=()
    for ((j=i; j < i+9 && j < ${#JOBS[@]}; j++)); do
        IFS='|' read -r COND SEED EXTRA <<< "${JOBS[$j]}"
        GPU=$(( (j - i) + 1 ))
        train_ablation "$COND" "$SEED" "$GPU" "$EXTRA" &
        PIDS+=($!)
    done

    echo "Waiting for Round $ROUND (${#PIDS[@]} jobs)..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" || echo "WARNING: PID $pid exited with error"
    done
    echo "[$(date '+%H:%M:%S')] Round $ROUND complete."
    echo ""
done

echo "=========================================="
echo "All ablation jobs complete."
echo "=========================================="
