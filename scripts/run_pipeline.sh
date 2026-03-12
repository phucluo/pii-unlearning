#!/bin/bash
# =============================================================================
# run_pipeline.sh — Full pipeline: SFT → Unlearn → Eval
# Usage: bash scripts/run_pipeline.sh grad_ascent forget10
# =============================================================================
set -e

METHOD=${1:-grad_ascent}    # grad_ascent | npo | dpo | task_vector
SPLIT=${2:-forget10}        # forget1 | forget5 | forget10
MODEL=${3:-llama2-7b-base}  # model family from model_config.yaml

echo "============================================"
echo "Pipeline: METHOD=$METHOD, SPLIT=$SPLIT, MODEL=$MODEL"
echo "============================================"

# --- Step 1: SFT Exposed (skip if already exists) ---
SFT_DIR="outputs/sft_exposed/${MODEL}"
if [ -d "$SFT_DIR" ] && [ -f "$SFT_DIR/config.json" ]; then
    echo "[Step 1] SFT model found at $SFT_DIR, skipping..."
else
    echo "[Step 1] Running SFT Exposed..."
    python train.py --config configs/sft.yaml \
        --model_family=$MODEL \
        --save_dir=$SFT_DIR
fi

# --- Step 2: Unlearning ---
UNLEARN_DIR="outputs/unlearn/${METHOD}/${SPLIT}/${MODEL}"
echo "[Step 2] Running Unlearning: $METHOD on $SPLIT..."
python train.py --config configs/unlearn.yaml \
    --model_family=$MODEL \
    --model_path=$SFT_DIR \
    --forget_loss=$METHOD \
    --split=$SPLIT \
    --save_dir=$UNLEARN_DIR

# --- Step 3: Evaluation ---
echo "[Step 3] Running Evaluation..."
python evaluate.py --config configs/eval.yaml \
    --model_family=$MODEL \
    --model_path=$UNLEARN_DIR \
    --save_dir=$UNLEARN_DIR/eval_results

echo "============================================"
echo "Done! Results at: $UNLEARN_DIR/eval_results/"
echo "============================================"
