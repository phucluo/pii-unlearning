#!/bin/bash
# =============================================================================
# run_tofu_pipeline.sh — Full pipeline cho TOFU: SFT → Unlearn → Eval
# Usage: bash scripts/run_tofu_pipeline.sh grad_ascent forget10
# =============================================================================
set -e

METHOD=${1:-grad_ascent}    # grad_ascent | grad_diff | npo | dpo | task_vector | aau_pii
SPLIT=${2:-forget10}        # forget01 | forget05 | forget10
MODEL=${3:-llama2-7b-base}

echo "============================================"
echo "TOFU Pipeline: METHOD=$METHOD, SPLIT=$SPLIT, MODEL=$MODEL"
echo "============================================"

# --- Step 1: SFT Exposed trên TOFU (skip nếu đã có) ---
SFT_DIR="outputs/sft_exposed/tofu/${MODEL}"
if [ -d "$SFT_DIR" ] && [ -f "$SFT_DIR/config.json" ]; then
    echo "[Step 1] TOFU SFT model found at $SFT_DIR, skipping..."
else
    echo "[Step 1] Running TOFU SFT Exposed..."
    python train.py --config configs/tofu_sft.yaml \
        --model_family=$MODEL \
        --save_dir=$SFT_DIR
fi

# --- Step 2: Unlearning ---
UNLEARN_DIR="outputs/unlearn/${METHOD}/${SPLIT}/tofu/${MODEL}"
echo "[Step 2] Running TOFU Unlearning: $METHOD on $SPLIT..."
python train.py --config configs/tofu_unlearn.yaml \
    --model_family=$MODEL \
    --model_path=$SFT_DIR \
    --forget_loss=$METHOD \
    --split=$SPLIT \
    --save_dir=$UNLEARN_DIR

# --- Step 3: Evaluation ---
echo "[Step 3] Running TOFU Evaluation..."
python evaluate.py --config configs/tofu_eval.yaml \
    --model_family=$MODEL \
    --model_path=$UNLEARN_DIR \
    --save_dir=$UNLEARN_DIR/eval_results

echo "============================================"
echo "Done! Results at: $UNLEARN_DIR/eval_results/"
echo "============================================"
