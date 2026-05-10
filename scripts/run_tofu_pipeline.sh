#!/bin/bash
# Full pipeline (TOFU track): SFT -> Unlearn -> Eval
# Usage: bash scripts/run_tofu_pipeline.sh grad_ascent forget10 [model_family]
set -e

METHOD=${1:-grad_ascent}
SPLIT=${2:-forget10}
MODEL=${3:-llama2-7b-base}

echo "TOFU pipeline: METHOD=$METHOD, SPLIT=$SPLIT, MODEL=$MODEL"

SFT_DIR="outputs/sft_exposed/tofu/${MODEL}"
if [ -d "$SFT_DIR" ] && [ -f "$SFT_DIR/config.json" ]; then
    echo "TOFU SFT model found at $SFT_DIR, skipping"
else
    echo "Running TOFU SFT"
    python train.py --config configs/tofu_sft.yaml \
        --model_family=$MODEL \
        --save_dir=$SFT_DIR
fi

UNLEARN_DIR="outputs/unlearn/${METHOD}/${SPLIT}/tofu/${MODEL}"
echo "Running TOFU unlearning ($METHOD on $SPLIT)"
python train.py --config configs/tofu_unlearn.yaml \
    --model_family=$MODEL \
    --model_path=$SFT_DIR \
    --forget_loss=$METHOD \
    --split=$SPLIT \
    --save_dir=$UNLEARN_DIR

echo "Running TOFU evaluation"
python evaluate.py --config configs/tofu_eval.yaml \
    --model_family=$MODEL \
    --model_path=$UNLEARN_DIR \
    --save_dir=$UNLEARN_DIR/eval_results

echo "Done. Results at $UNLEARN_DIR/eval_results/"
