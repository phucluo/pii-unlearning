#!/bin/bash
# =============================================================================
# setup_data.sh — Clone UnlearnPII repo and copy data files
# Usage: bash scripts/setup_data.sh
# =============================================================================
set -e

echo "Cloning UnlearnPII repo..."
if [ ! -d "Toward-Practical-PII-Unlearning" ]; then
    git clone https://github.com/pariidanDKE/Toward-Practical-PII-Unlearning.git
fi

echo "Copying data files..."
mkdir -p data/raw data/test data/tofu

# PII data → data/raw/
cp Toward-Practical-PII-Unlearning/data/PII/full_with_qa.json data/raw/
cp Toward-Practical-PII-Unlearning/data/PII/forget1.json data/raw/
cp Toward-Practical-PII-Unlearning/data/PII/forget5.json data/raw/
cp Toward-Practical-PII-Unlearning/data/PII/forget10.json data/raw/
cp Toward-Practical-PII-Unlearning/data/PII/retain90.json data/raw/
cp Toward-Practical-PII-Unlearning/data/PII/retain95.json data/raw/
cp Toward-Practical-PII-Unlearning/data/PII/retain99.json data/raw/
cp Toward-Practical-PII-Unlearning/data/PII/full_validation.json data/raw/
cp Toward-Practical-PII-Unlearning/data/idontknow.jsonl data/raw/

# Test/eval data → data/test/
cp Toward-Practical-PII-Unlearning/data/test/test_retain_pii.json data/test/
cp Toward-Practical-PII-Unlearning/data/test/real_authors_perturbed.json data/test/
cp Toward-Practical-PII-Unlearning/data/test/world_facts_perturbed.json data/test/
cp -r Toward-Practical-PII-Unlearning/data/test/targeted_extraction data/test/ 2>/dev/null || true
# TOFU retain test set (400 mẫu, có paraphrased + perturbed fields — dùng cho eval retain)
cp Toward-Practical-PII-Unlearning/data/test/unused_test/test_retain_tofu.json data/test/

# TOFU data → data/tofu/
# forget/retain splits (có paraphrased + perturbed fields, cần cho eval)
cp Toward-Practical-PII-Unlearning/data/TOFU/forget01.json data/tofu/
cp Toward-Practical-PII-Unlearning/data/TOFU/forget05.json data/tofu/
cp Toward-Practical-PII-Unlearning/data/TOFU/forget10.json data/tofu/
cp Toward-Practical-PII-Unlearning/data/TOFU/retain90.json data/tofu/
cp Toward-Practical-PII-Unlearning/data/TOFU/retain95.json data/tofu/
cp Toward-Practical-PII-Unlearning/data/TOFU/retain99.json data/tofu/
cp Toward-Practical-PII-Unlearning/data/idontknow.jsonl data/tofu/   # dùng chung idk

# TOFU full.json → dùng cho SFT Exposed (phải học TẤT CẢ 4000 mẫu trước khi unlearn)
# Download trực tiếp từ HuggingFace locuslab/TOFU subset="full"
echo "Downloading TOFU full split from HuggingFace..."
python3 - <<'EOF'
import json
from datasets import load_dataset

ds = load_dataset("locuslab/TOFU", "full", split="train")
data = [{"question": row["question"], "answer": row["answer"]} for row in ds]
with open("data/tofu/full.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"  Saved data/tofu/full.json ({len(data)} samples)")
EOF

# Cleanup
rm -rf Toward-Practical-PII-Unlearning

echo "Data setup complete!"
echo "  PII  → data/raw/  : $(ls data/raw/*.json 2>/dev/null | wc -l) files"
echo "  Test → data/test/ : $(ls data/test/*.json 2>/dev/null | wc -l) files"
echo "  TOFU → data/tofu/ : $(ls data/tofu/*.json 2>/dev/null | wc -l) files"
