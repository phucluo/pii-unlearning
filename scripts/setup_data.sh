#!/bin/bash
# Pull UnlearnPII data files + TOFU full split.
# Usage: bash scripts/setup_data.sh
set -e

REPO=Toward-Practical-PII-Unlearning

if [ ! -d "$REPO" ]; then
    echo "Cloning UnlearnPII repo"
    git clone https://github.com/pariidanDKE/Toward-Practical-PII-Unlearning.git
fi

mkdir -p data/raw data/test data/tofu

# PII training/eval data
cp $REPO/data/PII/full_with_qa.json data/raw/
cp $REPO/data/PII/forget1.json data/raw/
cp $REPO/data/PII/forget5.json data/raw/
cp $REPO/data/PII/forget10.json data/raw/
cp $REPO/data/PII/retain90.json data/raw/
cp $REPO/data/PII/retain95.json data/raw/
cp $REPO/data/PII/retain99.json data/raw/
cp $REPO/data/PII/full_validation.json data/raw/
cp $REPO/data/idontknow.jsonl data/raw/

cp $REPO/data/test/test_retain_pii.json data/test/
cp $REPO/data/test/real_authors_perturbed.json data/test/
cp $REPO/data/test/world_facts_perturbed.json data/test/
cp -r $REPO/data/test/targeted_extraction data/test/ 2>/dev/null || true

mkdir -p data/raw/split_person_names
cp $REPO/data/PII/split_person_names/*.json data/raw/split_person_names/
cp $REPO/data/PII/full_user_profiles.json data/raw/

# TOFU retain test set (used for eval)
cp $REPO/data/test/unused_test/test_retain_tofu.json data/test/

# TOFU forget/retain splits
cp $REPO/data/TOFU/forget01.json data/tofu/
cp $REPO/data/TOFU/forget05.json data/tofu/
cp $REPO/data/TOFU/forget10.json data/tofu/
cp $REPO/data/TOFU/retain90.json data/tofu/
cp $REPO/data/TOFU/retain95.json data/tofu/
cp $REPO/data/TOFU/retain99.json data/tofu/
cp $REPO/data/idontknow.jsonl data/tofu/

# TOFU full split is needed for SFT Exposed and isn't in the upstream repo.
echo "Downloading TOFU full split from HuggingFace"
python3 - <<'EOF'
import json
from datasets import load_dataset

ds = load_dataset("locuslab/TOFU", "full", split="train")
data = [{"question": row["question"], "answer": row["answer"]} for row in ds]
with open("data/tofu/full.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"  Saved data/tofu/full.json ({len(data)} samples)")
EOF

rm -rf $REPO

echo "Data setup complete"
echo "  PII  -> data/raw/  : $(ls data/raw/*.json 2>/dev/null | wc -l) files"
echo "  Test -> data/test/ : $(ls data/test/*.json 2>/dev/null | wc -l) files"
echo "  TOFU -> data/tofu/ : $(ls data/tofu/*.json 2>/dev/null | wc -l) files"
