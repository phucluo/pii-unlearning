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

# Cleanup
rm -rf Toward-Practical-PII-Unlearning

echo "Data setup complete! Files in data/raw/ and data/test/"
ls -la data/raw/
ls -la data/test/
