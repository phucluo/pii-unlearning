# PII Unlearning — Machine Unlearning for PII in LLMs

> GSP26AI23 — FPT University Capstone Project (SP26AI66)

Research and Implementation of Machine Unlearning Techniques to Remove Personally Identifiable Information (PII) in Large Language Models.

## Repo Structure

```
pii-unlearning/
├── configs/
│   ├── model_config.yaml    ← Model registry (chat templates) — shared
│   ├── pii_sft.yaml         ← SFT Exposed config (UnlearnPII track)
│   ├── pii_unlearn.yaml     ← Unlearning config (UnlearnPII track)
│   ├── pii_eval.yaml        ← Evaluation config (UnlearnPII track)
│   ├── tofu_sft.yaml        ← SFT Exposed config (TOFU track)
│   ├── tofu_unlearn.yaml    ← Unlearning config (TOFU track)
│   └── tofu_eval.yaml       ← Evaluation config (TOFU track)
├── src/
│   ├── data_module.py       ← Dataset + tokenization (on-the-fly)
│   ├── trainers.py          ← Loss functions (GA, NPO, DPO, ...)
│   └── utils.py             ← Config loading, model setup
├── scripts/
│   ├── setup_data.sh        ← Clone UnlearnPII repo, copy PII + TOFU data
│   ├── run_pii_pipeline.sh  ← Full pipeline: UnlearnPII track
│   └── run_tofu_pipeline.sh ← Full pipeline: TOFU track
├── data/
│   ├── raw/                 ← UnlearnPII PII JSON files (gitignored)
│   ├── tofu/                ← TOFU JSON files (gitignored)
│   └── test/                ← Eval-only data — shared (gitignored)
├── outputs/                 ← Checkpoints + results (gitignored)
├── notebooks/               ← EDA, visualization
├── train.py                 ← Entry point: SFT + Unlearning
├── evaluate.py              ← Entry point: Evaluation
└── requirements.txt
```

## Quickstart

### 1. Setup
```bash
git clone https://github.com/<your-team>/pii-unlearning.git
cd pii-unlearning
pip install -r requirements.txt
```

### 2. Data
```bash
# Clone UnlearnPII repo → copy PII + TOFU data → xóa clone
bash scripts/setup_data.sh
```

### 3. Run Full Pipeline
```bash
# UnlearnPII track
bash scripts/run_pii_pipeline.sh grad_ascent forget10
bash scripts/run_pii_pipeline.sh npo forget10
bash scripts/run_pii_pipeline.sh aau_pii forget10

# TOFU track
bash scripts/run_tofu_pipeline.sh grad_ascent forget10
bash scripts/run_tofu_pipeline.sh npo forget05
bash scripts/run_tofu_pipeline.sh aau_pii forget10
```

### 4. Individual Steps
```bash
# SFT
python train.py --config configs/pii_sft.yaml
python train.py --config configs/tofu_sft.yaml

# Unlearn
python train.py --config configs/pii_unlearn.yaml --forget_loss=npo
python train.py --config configs/tofu_unlearn.yaml --forget_loss=npo

# Eval
python evaluate.py --config configs/pii_eval.yaml --model_path=outputs/unlearn/npo/forget10/llama2-7b-base
python evaluate.py --config configs/tofu_eval.yaml --model_path=outputs/unlearn/npo/forget10/tofu/llama2-7b-base
```

## Supported Methods

| Method | `forget_loss` | Description |
|--------|--------------|-------------|
| Gradient Ascent | `grad_ascent` | Negate CE loss on forget set |
| Gradient Difference | `grad_diff` | GA + retain regularization |
| NPO | `npo` | Negative Preference Optimization |
| DPO | `dpo` | Direct Preference Optimization |
| Task Vector | `task_vector` | Train → subtract LoRA weights |
| AAU-PII | `aau_pii` | Adaptive Adversarial Unlearning (proposed) |

## Benchmarks
- **TOFU**: 200 synthetic authors, comparability with related work
- **UnlearnPII**: 225 synthetic persons, 17 PII types, PII-specific metrics

## Acknowledgements
Built upon the [UnlearnPII Benchmark](https://github.com/pariidanDKE/Toward-Practical-PII-Unlearning) by Parii Dan.
