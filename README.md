# PII Unlearning — Machine Unlearning for PII in LLMs

> GSP26AI23 — FPT University Capstone Project (SP26AI66)

Research and Implementation of Machine Unlearning Techniques to Remove Personally Identifiable Information (PII) in Large Language Models.

## Repo Structure

```
pii-unlearning/
├── configs/
│   ├── model_config.yaml    ← Model registry (chat templates)
│   ├── sft.yaml             ← SFT Exposed config
│   ├── unlearn.yaml         ← Unlearning config
│   └── eval.yaml            ← Evaluation config
├── src/
│   ├── data_module.py       ← Dataset + tokenization (on-the-fly)
│   ├── trainers.py          ← Loss functions (GA, NPO, DPO, ...)
│   └── utils.py             ← Config loading, model setup
├── scripts/
│   └── run_pipeline.sh      ← Full pipeline in one command
├── data/
│   ├── raw/                 ← UnlearnPII JSON files (gitignored)
│   └── test/                ← Eval-only data (gitignored)
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
Copy UnlearnPII data files into `data/`:
```bash
# From the UnlearnPII repo:
cp -r Toward-Practical-PII-Unlearning/data/PII/* data/raw/
cp -r Toward-Practical-PII-Unlearning/data/test/* data/test/
cp Toward-Practical-PII-Unlearning/data/idontknow.jsonl data/raw/
```

### 3. Run Full Pipeline
```bash
# One command: SFT → GA unlearning → Eval
bash scripts/run_pipeline.sh grad_ascent forget10

# Try different methods:
bash scripts/run_pipeline.sh npo forget10
bash scripts/run_pipeline.sh dpo forget10
```

### 4. Individual Steps
```bash
# SFT only
python train.py --config configs/sft.yaml

# Unlearn only
python train.py --config configs/unlearn.yaml --forget_loss=npo

# Eval only
python evaluate.py --config configs/eval.yaml --model_path=outputs/unlearn/npo/forget10
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
