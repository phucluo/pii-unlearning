# PII Unlearning

> SP26AI66 — FPT University Capstone Project

Machine Unlearning để xóa thông tin cá nhân (PII) khỏi LLMs, đánh giá trên hai benchmark: **UnlearnPII** và **TOFU**.

## Pipeline

```
[Base LLM] → (1) SFT Exposed → (2) Unlearning → (3) Evaluation → [Metrics]
```

| | **TOFU** | **UnlearnPII** |
|--|----------|----------------|
| Data | 4.000 QA — 200 tác giả hư cấu | QA — 225 người, 17 loại PII |
| Default model | `qwen2.5-1.5b` | `llama2-7b-base` |
| Config prefix | `tofu_*.yaml` | `pii_*.yaml` |

## Cấu trúc

```
configs/             YAML cho SFT / Unlearn / Eval / AAU
src/                 data_module, trainers, aau_pii, utils
scripts/             setup_data.sh + pipeline scripts
notebooks/           Analysis notebooks
results_artifacts/   CSV kết quả (source of truth)
train.py             Bước 1 + 2
evaluate.py          Bước 3
```

## Setup

```bash
pip install -r requirements.txt
bash scripts/setup_data.sh
```

## Cách chạy

```bash
# 1. SFT Exposed
python train.py --config configs/pii_sft.yaml

# 2. Unlearning (đổi --forget_loss và --split tùy ý)
python train.py --config configs/pii_unlearn.yaml \
  --model_path=outputs/sft_exposed/pii/llama2-7b-base \
  --forget_loss=npo --split=forget10

# 2b. AAU-PII (proposed)
python train.py --config configs/pii_aau.yaml \
  --model_path=<unlearned checkpoint>

# 3. Evaluation
python evaluate.py --config configs/pii_eval.yaml \
  --model_path=<checkpoint>
```

Thay `pii_*` bằng `tofu_*` để chạy track TOFU.

## Methods

| Method | `--forget_loss` |
|--------|-----------------|
| Gradient Ascent | `grad_ascent` |
| Gradient Difference | `grad_diff` |
| NPO | `npo` |
| DPO | `dpo` |
| Task Vector Negation | `task_vector` |
| **AAU-PII** *(proposed)* | mode riêng — `configs/pii_aau.yaml` |

## Acknowledgements

Built upon [UnlearnPII](https://github.com/pariidanDKE/Toward-Practical-PII-Unlearning) (Parii Dan et al.) và [TOFU](https://huggingface.co/datasets/locuslab/TOFU) (Maini et al., 2024).
