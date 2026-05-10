# PII Unlearning

> SP26AI66 — FPT University Capstone Project

Machine unlearning to remove personal information (PII) from LLMs, evaluated on
two benchmarks: **UnlearnPII** and **TOFU**.

## Pipeline

```
[Base LLM] -> (1) SFT Exposed -> (2) Unlearning -> (3) Evaluation -> [Metrics]
```

|              | **TOFU**                    | **UnlearnPII**              |
|--------------|-----------------------------|-----------------------------|
| Data         | 4,000 QA — 200 fictional authors | QA — 225 persons, 17 PII types |
| Default model| `qwen2.5-1.5b`              | `llama2-7b-base`            |
| Config prefix| `tofu_*.yaml`               | `pii_*.yaml`                |

## Layout

```
configs/             YAML for SFT / Unlearn / Eval / AAU
src/                 data_module, trainers, aau_pii, utils
scripts/             setup_data.sh + pipeline scripts
notebooks/           analysis notebooks
results_artifacts/   CSV results (source of truth)
train.py             steps 1 + 2
evaluate.py          step 3
```

## Setup

```bash
pip install -r requirements.txt
bash scripts/setup_data.sh
```

## Run

```bash
# 1. SFT Exposed
python train.py --config configs/pii_sft.yaml

# 2. Unlearning
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

Swap `pii_*` for `tofu_*` to run the TOFU track.

## Methods

| Method               | `--forget_loss`  |
|----------------------|------------------|
| Gradient Ascent      | `grad_ascent`    |
| Gradient Difference  | `grad_diff`      |
| NPO                  | `npo`            |
| DPO                  | `dpo`            |
| Task Vector Negation | `task_vector`    |
| **AAU-PII** *(proposed)* | dedicated mode — `configs/pii_aau.yaml` |

## Acknowledgements

Built upon [UnlearnPII](https://github.com/pariidanDKE/Toward-Practical-PII-Unlearning)
(Parii Dan et al.) and [TOFU](https://huggingface.co/datasets/locuslab/TOFU)
(Maini et al., 2024).
