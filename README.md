# PII Unlearning

> SP26AI66 — FPT University Capstone Project

Nghiên cứu và triển khai các kỹ thuật **Machine Unlearning** để xóa thông tin cá nhân (PII) ra khỏi Large Language Models, đánh giá trên hai benchmark: **UnlearnPII** và **TOFU**.

---

## Pipeline tổng quan

```
[Base LLM]
    │
    ▼  Step 1 — SFT Exposed
[SFT Model]          ← fine-tune trên toàn bộ data (model "nhớ" thông tin)
    │
    ▼  Step 2 — Unlearning
[Unlearned Model]    ← áp dụng unlearning method để "quên" forget set
    │
    ▼  Step 3 — Evaluation
[Metrics]            ← đo mức độ quên (forget) và giữ nguyên (retain)
```

Repo hỗ trợ **2 track** chạy song song, cùng codebase:

| Track | Dataset | Mục tiêu forget |
|-------|---------|----------------|
| **TOFU** | 4000 QA về 200 tác giả hư cấu | 10% (400 samples) |
| **UnlearnPII** | 225 người với 17 loại PII | 1% / 5% / 10% |

---

## Cấu trúc repo

```
pii-unlearning/
├── configs/
│   ├── model_config.yaml    ← Model registry (HF key + chat template tags)
│   ├── tofu_sft.yaml        ← Step 1: SFT config cho TOFU
│   ├── tofu_unlearn.yaml    ← Step 2: Unlearning config cho TOFU
│   ├── tofu_eval.yaml       ← Step 3: Eval config cho TOFU
│   ├── pii_sft.yaml         ← Step 1: SFT config cho UnlearnPII
│   ├── pii_unlearn.yaml     ← Step 2: Unlearning config cho UnlearnPII
│   └── pii_eval.yaml        ← Step 3: Eval config cho UnlearnPII
├── src/
│   ├── data_module.py       ← Dataset + tokenization (on-the-fly)
│   ├── trainers.py          ← Loss functions: GA, NPO, DPO, ...
│   └── utils.py             ← Config loading, model/LoRA setup
├── scripts/
│   ├── setup_data.sh        ← Tải PII + TOFU data về local
│   ├── run_pii_pipeline.sh  ← Chạy full pipeline UnlearnPII
│   └── run_tofu_pipeline.sh ← Chạy full pipeline TOFU
├── data/
│   ├── raw/                 ← UnlearnPII data (gitignored — tải bằng setup_data.sh)
│   ├── tofu/                ← TOFU data (gitignored — tải bằng setup_data.sh)
│   └── test/                ← Eval data (gitignored)
├── outputs/                 ← Checkpoints + kết quả (gitignored)
├── notebooks/               ← EDA, visualization
├── train.py                 ← Entry point: Step 1 (SFT) + Step 2 (Unlearn)
├── evaluate.py              ← Entry point: Step 3 (Eval)
└── requirements.txt
```

---

## Cài đặt

```bash
git clone https://github.com/phucluo/pii-unlearning.git
cd pii-unlearning
pip install -r requirements.txt
```

Tải data (cần chạy **1 lần duy nhất** trên mỗi môi trường):

```bash
bash scripts/setup_data.sh
```

Script này clone repo UnlearnPII, copy các file JSON cần thiết vào `data/`, rồi tự xóa clone.

---

## Chạy trên Google Colab

### Khuyến nghị GPU theo model

| GPU | VRAM | Qwen2.5-1.5B | Llama2-7B |
|-----|------|--------------|-----------|
| A100 | 40 GB | `batch_size=16` (default) | `batch_size=8 --quantization=4bit` |
| **L4** | 22.5 GB | `batch_size=4` | `batch_size=4 --quantization=4bit` |
| T4 | 16 GB | `batch_size=2 --bf16=false` | `batch_size=2 --quantization=4bit --bf16=false` |

> Tất cả preset giữ **effective batch size = 32** (`batch_size × gradient_accumulation_steps`).

### Notebook setup (copy vào Colab)

```python
# Cell 1 — Setup
!git clone https://github.com/phucluo/pii-unlearning.git
%cd pii-unlearning
!pip install -q -r requirements.txt

# Cell 2 — Data
!bash scripts/setup_data.sh

# Cell 3 — Step 1: SFT Exposed (L4 + Qwen2.5-1.5B)
!PYTORCH_ALLOC_CONF=expandable_segments:True \
  python train.py --config configs/tofu_sft.yaml \
  --batch_size=4 --gradient_accumulation_steps=8

# Cell 4 — Step 2: Unlearning
!python train.py --config configs/tofu_unlearn.yaml \
  --model_family=qwen2.5-1.5b \
  --model_path=outputs/sft_exposed/tofu/qwen2.5-1.5b \
  --forget_loss=npo --split=forget10

# Cell 5 — Step 3: Evaluation
!python evaluate.py --config configs/tofu_eval.yaml \
  --model_family=qwen2.5-1.5b \
  --model_path=outputs/unlearn/npo/forget10/tofu/qwen2.5-1.5b
```

---

## Chạy từng bước (CLI)

### Step 1 — SFT Exposed

```bash
# TOFU track
python train.py --config configs/tofu_sft.yaml

# UnlearnPII track
python train.py --config configs/pii_sft.yaml

# Override bất kỳ tham số nào qua CLI
python train.py --config configs/tofu_sft.yaml --model_family=llama2-7b-base --quantization=4bit
```

**Khi nào SFT đủ tốt?** Kiểm tra loss cuối epoch 5:
- Loss < 1.0 → memorization tốt, sẵn sàng unlearn
- Loss 1.0–1.3 → trung bình, unlearning vẫn chạy được
- Loss > 1.3 → chưa đủ (tăng `lora.r` hoặc `num_epochs`)

### Step 2 — Unlearning

```bash
# TOFU — NPO method, forget 10%
python train.py --config configs/tofu_unlearn.yaml \
  --model_family=qwen2.5-1.5b \
  --model_path=outputs/sft_exposed/tofu/qwen2.5-1.5b \
  --forget_loss=npo --split=forget10

# Đổi method: grad_ascent | grad_diff | npo | dpo | task_vector
python train.py --config configs/tofu_unlearn.yaml --forget_loss=grad_ascent
```

### Step 3 — Evaluation

```bash
python evaluate.py --config configs/tofu_eval.yaml \
  --model_family=qwen2.5-1.5b \
  --model_path=outputs/unlearn/npo/forget10/tofu/qwen2.5-1.5b
```

Kết quả lưu tại `outputs/unlearn/npo/forget10/tofu/qwen2.5-1.5b/eval_results/eval_log_aggregated.json`.

---

## Full pipeline (1 lệnh)

```bash
# TOFU
bash scripts/run_tofu_pipeline.sh npo forget10
bash scripts/run_tofu_pipeline.sh grad_ascent forget05

# UnlearnPII
bash scripts/run_pii_pipeline.sh npo forget10
```

---

## Các unlearning method được hỗ trợ

| Method | `--forget_loss` | Mô tả |
|--------|----------------|-------|
| Gradient Ascent | `grad_ascent` | Đảo ngược CE loss trên forget set |
| Gradient Difference | `grad_diff` | GA + retain regularization |
| NPO | `npo` | Negative Preference Optimization |
| DPO | `dpo` | Direct Preference Optimization (cần IDK responses) |
| Task Vector | `task_vector` | Trừ LoRA weights sau training |
| AAU-PII | `aau_pii` | Adaptive Adversarial Unlearning (đề xuất mới) |

---

## Các model được hỗ trợ

| `model_family` | Model | Ghi chú |
|----------------|-------|---------|
| `qwen2.5-1.5b` | Qwen/Qwen2.5-1.5B | Default — BASE model |
| `qwen2.5-7b` | Qwen/Qwen2.5-7B | Cần `--quantization=4bit` trên L4/T4 |
| `llama2-7b-base` | meta-llama/Llama-2-7b-hf | Cần HF token |

Thêm model mới: chỉnh `configs/model_config.yaml`.

---

## Acknowledgements

Built upon the [UnlearnPII Benchmark](https://github.com/pariidanDKE/Toward-Practical-PII-Unlearning) (Parii Dan et al.) and [TOFU Benchmark](https://huggingface.co/datasets/locuslab/TOFU) (Maini et al., 2024).
