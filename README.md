# PII Unlearning

> SP26AI66 — FPT University Capstone Project

Nghiên cứu và triển khai các kỹ thuật **Machine Unlearning** để xóa thông tin cá nhân (PII) ra khỏi Large Language Models, đánh giá trên hai benchmark: **UnlearnPII** và **TOFU**.

---

## Pipeline

Cả hai track đều chạy **cùng 3 bước**, chỉ khác nhau ở config và data:

```
[Base LLM]
    │
    ▼  Bước 1 — SFT Exposed (train.py)
[SFT Model]          ← fine-tune toàn bộ data → model "nhớ" thông tin
    │
    ▼  Bước 2 — Unlearning (train.py)
[Unlearned Model]    ← áp dụng unlearning method → model "quên" forget set
    │
    ▼  Bước 3 — Evaluation (evaluate.py)
[Metrics]            ← đo độ quên (forget↓) và giữ retain (retain↑)
```

---

## Hai track benchmark

|  | **TOFU** | **UnlearnPII** |
|--|----------|----------------|
| Dataset | 4 000 QA về 200 tác giả hư cấu | QA về 225 người với 17 loại PII |
| Forget set | `forget01` / `forget05` / `forget10` | `forget1` / `forget5` / `forget10` |
| Data dir | `data/tofu/` | `data/raw/` |
| Config prefix | `configs/tofu_*.yaml` | `configs/pii_*.yaml` |
| Default model | `qwen2.5-1.5b` (BASE) | `llama2-7b-base` |
| Metric chính | ROUGE-L, PPL trên forget/retain | ROUGE-L, PPL, PII Leakage Rate |

---

## Cấu trúc repo

```
pii-unlearning/
├── configs/
│   ├── model_config.yaml      ← Model registry (HF key + prompt format)
│   │
│   ├── tofu_sft.yaml          ← Bước 1: SFT — TOFU track
│   ├── tofu_unlearn.yaml      ← Bước 2: Unlearning — TOFU track
│   ├── tofu_eval.yaml         ← Bước 3: Eval — TOFU track
│   │
│   ├── pii_sft.yaml           ← Bước 1: SFT — UnlearnPII track
│   ├── pii_unlearn.yaml       ← Bước 2: Unlearning — UnlearnPII track
│   └── pii_eval.yaml          ← Bước 3: Eval — UnlearnPII track
│
├── src/
│   ├── data_module.py         ← Dataset + tokenization (on-the-fly)
│   ├── trainers.py            ← Loss functions: GA, NPO, DPO, ...
│   └── utils.py               ← Config loading, model/LoRA setup
│
├── scripts/
│   ├── setup_data.sh          ← Tải PII + TOFU data
│   ├── run_tofu_pipeline.sh   ← Full pipeline TOFU (1 lệnh)
│   └── run_pii_pipeline.sh    ← Full pipeline UnlearnPII (1 lệnh)
│
├── data/
│   ├── raw/                   ← UnlearnPII data  (gitignored)
│   ├── tofu/                  ← TOFU data        (gitignored)
│   └── test/                  ← Shared eval data (gitignored)
│
├── outputs/                   ← Checkpoints + kết quả (gitignored)
├── notebooks/                 ← EDA, visualization
├── train.py                   ← Entry point: Bước 1 + Bước 2
├── evaluate.py                ← Entry point: Bước 3
└── requirements.txt
```

---

## Bắt đầu nhanh

### 1. Cài đặt

```bash
git clone https://github.com/phucluo/pii-unlearning.git
cd pii-unlearning
pip install -r requirements.txt
```

### 2. Tải data (chạy 1 lần duy nhất)

```bash
bash scripts/setup_data.sh
```

Script tự động clone UnlearnPII repo, copy JSON files vào `data/`, tải TOFU full split từ HuggingFace, rồi xóa clone.

### 3. Chạy full pipeline (1 lệnh)

```bash
# TOFU track
bash scripts/run_tofu_pipeline.sh <method> <split>
bash scripts/run_tofu_pipeline.sh npo forget10

# UnlearnPII track
bash scripts/run_pii_pipeline.sh <method> <split>
bash scripts/run_pii_pipeline.sh npo forget10
```

---

## Hướng dẫn từng bước

### Bước 1 — SFT Exposed

Mục tiêu: model học thuộc toàn bộ data trước khi unlearn.

```bash
# TOFU
python train.py --config configs/tofu_sft.yaml

# UnlearnPII
python train.py --config configs/pii_sft.yaml
```

Override GPU preset (xem bảng GPU bên dưới):

```bash
python train.py --config configs/tofu_sft.yaml \
  --batch_size=4 --gradient_accumulation_steps=8
```

**SFT tốt khi nào?** Kiểm tra avg loss cuối epoch 5:

| Loss | Đánh giá |
|------|----------|
| < 1.0 | ✅ Memorization tốt |
| 1.0 – 1.3 | ⚠️ Trung bình, unlearning vẫn chạy được |
| > 1.3 | ❌ Chưa đủ — tăng `lora.r` hoặc `num_epochs` |

---

### Bước 2 — Unlearning

Mục tiêu: model quên forget set, giữ nguyên retain set.

```bash
# TOFU — NPO, forget 10%
python train.py --config configs/tofu_unlearn.yaml \
  --model_family=qwen2.5-1.5b \
  --model_path=outputs/sft_exposed/tofu/qwen2.5-1.5b \
  --forget_loss=npo --split=forget10

# UnlearnPII — NPO, forget 10%
python train.py --config configs/pii_unlearn.yaml \
  --model_family=llama2-7b-base \
  --model_path=outputs/sft_exposed/pii/llama2-7b-base \
  --forget_loss=npo --split=forget10
```

Đổi method bằng `--forget_loss`:

```bash
--forget_loss=grad_ascent   # hoặc grad_diff | npo | dpo | task_vector | aau_pii
```

---

### Bước 3 — Evaluation

```bash
# TOFU
python evaluate.py --config configs/tofu_eval.yaml \
  --model_family=qwen2.5-1.5b \
  --model_path=outputs/unlearn/npo/forget10/tofu/qwen2.5-1.5b

# UnlearnPII
python evaluate.py --config configs/pii_eval.yaml \
  --model_family=llama2-7b-base \
  --model_path=outputs/unlearn/npo/forget10/llama2-7b-base
```

Kết quả lưu tại `<model_path>/eval_results/eval_log_aggregated.json`.

---

## Chạy trên Google Colab

### GPU preset

| GPU | VRAM | Qwen2.5-1.5B | Llama2-7B |
|-----|------|-------------|-----------|
| A100 | 40 GB | `batch_size=16` (default) | `batch_size=8 --quantization=4bit` |
| **L4** | 22.5 GB | `batch_size=4 --gradient_accumulation_steps=8` | `batch_size=4 --quantization=4bit` |
| T4 | 16 GB | `batch_size=2 --gradient_accumulation_steps=16 --bf16=false` | `batch_size=2 --quantization=4bit --bf16=false` |

> Tất cả preset giữ effective batch size = 32.

### Template notebook

```python
# Cell 1 — Setup
!git clone https://github.com/phucluo/pii-unlearning.git
%cd pii-unlearning
!pip install -q -r requirements.txt

# Cell 2 — Data
!bash scripts/setup_data.sh

# ── TOFU track ──────────────────────────────────────────────

# Cell 3a — Bước 1: SFT (L4 + Qwen2.5-1.5B)
!PYTORCH_ALLOC_CONF=expandable_segments:True \
  python train.py --config configs/tofu_sft.yaml \
  --batch_size=4 --gradient_accumulation_steps=8

# Cell 4a — Bước 2: Unlearning
!python train.py --config configs/tofu_unlearn.yaml \
  --model_family=qwen2.5-1.5b \
  --model_path=outputs/sft_exposed/tofu/qwen2.5-1.5b \
  --forget_loss=npo --split=forget10

# Cell 5a — Bước 3: Evaluation
!python evaluate.py --config configs/tofu_eval.yaml \
  --model_family=qwen2.5-1.5b \
  --model_path=outputs/unlearn/npo/forget10/tofu/qwen2.5-1.5b

# ── UnlearnPII track ─────────────────────────────────────────

# Cell 3b — Bước 1: SFT (L4 + Llama2-7B)
!python train.py --config configs/pii_sft.yaml \
  --quantization=4bit --batch_size=4 --gradient_accumulation_steps=8

# Cell 4b — Bước 2: Unlearning
!python train.py --config configs/pii_unlearn.yaml \
  --model_family=llama2-7b-base \
  --model_path=outputs/sft_exposed/pii/llama2-7b-base \
  --forget_loss=npo --split=forget10

# Cell 5b — Bước 3: Evaluation
!python evaluate.py --config configs/pii_eval.yaml \
  --model_family=llama2-7b-base \
  --model_path=outputs/unlearn/npo/forget10/llama2-7b-base
```

---

## Các unlearning method

| Method | `--forget_loss` | Mô tả |
|--------|----------------|-------|
| Gradient Ascent | `grad_ascent` | Đảo ngược CE loss trên forget set |
| Gradient Difference | `grad_diff` | GA + retain regularization |
| NPO | `npo` | Negative Preference Optimization |
| DPO | `dpo` | Direct Preference Optimization |
| Task Vector | `task_vector` | Subtract LoRA weights sau training |
| AAU-PII | `aau_pii` | Adaptive Adversarial Unlearning (proposed) |

---

## Các model được hỗ trợ

| `model_family` | Hugging Face ID | Ghi chú |
|----------------|----------------|---------|
| `qwen2.5-1.5b` | Qwen/Qwen2.5-1.5B | Default TOFU — BASE model |
| `qwen2.5-7b` | Qwen/Qwen2.5-7B | Cần `--quantization=4bit` |
| `llama2-7b-base` | meta-llama/Llama-2-7b-hf | Default PII — cần HF token |

Thêm model mới: chỉnh `configs/model_config.yaml`.

---

## Acknowledgements

Built upon [UnlearnPII](https://github.com/pariidanDKE/Toward-Practical-PII-Unlearning) (Parii Dan et al.) and [TOFU](https://huggingface.co/datasets/locuslab/TOFU) (Maini et al., 2024).
