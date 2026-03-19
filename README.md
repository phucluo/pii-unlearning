# PII Unlearning

> SP26AI66 — FPT University Capstone Project

Nghiên cứu và triển khai các kỹ thuật **Machine Unlearning** để xóa thông tin cá nhân (PII) ra khỏi Large Language Models, đánh giá trên hai benchmark: **UnlearnPII** và **TOFU**.

---

## Pipeline (3 bước — dùng chung cho cả 2 track)

```
[Base LLM]
    │  Bước 1: SFT Exposed      → model "nhớ" toàn bộ data
    │  Bước 2: Unlearning       → model "quên" forget set, giữ retain set
    │  Bước 3: Evaluation       → đo độ quên (forget↓) và giữ retain (retain↑)
    ▼
[Metrics]
```

**Hai track** chạy cùng codebase, chỉ khác config và data:

| | **TOFU** | **UnlearnPII** |
|--|----------|----------------|
| Data | 4 000 QA — 200 tác giả hư cấu | QA — 225 người, 17 loại PII |
| Forget split | `forget01` / `forget05` / `forget10` | `forget1` / `forget5` / `forget10` |
| Config | `configs/tofu_*.yaml` | `configs/pii_*.yaml` |
| Data dir | `data/tofu/` | `data/raw/` |
| Default model | `qwen2.5-1.5b` | `llama2-7b-base` |

---

## Cấu trúc repo

```
pii-unlearning/
├── configs/
│   ├── model_config.yaml          ← Model registry
│   ├── tofu_{sft,unlearn,eval}.yaml
│   └── pii_{sft,unlearn,eval}.yaml
├── src/
│   ├── data_module.py             ← Dataset + tokenization
│   ├── trainers.py                ← Loss functions (GA, NPO, DPO, ...)
│   └── utils.py                   ← Config, model/LoRA setup
├── scripts/
│   ├── setup_data.sh              ← Tải data (chạy 1 lần)
│   ├── run_tofu_pipeline.sh       ← Full pipeline TOFU
│   └── run_pii_pipeline.sh        ← Full pipeline UnlearnPII
├── train.py                       ← Entry point: Bước 1 + 2
├── evaluate.py                    ← Entry point: Bước 3
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/phucluo/pii-unlearning.git
cd pii-unlearning
pip install -r requirements.txt
bash scripts/setup_data.sh        # tải data — chạy 1 lần duy nhất
```

---

## Chạy trên Colab / Kaggle

### Chọn GPU preset

| GPU | VRAM | Platform | Qwen2.5-1.5B | Llama2-7B |
|-----|------|----------|-------------|-----------|
| A100 | 40 GB | Colab Pro | `batch_size=16` *(default)* | `--batch_size=8 --quantization=4bit` |
| L4 | 22.5 GB | Colab Pro | `--batch_size=4 --gradient_accumulation_steps=8` | `--batch_size=4 --quantization=4bit` |
| T4 | 16 GB | Colab / Kaggle | `--batch_size=2 --gradient_accumulation_steps=16 --bf16=false` | `--batch_size=2 --quantization=4bit --bf16=false` |
| P100 | 16 GB | Kaggle | `--batch_size=2 --gradient_accumulation_steps=16 --bf16=false` | `--batch_size=2 --quantization=4bit --bf16=false` |

> Tất cả preset giữ **effective batch = 32** (`batch_size × gradient_accumulation_steps`).
> **Kaggle**: vào *Settings → Internet → On* trước khi chạy.

### Template notebook (Colab & Kaggle đều dùng được)

```python
# Cell 1 — Setup
!git clone https://github.com/phucluo/pii-unlearning.git
%cd pii-unlearning
!pip install -q -r requirements.txt
!bash scripts/setup_data.sh

# ── TOFU track (Qwen2.5-1.5B, T4/P100) ──────────────────────
# Bước 1: SFT
!PYTORCH_ALLOC_CONF=expandable_segments:True \
  python train.py --config configs/tofu_sft.yaml \
  --batch_size=2 --gradient_accumulation_steps=16 --bf16=false

# Bước 2: Unlearning
!python train.py --config configs/tofu_unlearn.yaml \
  --model_family=qwen2.5-1.5b \
  --model_path=outputs/sft_exposed/tofu/qwen2.5-1.5b \
  --forget_loss=npo --split=forget10

# Bước 3: Eval
!python evaluate.py --config configs/tofu_eval.yaml \
  --model_family=qwen2.5-1.5b \
  --model_path=outputs/unlearn/npo/forget10/tofu/qwen2.5-1.5b

# ── UnlearnPII track (Llama2-7B, T4/P100) ───────────────────
# Bước 1: SFT
!python train.py --config configs/pii_sft.yaml \
  --quantization=4bit --batch_size=2 --gradient_accumulation_steps=16 --bf16=false

# Bước 2: Unlearning
!python train.py --config configs/pii_unlearn.yaml \
  --model_family=llama2-7b-base \
  --model_path=outputs/sft_exposed/pii/llama2-7b-base \
  --forget_loss=npo --split=forget10

# Bước 3: Eval
!python evaluate.py --config configs/pii_eval.yaml \
  --model_family=llama2-7b-base \
  --model_path=outputs/unlearn/npo/forget10/llama2-7b-base
```

> **Lưu kết quả trên Kaggle** (session không lưu tự động):
> ```python
> from google.colab import files   # Colab
> # Kaggle: copy sang /kaggle/working/ rồi commit output
> import shutil
> shutil.copytree("outputs", "/kaggle/working/outputs")
> ```

---

## Chạy từng bước

### Bước 1 — SFT Exposed

```bash
python train.py --config configs/tofu_sft.yaml    # TOFU
python train.py --config configs/pii_sft.yaml     # UnlearnPII
```

SFT đủ tốt khi avg loss cuối epoch 5: **< 1.0** (tốt) / 1.0–1.3 (trung bình) / **> 1.3** (chưa đủ).

### Bước 2 — Unlearning

```bash
# Đổi method: --forget_loss= grad_ascent | grad_diff | npo | dpo | task_vector | aau_pii
# Đổi split:  --split= forget01/05/10 (TOFU) hoặc forget1/5/10 (PII)

python train.py --config configs/tofu_unlearn.yaml \
  --model_family=qwen2.5-1.5b \
  --model_path=outputs/sft_exposed/tofu/qwen2.5-1.5b \
  --forget_loss=npo --split=forget10
```

### Bước 3 — Evaluation

```bash
python evaluate.py --config configs/tofu_eval.yaml \
  --model_family=qwen2.5-1.5b \
  --model_path=outputs/unlearn/npo/forget10/tofu/qwen2.5-1.5b
```

Kết quả: `<model_path>/eval_results/eval_log_aggregated.json`

---

## Unlearning methods

| Method | `--forget_loss` | Mô tả |
|--------|----------------|-------|
| Gradient Ascent | `grad_ascent` | Đảo ngược CE loss trên forget set |
| Gradient Difference | `grad_diff` | GA + retain regularization |
| NPO | `npo` | Negative Preference Optimization |
| DPO | `dpo` | Direct Preference Optimization |
| Task Vector | `task_vector` | Subtract LoRA weights sau training |
| AAU-PII | `aau_pii` | Adaptive Adversarial Unlearning *(proposed)* |

---

## Models được hỗ trợ

| `--model_family` | Hugging Face ID | Ghi chú |
|----------------|----------------|---------|
| `qwen2.5-1.5b` | Qwen/Qwen2.5-1.5B | Default TOFU — BASE |
| `qwen2.5-7b` | Qwen/Qwen2.5-7B | Cần `--quantization=4bit` |
| `llama2-7b-base` | meta-llama/Llama-2-7b-hf | Default PII — cần HF token |

Thêm model mới: chỉnh `configs/model_config.yaml`.

---

## Acknowledgements

Built upon [UnlearnPII](https://github.com/pariidanDKE/Toward-Practical-PII-Unlearning) (Parii Dan et al.) and [TOFU](https://huggingface.co/datasets/locuslab/TOFU) (Maini et al., 2024).
