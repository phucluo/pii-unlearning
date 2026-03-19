# CLAUDE.md — Project Context

## Nhóm & Mục tiêu

- **Nhóm:** GSP26AI23
- **Mục tiêu:** Nghiên cứu và publish paper về **Machine Unlearning cho PII (Personally Identifiable Information)** trong LLMs
- **Bài toán:** Sau khi LLM được fine-tune trên dữ liệu có chứa PII, làm thế nào để "xóa" thông tin nhạy cảm đó khỏi model mà không cần retrain từ đầu

---

## Phương pháp đề xuất

**AAU-PII** (tên working title) — phương pháp unlearning tập trung vào PII, được so sánh với các baseline:

| Baseline | Mô tả |
|---|---|
| **GA** (Gradient Ascent) | Tối đa hóa loss trên forget set để model "quên" |
| **GD** (Gradient Difference) | GA + giữ performance trên retain set |
| **NPO** (Negative Preference Optimization) | Dựa trên DPO, đẩy output của forget set ra xa |
| **DPO** (Direct Preference Optimization) | Dùng preferred/rejected pairs để steer model |
| **Task Vector** | Trừ weight delta của forget task khỏi model |

---

## PEFT Strategy

- **LoRA / QLoRA** cho tất cả các bước (SFT + Unlearn)
- Lý do: hạ tầng giới hạn (Kaggle), không full fine-tune toàn bộ model
- Target modules: attention layers (`q_proj`, `v_proj`, ...)
- Rank thấp (r=8 hoặc r=16) để tiết kiệm VRAM

---

## Benchmarks & Datasets

| Benchmark | Mô tả |
|---|---|
| **TOFU** | Forget fictional author profiles; chuẩn đánh giá unlearning phổ biến |
| **UnlearnPII** | Dataset chuyên cho PII unlearning (email, phone, address, ...) |
- TOFU: forget01/05/10, data ở data/tofu/
- UnlearnPII: data ở data/unlearnpii/ (clone từ pariidanDKE repo)

- **Forget set:** tập dữ liệu chứa PII cần xóa
- **Retain set:** tập dữ liệu sạch cần giữ nguyên performance
- **Evaluation metrics:** Forget Quality, Model Utility, ROUGE, Probability

---

## Model chính

- **Llama 2 7B base** (`meta-llama/Llama-2-7b-hf`) — model chính dùng trong experiments
- Các model phụ (ablation / quick test): `Qwen2.5-1.5B-Instruct`
- Chat format: xem `configs/model_config.yaml`

---

## Hạ tầng

- **Chạy trên Kaggle** (2x T4 hoặc P100, ~16GB VRAM)
- Không có GPU server riêng — mọi experiment cần tối ưu memory
- Dùng QLoRA (4-bit quantization) khi VRAM không đủ cho LoRA thường
- Batch size nhỏ + gradient accumulation

---

## Cấu trúc repo

```
configs/          # YAML config cho model, SFT, unlearn, eval
src/
  data_module.py  # Dataset, DataLoader, prompt formatting
  trainers.py     # SFT trainer, Unlearn trainer (GA/GD/NPO/...)
  utils.py        # Helper functions
train.py          # Entry point: SFT hoặc unlearning
evaluate.py       # Entry point: evaluation
scripts/          # Shell scripts chạy pipeline
```

---

## Proposed method
AAU-PII (Adaptive Adversarial Unlearning for PII)
→ implement vào src/trainers.py class AauPIITrainer

---

## Pipeline chuẩn

1. **SFT** — fine-tune model trên dữ liệu có PII (tạo "model đã bị nhiễm")
2. **Unlearn** — áp dụng phương pháp unlearning (baseline hoặc AAU-PII)
3. **Eval** — đo Forget Quality + Model Utility trên TOFU/UnlearnPII

---

## Lưu ý khi làm việc

- Ưu tiên code chạy được trên Kaggle trước khi optimize
- Mọi experiment nên log đầy đủ metrics để so sánh baselines
- Config-driven: thêm model/method mới qua YAML, không hardcode
- Paper target: so sánh định lượng AAU-PII vs tất cả baselines trên cả hai benchmarks

---

## DO NOT
- Không dùng Hydra, DeepSpeed, WandB
- Không tạo JSONL preprocessing step riêng
- Không thay AutoModel bằng custom model class