"""
evaluate.py — Evaluate unlearned model on PII metrics.
Mirrors UnlearnPII's evaluate_PII.py but simplified.

Usage:
  python evaluate.py --config configs/pii_eval.yaml
  python evaluate.py --config configs/tofu_eval.yaml
  python evaluate.py --config configs/pii_eval.yaml --model_path=outputs/unlearn/npo/forget10/llama2-7b-base
"""
import json
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils import parse_args, load_config, get_model_identifiers, load_model_and_tokenizer
from src.data_module import SFTDataset, sft_collator, convert_to_model_format
from src.trainers import get_batch_loss


# ========================= METRICS =========================

def compute_perplexity(model, dataloader, device):
    """Compute average perplexity over dataset."""
    model.eval()
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing PPL", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * input_ids.size(0)
            total_count += input_ids.size(0)
    avg_loss = total_loss / max(total_count, 1)
    return {"perplexity": float(np.exp(avg_loss)), "avg_loss": avg_loss, "n_samples": total_count}


def compute_rouge_and_generation(model, tokenizer, dataset, model_configs,
                                 max_new_tokens=128, device="cuda"):
    """Generate answers and compute ROUGE-L recall + exact match."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        print("WARNING: rouge_score not installed, skipping ROUGE. pip install rouge-score")
        scorer = None

    model.eval()
    results = []
    q_start = model_configs["question_start_tag"]
    q_end = model_configs["question_end_tag"]
    a_start = model_configs["answer_tag"]

    for i in tqdm(range(len(dataset)), desc="Generating", leave=False):
        item = dataset.data[i]
        question = item[dataset.question_key]
        gold_answer = item[dataset.answer_key]
        subject_pii = item.get("subject_pii", [])

        # Build prompt
        prompt = q_start + question + q_end + a_start
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only generated tokens
        gen_text = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        # ROUGE
        rouge_l = 0.0
        if scorer:
            rouge_l = scorer.score(gold_answer, gen_text)["rougeL"].recall

        # Exact match
        em = 1.0 if gold_answer.strip().lower() in gen_text.strip().lower() else 0.0

        # PII Leakage: check if any PII token appears in generation
        pii_leaked = 0
        for pii_val in subject_pii:
            if pii_val.lower() in gen_text.lower():
                pii_leaked = 1
                break

        results.append({
            "question": question,
            "gold": gold_answer,
            "generated": gen_text,
            "rouge_l": rouge_l,
            "exact_match": em,
            "pii_leaked": pii_leaked,
        })

    # Aggregate
    n = len(results)
    agg = {
        "rouge_l_mean": np.mean([r["rouge_l"] for r in results]) if n else 0,
        "exact_match_rate": np.mean([r["exact_match"] for r in results]) if n else 0,
        "pii_leakage_rate": np.mean([r["pii_leaked"] for r in results]) if n else 0,
        "n_samples": n,
    }
    return agg, results


# ========================= MAIN EVAL =========================

def run_eval(cfg):
    print("=" * 60)
    print(f"EVALUATION — model: {cfg['model_path']}")
    print("=" * 60)

    model_cfg = get_model_identifiers(cfg["model_family"])

    # Load model — handles both full models and PEFT/LoRA checkpoints
    model, tokenizer = load_model_and_tokenizer(cfg, model_cfg)

    device = model.device

    # Save dir
    save_dir = cfg.get("save_dir") or os.path.join(cfg["model_path"], "eval_results")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run each eval task
    for task_cfg in cfg.get("eval_tasks", []):
        task_name = task_cfg["name"]
        tier = task_cfg.get("tier", 1)
        print(f"\n--- [{task_name}] (Tier {tier}) ---")

        data_file = os.path.join(task_cfg["data_path"], f"{task_cfg['split']}.json")
        if not os.path.exists(data_file):
            print(f"  SKIP: {data_file} not found")
            continue

        ds = SFTDataset(
            data_file, tokenizer, cfg["model_family"],
            question_key=task_cfg.get("question_key", "question"),
            answer_key=task_cfg.get("answer_key", "answer"),
        )

        # Perplexity
        dl = torch.utils.data.DataLoader(
            ds, batch_size=cfg.get("batch_size", 16),
            collate_fn=sft_collator, shuffle=False,
        )
        ppl_results = compute_perplexity(model, dl, device)
        print(f"  PPL: {ppl_results['perplexity']:.2f}")

        # ROUGE + PII Leakage (generation-based)
        gen_agg, gen_details = compute_rouge_and_generation(
            model, tokenizer, ds, model_cfg,
            max_new_tokens=cfg.get("max_new_tokens", 128), device=device,
        )
        print(f"  ROUGE-L: {gen_agg['rouge_l_mean']:.4f}")
        print(f"  PII Leakage Rate: {gen_agg['pii_leakage_rate']:.4f}")

        all_results[task_name] = {**ppl_results, **gen_agg}

        # Save per-task generation details
        detail_path = os.path.join(save_dir, f"{task_name}_details.json")
        with open(detail_path, "w") as f:
            json.dump(gen_details, f, indent=2, ensure_ascii=False)

    # Save aggregated results
    agg_path = os.path.join(save_dir, "eval_log_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {agg_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Task':<35} {'PPL':>8} {'ROUGE-L':>10} {'PII Leak':>10}")
    print("-" * 80)
    for name, r in all_results.items():
        print(f"{name:<35} {r.get('perplexity', 0):>8.2f} "
              f"{r.get('rouge_l_mean', 0):>10.4f} "
              f"{r.get('pii_leakage_rate', 0):>10.4f}")
    print("=" * 80)


def main():
    config_path, overrides = parse_args()
    cfg = load_config(config_path, overrides)
    run_eval(cfg)


if __name__ == "__main__":
    main()
