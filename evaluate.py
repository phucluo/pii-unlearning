"""
evaluate.py — Evaluate unlearned model on TOFU / UnlearnPII metrics.
Follows the evaluation methodology of:
  https://github.com/pariidanDKE/Toward-Practical-PII-Unlearning/blob/main/evaluate_PII.py

Metrics implemented:
  - Per-token GT loss (avg_gt_loss)
  - ROUGE-L recall, ROUGE-1 recall
  - Fluency (n-gram entropy)
  - Truth Ratio (requires perturbed answers)
  - Probability (normalized GT probability)
  - Model Utility (harmonic mean of non-forget metrics)

Usage:
  python evaluate.py --config configs/tofu_eval.yaml
  python evaluate.py --config configs/tofu_eval.yaml --model_path=outputs/unlearn/npo/forget10/tofu/qwen2.5-1.5b
"""
import json
import os
import torch
import numpy as np
import nltk
import scipy.stats
from pathlib import Path
from tqdm import tqdm
from scipy.stats import hmean

from src.utils import parse_args, load_config, get_model_identifiers, load_model_and_tokenizer
from src.data_module import SFTDataset, sft_collator, convert_to_model_format
from src.trainers import get_batch_loss


# ========================= HELPER: N-GRAM ENTROPY =========================

def compute_freq(sentence, n=2):
    """Compute n-gram frequency distribution."""
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def compute_n_gram_entropy(sentence, ns=None, weights=None):
    """Compute weighted n-gram entropy for a single sentence."""
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        if len(freqs) == 0:
            entropy_list.append(0.0)
            continue
        freqs = freqs / freqs.sum()
        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)
    return np.mean(entropy_list)


def n_gram_entropy(gen_texts):
    """Compute mean n-gram entropy across generated texts."""
    if not gen_texts:
        return 0.0
    return np.mean([compute_n_gram_entropy(txt) for txt in gen_texts]).item()


# ========================= DATASET FOR PERTURBED ANSWERS =========================

class PerturbedDataset(torch.utils.data.Dataset):
    """Dataset that tokenizes perturbed answers for Truth Ratio computation."""
    def __init__(self, data, tokenizer, model_family, max_length=500,
                 question_key="question", perturbed_answer_key="perturbed_answer"):
        from src.utils import get_model_identifiers
        self.data = data
        self.tokenizer = tokenizer
        self.model_configs = get_model_identifiers(model_family)
        self.max_length = max_length
        self.question_key = question_key
        self.perturbed_answer_key = perturbed_answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item[self.question_key]
        perturbed_answers = item[self.perturbed_answer_key]

        # Tokenize each perturbed answer separately
        all_input_ids = []
        all_labels = []
        all_masks = []
        for pa in perturbed_answers:
            input_ids, labels, attention_mask = convert_to_model_format(
                self.tokenizer, self.max_length, question, pa, self.model_configs
            )
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_masks.append(attention_mask)

        return (
            torch.stack(all_input_ids),   # (num_perturb, seq_len)
            torch.stack(all_labels),
            torch.stack(all_masks),
            idx,
        )


def perturbed_collator(batch):
    """Collate perturbed data — each sample has multiple perturbed answers."""
    # batch: list of (input_ids, labels, mask, idx) where input_ids is (num_perturb, seq_len)
    input_ids = torch.stack([b[0] for b in batch])    # (bs, num_perturb, seq_len)
    labels = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    indices = [b[3] for b in batch]
    return input_ids, labels, masks, indices


# ========================= CORE EVAL: LOSS-BASED =========================

def compute_loss_metrics(model, dataloader, device):
    """
    Compute per-sample GT loss metrics (mirrors UnlearnPII's get_all_evals loss section).
    Returns dict with per-sample: avg_gt_loss, gt_loss, num_token_gt
    """
    model.eval()
    eval_logs = {"avg_gt_loss": {}, "gt_loss": {}, "num_token_gt": {}}

    sample_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing loss", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Per-sample loss using get_batch_loss
            per_sample_loss = get_batch_loss(outputs.logits, labels)  # (bs,)
            num_tokens = (labels != -100).sum(dim=-1)  # (bs,)
            per_token_loss = per_sample_loss  # already per-token from get_batch_loss

            for i in range(input_ids.size(0)):
                idx = sample_idx + i
                eval_logs["avg_gt_loss"][idx] = per_token_loss[i].item()
                eval_logs["gt_loss"][idx] = (per_sample_loss[i] * num_tokens[i]).item()
                eval_logs["num_token_gt"][idx] = num_tokens[i].item()

            sample_idx += input_ids.size(0)

    # Compute PPL from avg losses
    all_losses = list(eval_logs["avg_gt_loss"].values())
    avg_loss = np.mean(all_losses) if all_losses else 0
    eval_logs["perplexity"] = float(np.exp(avg_loss))

    return eval_logs


# ========================= CORE EVAL: PERTURBATION RATIO =========================

def eval_perturbation_ratio(model, gt_dataloader, perturb_dataset, device, batch_size=4):
    """
    Compute Truth Ratio = exp(gt_loss_per_token - perturb_loss_per_token.mean).
    Mirrors UnlearnPII's eval_perturbation_ratio().

    Args:
        model: the model to evaluate
        gt_dataloader: dataloader for ground truth QA pairs
        perturb_dataset: PerturbedDataset with perturbed answers
        device: cuda/cpu
        batch_size: batch size for perturbed eval
    """
    model.eval()
    eval_logs = {
        "average_perturb_loss": {},
        "avg_paraphrased_loss": {},
        "truth_ratio": {},
    }

    # First pass: get GT loss per sample (reuse from loss_metrics if available)
    gt_losses = {}
    sample_idx = 0
    with torch.no_grad():
        for batch in tqdm(gt_dataloader, desc="GT loss for Truth Ratio", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            per_sample_loss = get_batch_loss(outputs.logits, labels)

            for i in range(input_ids.size(0)):
                gt_losses[sample_idx + i] = per_sample_loss[i].item()
            sample_idx += input_ids.size(0)

    # Second pass: get perturbed loss per sample
    perturb_loader = torch.utils.data.DataLoader(
        perturb_dataset, batch_size=batch_size, shuffle=False, collate_fn=perturbed_collator,
    )

    with torch.no_grad():
        for p_input_ids, p_labels, p_masks, indices in tqdm(perturb_loader, desc="Perturbed loss", leave=False):
            bs, num_perturb, seq_len = p_input_ids.shape
            # Flatten: (bs*num_perturb, seq_len)
            flat_ids = p_input_ids.view(bs * num_perturb, seq_len).to(device)
            flat_labels = p_labels.view(bs * num_perturb, seq_len).to(device)
            flat_masks = p_masks.view(bs * num_perturb, seq_len).to(device)

            outputs = model(input_ids=flat_ids, attention_mask=flat_masks, labels=flat_labels)
            perturb_loss = get_batch_loss(outputs.logits, flat_labels)  # (bs*num_perturb,)
            perturb_loss = perturb_loss.view(bs, num_perturb)  # (bs, num_perturb)

            for i, idx in enumerate(indices):
                gt_loss_val = gt_losses[idx]
                perturb_loss_vals = perturb_loss[i].cpu().numpy().tolist()
                mean_perturb_loss = np.mean(perturb_loss_vals)

                eval_logs["average_perturb_loss"][idx] = perturb_loss_vals
                eval_logs["avg_paraphrased_loss"][idx] = gt_loss_val
                eval_logs["truth_ratio"][idx] = float(np.exp(gt_loss_val - mean_perturb_loss))

    return eval_logs


# ========================= CORE EVAL: GENERATION =========================

def compute_generation_metrics(model, tokenizer, dataset, model_configs,
                               max_new_tokens=128, device="cuda"):
    """
    Generate answers and compute:
      - ROUGE-L recall, ROUGE-1 recall
      - Fluency (n-gram entropy)
      - PII leakage (exact substring match — basic version)

    Returns: (eval_logs, gen_details)
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    except ImportError:
        print("WARNING: rouge_score not installed. pip install rouge-score")
        scorer = None

    model.eval()
    q_start = model_configs["question_start_tag"]
    q_end = model_configs["question_end_tag"]
    a_start = model_configs["answer_tag"]

    rouge1_recall = {}
    rougeL_recall = {}
    gen_texts_list = []
    gen_details = []

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

        gen_text = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        gen_texts_list.append(gen_text)

        # ROUGE
        if scorer:
            scores = scorer.score(gold_answer, gen_text)
            rouge1_recall[i] = scores["rouge1"].recall
            rougeL_recall[i] = scores["rougeL"].recall
        else:
            rouge1_recall[i] = 0.0
            rougeL_recall[i] = 0.0

        # PII leakage (basic exact match)
        pii_leaked = 0
        for pii_val in subject_pii:
            if pii_val.lower() in gen_text.lower():
                pii_leaked = 1
                break

        gen_details.append({
            "question": question,
            "gold": gold_answer,
            "generated": gen_text,
            "rouge1_recall": rouge1_recall[i],
            "rougeL_recall": rougeL_recall[i],
            "pii_leaked": pii_leaked,
        })

    # Fluency
    fluency = n_gram_entropy(gen_texts_list)

    eval_logs = {
        "rouge1_recall": rouge1_recall,
        "rougeL_recall": rougeL_recall,
        "fluency": fluency,
    }

    return eval_logs, gen_details


# ========================= AGGREGATE: MODEL UTILITY =========================

def compute_aggregate_metrics(all_task_logs, eval_task_configs):
    """
    Compute aggregate metrics following UnlearnPII's aggregate_eval_stat.py:
      - Probability per task
      - Truth Ratio per task (forget vs non-forget use different formulas)
      - ROUGE per task
      - Model Utility = hmean of non-forget metrics

    Args:
        all_task_logs: dict { task_name: eval_logs }
        eval_task_configs: list of task config dicts from YAML
    """
    output = {}
    task_name_map = {}  # task_name -> display name

    for tcfg in eval_task_configs:
        name = tcfg["name"]
        if "forget" in name:
            if "paraphrase" in name or "rephrase" in name:
                task_name_map[name] = "Forget Rephrase"
            else:
                task_name_map[name] = "Forget"
        elif "retain" in name:
            if "paraphrase" in name or "rephrase" in name:
                task_name_map[name] = "Retain Rephrase"
            else:
                task_name_map[name] = "Retain"
        elif "real_world" in name:
            task_name_map[name] = "Real World"
        elif "real_author" in name:
            task_name_map[name] = "Real Authors"
        else:
            task_name_map[name] = name

    for task_name, logs in all_task_logs.items():
        display = task_name_map.get(task_name, task_name)

        # --- Probability ---
        if "avg_gt_loss" in logs:
            if "eval_log" in task_name:
                # For forget/retain: simple mean of exp(-loss)
                gt_probs = np.exp(-1 * np.array(list(logs["avg_gt_loss"].values())))
                output[f"Prob. {display}"] = float(np.mean(gt_probs))
            elif "average_perturb_loss" in logs:
                # For real-world: normalized probability
                avg_true = np.exp(-1 * np.array(list(logs["avg_gt_loss"].values())))
                avg_false = np.exp(-1 * np.array(list(logs["average_perturb_loss"].values())))
                avg_all = np.concatenate([np.expand_dims(avg_true, axis=-1), avg_false], axis=1).sum(-1)
                output[f"Prob. {display}"] = float(np.mean(avg_true / avg_all))
            else:
                gt_probs = np.exp(-1 * np.array(list(logs["avg_gt_loss"].values())))
                output[f"Prob. {display}"] = float(np.mean(gt_probs))

        # --- ROUGE ---
        if "rougeL_recall" in logs:
            output[f"ROUGE {display}"] = float(np.mean(list(logs["rougeL_recall"].values())))

        # --- Truth Ratio ---
        if "avg_paraphrased_loss" in logs and "average_perturb_loss" in logs:
            para_vals = np.array(list(logs["avg_paraphrased_loss"].values()))
            perturb_vals = np.array(list(logs["average_perturb_loss"].values()))
            perturb_mean = perturb_vals.mean(axis=-1) if perturb_vals.ndim > 1 else perturb_vals

            curr_stat = np.exp(perturb_mean - para_vals)

            if "forget" in task_name:
                # Forget: closer to 1 = model is confused = good unlearning
                tr = float(np.mean(np.minimum(curr_stat, 1 / curr_stat)))
            else:
                # Non-forget: higher = model is confident on correct answer = good utility
                tr = float(np.mean(np.maximum(0, 1 - 1 / curr_stat)))

            output[f"Truth Ratio {display}"] = tr

        # --- Fluency ---
        if "fluency" in logs:
            output[f"Fluency {display}"] = logs["fluency"]

    # --- Model Utility: hmean of non-forget metrics ---
    utility_cands = []
    for k, v in output.items():
        if "Forget" not in k and "Fluency" not in k and isinstance(v, (int, float)):
            if v > 0:  # hmean requires positive values
                utility_cands.append(v)

    if utility_cands:
        output["Model Utility"] = float(hmean(utility_cands))

    return output


# ========================= MAIN EVAL =========================

def run_eval(cfg):
    print("=" * 60)
    print(f"EVALUATION — model: {cfg['model_path']}")
    print("=" * 60)

    model_cfg = get_model_identifiers(cfg["model_family"])
    model, tokenizer = load_model_and_tokenizer(cfg, model_cfg)
    device = model.device

    save_dir = cfg.get("save_dir") or os.path.join(cfg["model_path"], "eval_results")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_task_logs = {}

    for task_cfg in cfg.get("eval_tasks", []):
        task_name = task_cfg["name"]
        print(f"\n{'='*60}")
        print(f"[{task_name}]")
        print(f"{'='*60}")

        data_file = os.path.join(task_cfg["data_path"], f"{task_cfg['split']}.json")
        if not os.path.exists(data_file):
            print(f"  SKIP: {data_file} not found")
            continue

        # Load raw data for perturbed access
        with open(data_file) as f:
            raw_data = json.load(f)

        question_key = task_cfg.get("question_key", "question")
        answer_key = task_cfg.get("answer_key", "answer")
        perturbed_key = task_cfg.get("perturbed_answer_key", None)

        # Main dataset + dataloader
        ds = SFTDataset(
            data_file, tokenizer, cfg["model_family"],
            max_length=cfg.get("max_length", 500),
            question_key=question_key,
            answer_key=answer_key,
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=cfg.get("batch_size", 4),
            collate_fn=sft_collator, shuffle=False,
        )

        task_logs = {}

        # 1) Loss-based metrics: avg_gt_loss, gt_loss, num_token_gt, PPL
        print("  [1/4] Computing loss metrics...")
        loss_logs = compute_loss_metrics(model, dl, device)
        task_logs.update(loss_logs)
        print(f"    PPL: {loss_logs['perplexity']:.2f}")

        # 2) Generation metrics: ROUGE-1, ROUGE-L, Fluency
        print("  [2/4] Computing generation metrics (ROUGE, Fluency)...")
        gen_logs, gen_details = compute_generation_metrics(
            model, tokenizer, ds, model_cfg,
            max_new_tokens=cfg.get("max_new_tokens", 128), device=device,
        )
        task_logs.update(gen_logs)
        avg_rouge_l = float(np.mean(list(gen_logs["rougeL_recall"].values()))) if gen_logs["rougeL_recall"] else 0
        avg_rouge_1 = float(np.mean(list(gen_logs["rouge1_recall"].values()))) if gen_logs["rouge1_recall"] else 0
        print(f"    ROUGE-L: {avg_rouge_l:.4f}  ROUGE-1: {avg_rouge_1:.4f}  Fluency: {gen_logs['fluency']:.4f}")

        # 3) Truth Ratio (if perturbed data available)
        if perturbed_key and len(raw_data) > 0 and perturbed_key in raw_data[0]:
            print("  [3/4] Computing Truth Ratio (perturbed answers)...")
            perturb_ds = PerturbedDataset(
                raw_data, tokenizer, cfg["model_family"],
                max_length=cfg.get("max_length", 500),
                question_key=question_key,
                perturbed_answer_key=perturbed_key,
            )
            perturb_logs = eval_perturbation_ratio(
                model, dl, perturb_ds, device,
                batch_size=max(1, cfg.get("batch_size", 4) // 4),
            )
            task_logs.update(perturb_logs)

            tr_vals = list(perturb_logs["truth_ratio"].values())
            print(f"    Truth Ratio (raw mean): {np.mean(tr_vals):.4f}")
        else:
            print("  [3/4] Truth Ratio: SKIPPED (no perturbed_answer_key)")

        # 4) PII leakage summary
        pii_rates = [d["pii_leaked"] for d in gen_details]
        pii_rate = np.mean(pii_rates) if pii_rates else 0
        task_logs["pii_leakage_rate"] = float(pii_rate)
        print(f"  [4/4] PII Leakage Rate: {pii_rate:.4f}")

        all_task_logs[task_name] = task_logs

        # Save per-task details
        detail_path = os.path.join(save_dir, f"{task_name}_details.json")
        with open(detail_path, "w") as f:
            json.dump(gen_details, f, indent=2, ensure_ascii=False)

        # Save per-task eval logs
        task_log_path = os.path.join(save_dir, f"{task_name}.json")
        with open(task_log_path, "w") as f:
            json.dump(task_logs, f, indent=2)

    # =================== AGGREGATE ===================
    print(f"\n{'='*60}")
    print("AGGREGATE METRICS")
    print(f"{'='*60}")

    agg = compute_aggregate_metrics(all_task_logs, cfg.get("eval_tasks", []))

    # Print aggregate table
    print(f"\n{'Metric':<40} {'Value':>10}")
    print("-" * 52)
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"{k:<40} {v:>10.4f}")
        else:
            print(f"{k:<40} {str(v):>10}")

    # Save aggregated
    agg_path = os.path.join(save_dir, "eval_log_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump({"per_task": {k: _make_serializable(v) for k, v in all_task_logs.items()},
                    "aggregate": agg}, f, indent=2)
    print(f"\nAll results saved to {save_dir}")

    return all_task_logs, agg


def _make_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    config_path, overrides = parse_args()
    cfg = load_config(config_path, overrides)
    run_eval(cfg)


if __name__ == "__main__":
    main()
