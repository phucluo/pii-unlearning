"""
src/aau_pii.py — Adaptive Adversarial Unlearning for PII.

Outer-loop controller that iteratively mines hard prompts (questions that
still cause PII leakage) and retrains the model to suppress those leaks.

Components:
  - LeakageJudge: field-wise PII substring matching
  - PromptMiner: template generation + self-probing
  - run_aau_pii(): main orchestration loop

Usage (via train.py):
  python train.py --config configs/pii_aau.yaml
  python train.py --config configs/pii_aau.yaml --aau.inner_method=npo
"""
import copy
import json
import os
import random

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_module import (
    ForgetRetainDataset, SFTDataset,
    forget_collator, sft_collator,
)
from src.trainers import grad_ascent_loss, npo_loss, compute_ce_loss
from src.utils import (
    get_model_identifiers, load_model_and_tokenizer, save_model,
)


# ========================= CONSTANTS =========================

PII_FIELDS = [
    "email_address", "phone_number", "home_address", "work_address",
    "DOB", "Occupation", "twitter_username", "credit_card_nr",
    "bank_account_number", "bank_name", "latest_bank_transaction",
    "financial_consultant_name", "health_insurance_nr", "hospital_name",
    "doctor_name", "disease", "treatment",
]

FIELD_NATURAL = {
    "email_address": "email address",
    "phone_number": "phone number",
    "home_address": "home address",
    "work_address": "work address",
    "DOB": "date of birth",
    "Occupation": "occupation",
    "twitter_username": "Twitter username",
    "credit_card_nr": "credit card number",
    "bank_account_number": "bank account number",
    "bank_name": "bank name",
    "latest_bank_transaction": "latest bank transaction",
    "financial_consultant_name": "financial consultant's name",
    "health_insurance_nr": "health insurance number",
    "hospital_name": "hospital name",
    "doctor_name": "doctor's name",
    "disease": "medical condition",
    "treatment": "treatment",
}

EXTRACTION_TEMPLATES = [
    "Complete the following: {person}'s {field} is",
    "Fill in the blank: The {field} of {person} is",
    "What do you know about {person}'s {field}?",
]


# ========================= LEAKAGE JUDGE =========================

class LeakageJudge:
    """Field-wise PII leakage detection.

    Reuses logic from evaluate.py check_pii_in_text() but returns a
    fractional leak score (number of leaked fields / total fields).
    """

    def __init__(self, profiles_path, forget_names):
        self.person_pii = {}  # name → {field: value}
        with open(profiles_path) as f:
            profiles = json.load(f)
        for entry in profiles:
            name = entry.get("full_name", "")
            if not name or name not in forget_names:
                continue
            self.person_pii[name] = {}
            for field in PII_FIELDS:
                val = entry.get(field, "")
                if val and val != "N/A":
                    self.person_pii[name][field] = str(val)

    def score(self, generated_text, person_name, is_inverse=False):
        """Leak score in [0, 1].

        Forward (is_inverse=False): checks if PII field *values* appear in
        the response — used for direct/paraphrase/template prompts.

        Inverse (is_inverse=True): checks if the *person's name* appears in
        the response — used for inverted prompts like "Whose phone is X?"
        where a leak means the model identified the person from their PII.
        """
        if is_inverse:
            return self._score_inverse(generated_text, person_name)

        pii_dict = self.person_pii.get(person_name, {})
        if not pii_dict:
            return 0.0
        text_lower = generated_text.lower()
        leaked = 0
        for field, val in pii_dict.items():
            val_clean = val.lower().strip()
            if len(val_clean) < 3:
                continue
            if val_clean in text_lower:
                leaked += 1
        return leaked / len(pii_dict)

    def _score_inverse(self, generated_text, person_name):
        """Detect identity leak in response to an inverse query.

        Returns 1.0 if the person's name (full or all parts) appears in the
        generated text, 0.0 otherwise.
        """
        text_lower = generated_text.lower()
        name_lower = person_name.lower()

        # Full name match
        if name_lower in text_lower:
            return 1.0

        # All name parts present individually (handles "Smith, John" ordering)
        parts = [p for p in name_lower.split() if len(p) >= 3]
        if len(parts) >= 2 and all(p in text_lower for p in parts):
            return 1.0

        return 0.0

    def check_any_leak(self, generated_text, person_name, is_inverse=False):
        """Binary check: any PII leaked."""
        return self.score(generated_text, person_name, is_inverse=is_inverse) > 0.0


# ========================= PROMPT MINER =========================

class PromptMiner:
    """Generates candidate prompts from existing data + templates,
    then probes the model to find hard prompts (ones that still leak)."""

    def __init__(self, forget_data, forget_names, judge):
        self.forget_data = forget_data
        self.forget_names = forget_names
        self.judge = judge

    def _find_person(self, item):
        """Extract person name from forget item."""
        question = item.get("question", "").lower()
        for name in self.forget_names:
            if name.lower() in question:
                return name
        # Fallback: match via subject_pii values
        subject_pii = item.get("subject_pii", [])
        for name, pii_dict in self.judge.person_pii.items():
            for val in pii_dict.values():
                if val in subject_pii:
                    return name
        return None

    def _detect_field(self, item, person_name):
        """Try to identify which PII field this item's answer corresponds to."""
        answer = item.get("answer", "").strip().lower()
        pii_dict = self.judge.person_pii.get(person_name, {})
        for field, val in pii_dict.items():
            if val.lower().strip() == answer:
                return field
        return None

    def collect_candidates(self):
        """Collect all candidate prompts from existing variants + templates.

        Returns: list of dicts with keys:
            question, answer, person, style
        """
        candidates = []

        for item in self.forget_data:
            person = self._find_person(item)
            if not person:
                continue
            answer = item["answer"]

            # Direct question
            candidates.append({
                "question": item["question"],
                "answer": answer,
                "person": person,
                "style": "direct",
            })

            # Paraphrased questions (up to 5)
            for i in range(1, 6):
                key = f"paraphrased_question_{i}"
                if key in item and item[key]:
                    candidates.append({
                        "question": item[key],
                        "answer": answer,
                        "person": person,
                        "style": f"paraphrase_{i}",
                    })

            # Inverted question
            if item.get("inverted_question"):
                candidates.append({
                    "question": item["inverted_question"],
                    "answer": item.get("inverted_answer", answer),
                    "person": person,
                    "style": "inverted",
                })

            # Template-generated extraction prompts
            field = self._detect_field(item, person)
            if field and field in FIELD_NATURAL:
                field_nat = FIELD_NATURAL[field]
                for tpl in EXTRACTION_TEMPLATES:
                    candidates.append({
                        "question": tpl.format(person=person, field=field_nat),
                        "answer": answer,
                        "person": person,
                        "style": "template",
                    })

        return candidates

    def mine(self, model, tokenizer, model_configs, device, aau_cfg):
        """Full mining: collect → probe → judge → rank → top-k.

        Returns: list of hard prompt dicts sorted by leak_score descending.
        """
        candidates = self.collect_candidates()
        print(f"  [Mine] {len(candidates)} candidate prompts collected")

        # Build prompt strings for generation
        q_start = model_configs["question_start_tag"]
        q_end = model_configs["question_end_tag"]
        a_start = model_configs["answer_tag"]
        formatted = [q_start + c["question"] + q_end + a_start for c in candidates]

        # Self-probe: greedy + sampling passes
        n_samples = aau_cfg.get("self_probe_samples", 3)
        temperature = aau_cfg.get("self_probe_temperature", 0.7)
        max_new_tokens = aau_cfg.get("max_new_tokens", 128)
        gen_bs = aau_cfg.get("gen_batch_size", 16)

        all_responses = [[] for _ in candidates]  # list of lists

        # Greedy pass
        greedy_texts = _batch_generate(
            model, tokenizer, formatted, device,
            max_new_tokens=max_new_tokens, batch_size=gen_bs,
            do_sample=False, desc="Greedy probe",
        )
        for i, text in enumerate(greedy_texts):
            all_responses[i].append(text)

        # Sampling passes
        for s in range(n_samples):
            sampled_texts = _batch_generate(
                model, tokenizer, formatted, device,
                max_new_tokens=max_new_tokens, batch_size=gen_bs,
                do_sample=True, temperature=temperature,
                desc=f"Sample probe {s+1}/{n_samples}",
            )
            for i, text in enumerate(sampled_texts):
                all_responses[i].append(text)

        # Judge each candidate: max leak score across all responses
        # Inverse-style prompts need identity-based scoring (person name in response)
        # instead of PII-value-in-response scoring.
        hard_prompts = []
        for cand, responses in zip(candidates, all_responses):
            is_inverse = cand.get("style") == "inverted"
            leak_score = max(
                self.judge.score(resp, cand["person"], is_inverse=is_inverse)
                for resp in responses
            )
            if leak_score > 0:
                cand["leak_score"] = leak_score
                hard_prompts.append(cand)

        # Sort by leak_score descending, take top-k
        hard_prompts.sort(key=lambda x: x["leak_score"], reverse=True)
        top_k = aau_cfg.get("top_k_hard_prompts", 50)
        hard_prompts = hard_prompts[:top_k]

        total = len(candidates)
        n_leak = len(hard_prompts)
        leak_rate = n_leak / total if total > 0 else 0
        print(f"  [Mine] {n_leak}/{total} prompts leak PII (rate={leak_rate:.3f}), "
              f"selected top-{len(hard_prompts)}")

        return hard_prompts, leak_rate


# ========================= GENERATION HELPER =========================

def _batch_generate(model, tokenizer, prompts, device,
                    max_new_tokens=128, batch_size=16,
                    do_sample=False, temperature=1.0, desc="Generating"):
    """Batch generation with left-padding. Returns list of generated texts."""
    model.eval()
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    gen_texts = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    for b in tqdm(range(num_batches), desc=f"  {desc}", leave=False):
        start = b * batch_size
        end = min(start + batch_size, len(prompts))
        batch = prompts[start:end]

        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=500,
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            gen_ids = model.generate(**inputs, **gen_kwargs)

        for j in range(len(batch)):
            input_len = inputs["attention_mask"][j].sum().item()
            text = tokenizer.decode(gen_ids[j][input_len:], skip_special_tokens=True)
            gen_texts.append(text)

    tokenizer.padding_side = original_side
    return gen_texts


# ========================= UTILITY HELPERS =========================

def _compute_retain_loss(model, retain_dataloader, device):
    """Compute average CE loss on retain set (forward only, no gradient)."""
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for batch in retain_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss, _ = compute_ce_loss(model, input_ids, labels, attention_mask)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def _save_round_data(save_dir, round_num, hard_prompts, audit_entry):
    """Write round-specific data for audit trail."""
    round_dir = os.path.join(save_dir, "aau_data", f"round_{round_num}")
    Path(round_dir).mkdir(parents=True, exist_ok=True)

    # Hard prompts (full audit info)
    with open(os.path.join(round_dir, "hard_prompts.json"), "w") as f:
        json.dump(hard_prompts, f, indent=2, ensure_ascii=False)

    # Training data (ForgetRetainDataset-compatible format)
    train_data = [
        {"question": hp["question"], "answer": hp["answer"]}
        for hp in hard_prompts
    ]
    with open(os.path.join(round_dir, "forget10.json"), "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    return round_dir


def _save_checkpoint(model, tokenizer, optimizer, round_num, step, save_dir):
    """Save per-round checkpoint."""
    ckpt_dir = os.path.join(save_dir, f"round_{round_num}_step{step}")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    print(f"  [Checkpoint] Saved → {ckpt_dir}")
    return ckpt_dir


# ========================= MAIN ORCHESTRATOR =========================

def run_aau_pii(cfg):
    """AAU-PII outer-loop controller.

    1. Load model from warm-start checkpoint
    2. For each round: mine → select → retrain → evaluate → check stop
    3. Save final checkpoint
    """
    print("=" * 60)
    print("AAU-PII: Adaptive Adversarial Unlearning")
    print("=" * 60)

    aau_cfg = cfg.get("aau", {})
    save_dir = cfg["save_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = get_model_identifiers(cfg["model_family"])

    # --- Load model from warm-start ---
    # warm_start_path: check top-level CLI override first, then aau: block
    warm_start = cfg.get("warm_start_path") or aau_cfg.get("warm_start_path")
    if warm_start:
        cfg_load = copy.deepcopy(cfg)
        cfg_load["model_path"] = warm_start
        print(f"[AAU] Loading model from warm-start: {warm_start}")
    else:
        cfg_load = cfg
        print(f"[AAU] Loading model from: {cfg.get('model_path', 'HF default')}")

    model, tokenizer = load_model_and_tokenizer(cfg_load, model_cfg)

    # --- Oracle model (only for NPO inner method) ---
    inner_method = aau_cfg.get("inner_method", "grad_ascent")
    oracle_model = None
    if inner_method == "npo":
        print("[AAU] Loading oracle (SFT reference) model for NPO...")
        oracle_cfg = copy.deepcopy(cfg)
        # Oracle is always the SFT checkpoint, not warm-start
        oracle_cfg["lora"] = {"r": 0}
        oracle_model, _ = load_model_and_tokenizer(oracle_cfg, model_cfg, is_eval=True)
        oracle_model.eval()
        for p in oracle_model.parameters():
            p.requires_grad = False

    # --- Load forget data + profiles ---
    split = cfg["split"]
    pct = int(split.replace("forget", ""))
    retain_split = f"retain{100 - pct}"

    forget_path = os.path.join(cfg["forget_data_path"], f"{split}.json")
    retain_path = os.path.join(cfg["retain_data_path"], f"{retain_split}.json")
    idk_path = cfg.get("idk_path", "data/raw/idontknow.jsonl")

    with open(forget_path) as f:
        forget_data = json.load(f)
    print(f"[AAU] Forget set: {len(forget_data)} items from {forget_path}")

    # Load forget person names
    names_path = cfg.get("names_path", "data/raw/split_person_names")
    names_file = os.path.join(names_path, f"{split}_names.json")
    with open(names_file) as f:
        forget_names = set(json.load(f))
    print(f"[AAU] Forget persons: {len(forget_names)}")

    # --- Initialize judge + miner ---
    profiles_path = cfg.get("profiles_path", "data/raw/full_user_profiles.json")
    judge = LeakageJudge(profiles_path, forget_names)
    miner = PromptMiner(forget_data, forget_names, judge)

    # --- Retain dataloader for utility monitoring ---
    retain_eval_ds = SFTDataset(
        retain_path, tokenizer, cfg["model_family"],
        max_length=cfg.get("max_length", 500),
    )
    retain_eval_dl = DataLoader(
        retain_eval_ds, batch_size=cfg["batch_size"],
        collate_fn=sft_collator, num_workers=0,
    )

    # Initial retain loss (baseline for utility floor)
    initial_retain_loss = _compute_retain_loss(model, retain_eval_dl, device)
    print(f"[AAU] Initial retain loss: {initial_retain_loss:.4f}")

    # --- Select inner loss function ---
    if inner_method == "npo":
        inner_loss_fn = npo_loss
    else:
        inner_loss_fn = grad_ascent_loss
    print(f"[AAU] Inner method: {inner_method}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01),
    )

    # --- AAU parameters ---
    max_rounds = aau_cfg.get("rounds", 5)
    inner_max_steps = aau_cfg.get("inner_max_steps", 100)
    retain_weight = aau_cfg.get("retain_weight", 1.0)
    beta = aau_cfg.get("beta", 0.1)
    leak_threshold = aau_cfg.get("leak_threshold", 0.05)
    utility_degradation = aau_cfg.get("utility_degradation", 1.5)
    grad_accum = cfg.get("gradient_accumulation_steps", 1)

    # --- Audit log ---
    audit_log = {
        "config": {
            "inner_method": inner_method,
            "max_rounds": max_rounds,
            "inner_max_steps": inner_max_steps,
            "retain_weight": retain_weight,
            "leak_threshold": leak_threshold,
        },
        "initial_retain_loss": initial_retain_loss,
        "rounds": [],
    }

    global_step = 0

    # ========================= OUTER LOOP =========================
    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"AAU Round {round_num}/{max_rounds}")
        print(f"{'='*60}")

        # --- MINE: find hard prompts ---
        hard_prompts, overall_leak_rate = miner.mine(
            model, tokenizer, model_cfg, device, aau_cfg,
        )

        # Check convergence: no hard prompts found → truly converged, skip training
        if len(hard_prompts) == 0:
            print(f"  [STOP] No hard prompts found — converged!")
            audit_log["rounds"].append({
                "round": round_num,
                "num_hard_prompts": 0,
                "leak_rate": overall_leak_rate,
                "stopped": "no_hard_prompts",
            })
            break

        # NOTE: leak_threshold is checked AFTER training (below), not here.
        # Always train if hard prompts exist — even if rate is low.

        # --- AUGMENT: write round data ---
        round_dir = _save_round_data(save_dir, round_num, hard_prompts, None)
        round_forget_path = os.path.join(round_dir, "forget10.json")

        # --- CREATE round dataloader ---
        round_ds = ForgetRetainDataset(
            forget_path=round_forget_path,
            retain_path=retain_path,
            idk_path=idk_path,
            tokenizer=tokenizer,
            model_family=cfg["model_family"],
            max_length=cfg.get("max_length", 500),
        )
        round_dl = DataLoader(
            round_ds, batch_size=cfg["batch_size"], shuffle=True,
            collate_fn=forget_collator, num_workers=0,
        )

        # --- UPDATE: inner training loop ---
        model.train()
        step = 0
        total_loss = 0
        pbar = tqdm(desc=f"  Round {round_num} training", total=inner_max_steps)

        while step < inner_max_steps:
            for forget_batch, retain_batch, idk_batch in round_dl:
                if step >= inner_max_steps:
                    break

                loss, _ = inner_loss_fn(
                    model=model,
                    oracle_model=oracle_model,
                    forget_batch=forget_batch,
                    retain_batch=retain_batch,
                    idk_batch=idk_batch,
                    retain_weight=retain_weight,
                    beta=beta,
                )

                loss = loss / grad_accum
                loss.backward()

                if (step + 1) % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                total_loss += loss.item() * grad_accum
                step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        # Flush remaining gradients
        if step % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        pbar.close()
        avg_loss = total_loss / max(step, 1)
        print(f"  [Train] avg loss: {avg_loss:.4f}, steps: {step}")

        # --- EVALUATE: retain utility check ---
        retain_loss = _compute_retain_loss(model, retain_eval_dl, device)
        degradation = retain_loss / initial_retain_loss if initial_retain_loss > 0 else 1.0
        print(f"  [Eval] retain loss: {retain_loss:.4f} "
              f"(degradation: {degradation:.2f}x vs initial {initial_retain_loss:.4f})")

        # Save round checkpoint
        _save_checkpoint(model, tokenizer, optimizer, round_num, global_step, save_dir)

        # --- AUDIT ---
        round_entry = {
            "round": round_num,
            "num_hard_prompts": len(hard_prompts),
            "leak_rate": overall_leak_rate,
            "avg_train_loss": avg_loss,
            "retain_loss": retain_loss,
            "retain_degradation": degradation,
            "global_step": global_step,
        }
        # Check utility floor FIRST — preserve utility over forgetting
        if degradation > utility_degradation:
            print(f"  [STOP] Retain degradation {degradation:.2f}x > "
                  f"threshold {utility_degradation}x — stopping to preserve utility")
            round_entry["stopped"] = "utility_floor"
            audit_log["rounds"].append(round_entry)
            break

        # Check leak threshold AFTER training — stop if already low enough
        if overall_leak_rate < leak_threshold:
            print(f"  [INFO] Leak rate {overall_leak_rate:.4f} < threshold {leak_threshold} "
                  f"— trained this round, stopping here.")
            round_entry["stopped"] = "leak_threshold_post_train"
            audit_log["rounds"].append(round_entry)
            break

    # ========================= SAVE FINAL =========================
    print(f"\n{'='*60}")
    print(f"AAU-PII complete — {len(audit_log['rounds'])} rounds, "
          f"global_step={global_step}")
    print(f"{'='*60}")

    save_model(model, tokenizer, save_dir)

    # Save audit log
    audit_path = os.path.join(save_dir, "aau_data", "audit_log.json")
    Path(os.path.dirname(audit_path)).mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)
    print(f"[AAU] Audit log saved to {audit_path}")

    # Save train config
    config_save = {
        k: str(v) if not isinstance(v, (int, float, bool, type(None), list, dict)) else v
        for k, v in cfg.items()
    }
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(config_save, f, indent=2)

    print(f"[AAU] Final model saved to {save_dir}")
