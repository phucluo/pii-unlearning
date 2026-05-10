"""Adaptive Adversarial Unlearning for PII (AAU-PII).

Outer loop: mine hard prompts that still leak PII, then retrain on them.
"""
import copy
import json
import os

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_module import (
    ForgetRetainDataset,
    SFTDataset,
    forget_collator,
    sft_collator,
)
from src.trainers import compute_ce_loss, grad_ascent_loss, npo_loss
from src.utils import get_model_identifiers, load_model_and_tokenizer, save_model

# Reuse the same word-boundary PII matcher used at evaluation time so that
# training-time judging stays consistent with the reported metrics.
from evaluate import pii_exact_match


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


class LeakageJudge:
    """Detect PII leakage in generated text on a per-field basis."""

    def __init__(self, profiles_path, forget_names):
        self.person_pii = {}
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

    def score(self, generated_text, person_name):
        pii_dict = self.person_pii.get(person_name, {})
        if not pii_dict or not generated_text:
            return 0.0

        leaked = 0
        for field, val in pii_dict.items():
            if pii_exact_match(field, val, generated_text):
                leaked += 1
        return leaked / len(pii_dict)

    def check_any_leak(self, generated_text, person_name):
        return self.score(generated_text, person_name) > 0.0


class PromptMiner:
    """Generate candidate prompts and find the ones that still leak."""

    def __init__(self, forget_data, forget_names, judge):
        self.forget_data = forget_data
        self.forget_names = forget_names
        self.judge = judge

    def _find_person(self, item):
        question = item.get("question", "").lower()
        for name in self.forget_names:
            if name.lower() in question:
                return name
        # If the name isn't in the question, look for a matching PII value.
        subject_pii = item.get("subject_pii", [])
        for name, pii_dict in self.judge.person_pii.items():
            for val in pii_dict.values():
                if val in subject_pii:
                    return name
        return None

    def _detect_field(self, item, person_name):
        answer = item.get("answer", "").strip().lower()
        pii_dict = self.judge.person_pii.get(person_name, {})
        for field, val in pii_dict.items():
            if val.lower().strip() == answer:
                return field
        return None

    def collect_candidates(self):
        candidates = []

        for item in self.forget_data:
            person = self._find_person(item)
            if not person:
                continue
            answer = item["answer"]

            candidates.append({
                "question": item["question"],
                "answer": answer,
                "person": person,
                "style": "direct",
            })

            for i in range(1, 6):
                key = f"paraphrased_question_{i}"
                if key in item and item[key]:
                    candidates.append({
                        "question": item[key],
                        "answer": answer,
                        "person": person,
                        "style": f"paraphrase_{i}",
                    })

            if item.get("inverted_question"):
                candidates.append({
                    "question": item["inverted_question"],
                    "answer": item.get("inverted_answer", answer),
                    "person": person,
                    "style": "inverted",
                })

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
        candidates = self.collect_candidates()
        print(f"  [Mine] {len(candidates)} candidate prompts collected")

        q_start = model_configs["question_start_tag"]
        q_end = model_configs["question_end_tag"]
        a_start = model_configs["answer_tag"]
        formatted = [q_start + c["question"] + q_end + a_start for c in candidates]

        n_samples = aau_cfg.get("self_probe_samples", 3)
        temperature = aau_cfg.get("self_probe_temperature", 0.7)
        max_new_tokens = aau_cfg.get("max_new_tokens", 128)
        gen_bs = aau_cfg.get("gen_batch_size", 16)

        all_responses = [[] for _ in candidates]

        greedy_texts = _batch_generate(
            model, tokenizer, formatted, device,
            max_new_tokens=max_new_tokens, batch_size=gen_bs,
            do_sample=False, desc="Greedy probe",
        )
        for i, text in enumerate(greedy_texts):
            all_responses[i].append(text)

        for s in range(n_samples):
            sampled_texts = _batch_generate(
                model, tokenizer, formatted, device,
                max_new_tokens=max_new_tokens, batch_size=gen_bs,
                do_sample=True, temperature=temperature,
                desc=f"Sample probe {s+1}/{n_samples}",
            )
            for i, text in enumerate(sampled_texts):
                all_responses[i].append(text)

        hard_prompts = []
        for cand, responses in zip(candidates, all_responses):
            leak_score = max(
                self.judge.score(resp, cand["person"]) for resp in responses
            )
            if leak_score > 0:
                cand["leak_score"] = leak_score
                hard_prompts.append(cand)

        # Compute leak_rate over the full candidate pool before truncating
        # to top-k, so the rate isn't capped by the top-k size.
        total = len(candidates)
        n_leak_total = len(hard_prompts)
        leak_rate = n_leak_total / total if total > 0 else 0

        hard_prompts.sort(key=lambda x: x["leak_score"], reverse=True)
        top_k = aau_cfg.get("top_k_hard_prompts", 50)
        hard_prompts = hard_prompts[:top_k]

        print(f"  [Mine] {n_leak_total}/{total} prompts leak PII "
              f"(rate={leak_rate:.3f}), selected top-{len(hard_prompts)} "
              f"for training")

        return hard_prompts, leak_rate


def _batch_generate(model, tokenizer, prompts, device,
                    max_new_tokens=128, batch_size=16,
                    do_sample=False, temperature=1.0, desc="Generating"):
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


def _compute_retain_loss(model, retain_dataloader, device):
    model.eval()
    total_loss = 0.0
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
    round_dir = os.path.join(save_dir, "aau_data", f"round_{round_num}")
    Path(round_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(round_dir, "hard_prompts.json"), "w") as f:
        json.dump(hard_prompts, f, indent=2, ensure_ascii=False)

    train_data = [
        {"question": hp["question"], "answer": hp["answer"]}
        for hp in hard_prompts
    ]
    with open(os.path.join(round_dir, "forget10.json"), "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    return round_dir


def _save_checkpoint(model, tokenizer, optimizer, round_num, step, save_dir):
    ckpt_dir = os.path.join(save_dir, f"round_{round_num}_step{step}")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    print(f"  [Checkpoint] Saved → {ckpt_dir}")
    return ckpt_dir


def run_aau_pii(cfg):
    print("=" * 60)
    print("AAU-PII: Adaptive Adversarial Unlearning")
    print("=" * 60)

    aau_cfg = cfg.get("aau", {})
    save_dir = cfg["save_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = get_model_identifiers(cfg["model_family"])

    warm_start = cfg.get("warm_start_path") or aau_cfg.get("warm_start_path")
    if warm_start:
        cfg_load = copy.deepcopy(cfg)
        cfg_load["model_path"] = warm_start
        print(f"[AAU] Loading model from warm-start: {warm_start}")
    else:
        cfg_load = cfg
        print(f"[AAU] Loading model from: {cfg.get('model_path', 'HF default')}")

    model, tokenizer = load_model_and_tokenizer(cfg_load, model_cfg)

    inner_method = aau_cfg.get("inner_method", "grad_ascent")
    oracle_model = None
    if inner_method == "npo":
        print("[AAU] Loading oracle (SFT reference) model for NPO...")
        oracle_cfg = copy.deepcopy(cfg)
        oracle_cfg["lora"] = {"r": 0}
        oracle_model, _ = load_model_and_tokenizer(oracle_cfg, model_cfg, is_eval=True)
        oracle_model.eval()
        for p in oracle_model.parameters():
            p.requires_grad = False

    split = cfg["split"]
    pct = int(split.replace("forget", ""))
    retain_split = f"retain{100 - pct}"

    forget_path = os.path.join(cfg["forget_data_path"], f"{split}.json")
    retain_path = os.path.join(cfg["retain_data_path"], f"{retain_split}.json")
    idk_path = cfg.get("idk_path", "data/raw/idontknow.jsonl")

    with open(forget_path) as f:
        forget_data = json.load(f)
    print(f"[AAU] Forget set: {len(forget_data)} items from {forget_path}")

    names_path = cfg.get("names_path", "data/raw/split_person_names")
    names_file = os.path.join(names_path, f"{split}_names.json")
    with open(names_file) as f:
        forget_names = set(json.load(f))
    print(f"[AAU] Forget persons: {len(forget_names)}")

    profiles_path = cfg.get("profiles_path", "data/raw/full_user_profiles.json")
    judge = LeakageJudge(profiles_path, forget_names)
    miner = PromptMiner(forget_data, forget_names, judge)

    retain_eval_ds = SFTDataset(
        retain_path, tokenizer, cfg["model_family"],
        max_length=cfg.get("max_length", 500),
    )
    retain_eval_dl = DataLoader(
        retain_eval_ds, batch_size=cfg["batch_size"],
        collate_fn=sft_collator, num_workers=0,
    )

    initial_retain_loss = _compute_retain_loss(model, retain_eval_dl, device)
    print(f"[AAU] Initial retain loss: {initial_retain_loss:.4f}")

    inner_loss_fn = npo_loss if inner_method == "npo" else grad_ascent_loss
    print(f"[AAU] Inner method: {inner_method}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01),
    )

    max_rounds = aau_cfg.get("rounds", 5)
    inner_max_steps = aau_cfg.get("inner_max_steps", 100)
    retain_weight = aau_cfg.get("retain_weight", 1.0)
    beta = aau_cfg.get("beta", 0.1)
    leak_threshold = aau_cfg.get("leak_threshold", 0.05)
    utility_degradation = aau_cfg.get("utility_degradation", 1.5)
    grad_accum = cfg.get("gradient_accumulation_steps", 1)

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

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"AAU Round {round_num}/{max_rounds}")
        print(f"{'='*60}")

        hard_prompts, overall_leak_rate = miner.mine(
            model, tokenizer, model_cfg, device, aau_cfg,
        )

        if len(hard_prompts) == 0:
            print(f"  [STOP] No hard prompts found — converged!")
            audit_log["rounds"].append({
                "round": round_num,
                "num_hard_prompts": 0,
                "leak_rate": overall_leak_rate,
                "stopped": "no_hard_prompts",
            })
            break

        round_dir = _save_round_data(save_dir, round_num, hard_prompts, None)
        round_forget_path = os.path.join(round_dir, "forget10.json")

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

        model.train()
        step = 0
        total_loss = 0.0
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

        if step % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        pbar.close()
        avg_loss = total_loss / max(step, 1)
        print(f"  [Train] avg loss: {avg_loss:.4f}, steps: {step}")

        retain_loss = _compute_retain_loss(model, retain_eval_dl, device)
        degradation = retain_loss / initial_retain_loss if initial_retain_loss > 0 else 1.0
        print(f"  [Eval] retain loss: {retain_loss:.4f} "
              f"(degradation: {degradation:.2f}x vs initial {initial_retain_loss:.4f})")

        _save_checkpoint(model, tokenizer, optimizer, round_num, global_step, save_dir)

        round_entry = {
            "round": round_num,
            "num_hard_prompts": len(hard_prompts),
            "leak_rate": overall_leak_rate,
            "avg_train_loss": avg_loss,
            "retain_loss": retain_loss,
            "retain_degradation": degradation,
            "global_step": global_step,
        }

        # Stop if utility has degraded too much.
        if degradation > utility_degradation:
            print(f"  [STOP] Retain degradation {degradation:.2f}x > "
                  f"threshold {utility_degradation}x — stopping to preserve utility")
            round_entry["stopped"] = "utility_floor"
            audit_log["rounds"].append(round_entry)
            break

        # Stop if the leak rate is already low enough.
        if overall_leak_rate < leak_threshold:
            print(f"  [INFO] Leak rate {overall_leak_rate:.4f} < threshold {leak_threshold} "
                  f"— trained this round, stopping here.")
            round_entry["stopped"] = "leak_threshold_post_train"
            audit_log["rounds"].append(round_entry)
            break

    print(f"\n{'='*60}")
    print(f"AAU-PII complete — {len(audit_log['rounds'])} rounds, "
          f"global_step={global_step}")
    print(f"{'='*60}")

    save_model(model, tokenizer, save_dir)

    audit_path = os.path.join(save_dir, "aau_data", "audit_log.json")
    Path(os.path.dirname(audit_path)).mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)
    print(f"[AAU] Audit log saved to {audit_path}")

    config_save = {
        k: (str(v) if not isinstance(v, (int, float, bool, type(None), list, dict)) else v)
        for k, v in cfg.items()
    }
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(config_save, f, indent=2)

    print(f"[AAU] Final model saved to {save_dir}")
