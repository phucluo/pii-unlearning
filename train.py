"""
train.py — Entry point for SFT Exposed and Unlearning.
Mirrors UnlearnPII's finetune.py (mode=sft) and forget.py (mode=unlearn).

Usage:
  python train.py --config configs/sft.yaml                          # SFT
  python train.py --config configs/unlearn.yaml                      # Unlearn (default GA)
  python train.py --config configs/unlearn.yaml --forget_loss=npo    # Override method
"""
import os
import json
import copy
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import set_seed

from src.utils import (
    parse_args, load_config, get_model_identifiers, resolve_save_dir,
    load_model_and_tokenizer, save_model, find_all_linear_names,
)
from src.data_module import get_sft_dataloader, get_forget_retain_dataloaders
from src.trainers import LOSS_REGISTRY, NEEDS_ORACLE, compute_ce_loss


def run_sft(cfg):
    """Bước 1: SFT Exposed — fine-tune model to memorize PII."""
    print("=" * 60)
    print("STEP 1: SFT Exposed")
    print("=" * 60)

    model_cfg = get_model_identifiers(cfg["model_family"])
    model, tokenizer = load_model_and_tokenizer(cfg, model_cfg)
    dataloader = get_sft_dataloader(cfg, tokenizer)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01),
    )
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    num_epochs = cfg.get("num_epochs", 5)

    model.train()
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            loss, _ = compute_ce_loss(model, input_ids, labels, attention_mask)
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * grad_accum
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

    save_model(model, tokenizer, cfg["save_dir"])
    print(f"SFT Exposed model saved to {cfg['save_dir']}")


def run_unlearn(cfg):
    """Bước 2: Unlearning — apply forget method on SFT Exposed model."""
    print("=" * 60)
    print(f"STEP 2: Unlearning — method={cfg['forget_loss']}")
    print("=" * 60)

    model_cfg = get_model_identifiers(cfg["model_family"])
    model, tokenizer = load_model_and_tokenizer(cfg, model_cfg)
    dataloader = get_forget_retain_dataloaders(cfg, tokenizer)

    # Oracle model (for NPO/DPO — frozen copy of SFT Exposed)
    oracle_model = None
    if cfg["forget_loss"] in NEEDS_ORACLE:
        print("Loading oracle (reference) model...")
        oracle_cfg = copy.deepcopy(cfg)
        oracle_cfg["lora"] = {"r": 0}  # no LoRA for oracle
        oracle_model, _ = load_model_and_tokenizer(oracle_cfg, model_cfg)
        oracle_model.eval()
        for p in oracle_model.parameters():
            p.requires_grad = False

    # Get loss function
    loss_fn = LOSS_REGISTRY[cfg["forget_loss"]]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01),
    )
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    num_epochs = cfg.get("num_epochs", 8)
    max_steps = cfg.get("max_steps")

    model.train()
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Unlearn Epoch {epoch+1}/{num_epochs}")
        for step, (forget_batch, retain_batch, idk_batch) in enumerate(pbar):

            loss, _ = loss_fn(
                model=model,
                oracle_model=oracle_model,
                forget_batch=forget_batch,
                retain_batch=retain_batch,
                idk_batch=idk_batch,
                retain_weight=cfg.get("retain_weight", 1.0),
                beta=cfg.get("beta", 0.1),
            )

            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * grad_accum
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}", step=global_step)

            if max_steps and global_step >= max_steps:
                break

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

        if max_steps and global_step >= max_steps:
            print(f"Reached max_steps={max_steps}, stopping.")
            break

    save_model(model, tokenizer, cfg["save_dir"])

    # Save training config for reproducibility
    config_save = {k: str(v) if not isinstance(v, (int, float, bool, type(None), list, dict)) else v
                   for k, v in cfg.items()}
    with open(os.path.join(cfg["save_dir"], "train_config.json"), "w") as f:
        json.dump(config_save, f, indent=2)

    print(f"Unlearned model saved to {cfg['save_dir']}")


# ========================= MAIN =========================

def main():
    config_path, overrides = parse_args()
    cfg = load_config(config_path, overrides)

    set_seed(cfg.get("seed", 42))
    cfg = resolve_save_dir(cfg)

    mode = cfg.get("mode", "sft")
    if mode == "sft":
        run_sft(cfg)
    elif mode == "unlearn":
        run_unlearn(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'sft' or 'unlearn'.")


if __name__ == "__main__":
    main()
