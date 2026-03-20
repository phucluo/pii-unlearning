"""
train.py — Entry point for SFT Exposed and Unlearning.
Mirrors UnlearnPII's finetune.py (mode=sft) and forget.py (mode=unlearn).

Usage:
  python train.py --config configs/pii_sft.yaml                          # SFT (PII track)
  python train.py --config configs/tofu_sft.yaml                         # SFT (TOFU track)
  python train.py --config configs/pii_unlearn.yaml                      # Unlearn PII (default GA)
  python train.py --config configs/tofu_unlearn.yaml --forget_loss=npo   # Unlearn TOFU
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


# ========================= CHECKPOINT UTILS =========================

def save_checkpoint(model, tokenizer, optimizer, epoch, global_step, save_dir):
    """Save mid-training checkpoint với metadata để resume."""
    ckpt_dir = os.path.join(save_dir, f"checkpoint-step{global_step}")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    # Lưu optimizer state + progress
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    meta = {"epoch": epoch, "global_step": global_step}
    with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[CKPT] Saved checkpoint → {ckpt_dir}")
    return ckpt_dir


def find_latest_checkpoint(save_dir):
    """Tìm checkpoint mới nhất trong save_dir (theo global_step lớn nhất)."""
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None, 0, 0
    ckpts = sorted(
        [d for d in save_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-step")],
        key=lambda d: int(d.name.replace("checkpoint-step", ""))
    )
    if not ckpts:
        return None, 0, 0
    latest = ckpts[-1]
    meta_path = latest / "checkpoint_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[CKPT] Found checkpoint: {latest} (epoch={meta['epoch']}, step={meta['global_step']})")
        return str(latest), meta["epoch"], meta["global_step"]
    return None, 0, 0


# ========================= SFT =========================

def run_sft(cfg):
    """Bước 1: SFT Exposed — fine-tune model to memorize PII."""
    print("=" * 60)
    print("STEP 1: SFT Exposed")
    print("=" * 60)

    save_dir = cfg["save_dir"]
    save_steps = cfg.get("save_steps", 500)      # save mỗi 500 steps
    resume = cfg.get("resume", True)             # mặc định auto-resume

    # --- Auto-resume từ checkpoint mới nhất ---
    start_epoch, global_step = 0, 0
    ckpt_path = cfg.get("model_path")            # manual override nếu có

    if resume and not ckpt_path:
        ckpt_path, start_epoch, global_step = find_latest_checkpoint(save_dir)

    # Override model_path để load_model_and_tokenizer tự xử lý PEFT
    if ckpt_path:
        cfg["model_path"] = ckpt_path
        print(f"[INFO] Resuming from epoch={start_epoch}, step={global_step}")

    model_cfg = get_model_identifiers(cfg["model_family"])
    model, tokenizer = load_model_and_tokenizer(cfg, model_cfg)
    dataloader = get_sft_dataloader(cfg, tokenizer)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01),
    )

    # Resume optimizer state nếu có
    if ckpt_path:
        opt_path = os.path.join(ckpt_path, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            print(f"[INFO] Optimizer state restored from {opt_path}")

    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    num_epochs = cfg.get("num_epochs", 5)

    model.train()
    for epoch in range(start_epoch, num_epochs):
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

                # --- Save checkpoint mỗi save_steps ---
                if global_step % save_steps == 0:
                    save_checkpoint(model, tokenizer, optimizer, epoch, global_step, save_dir)

            total_loss += loss.item() * grad_accum
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}", step=global_step)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

        # --- Save checkpoint cuối mỗi epoch ---
        save_checkpoint(model, tokenizer, optimizer, epoch + 1, global_step, save_dir)

    # Save final model (không có optimizer state — dùng cho inference/unlearning)
    save_model(model, tokenizer, save_dir)
    print(f"SFT Exposed model saved to {save_dir}")


# ========================= UNLEARN =========================

def run_unlearn(cfg):
    """Bước 2: Unlearning — apply forget method on SFT Exposed model."""
    print("=" * 60)
    print(f"STEP 2: Unlearning — method={cfg['forget_loss']}")
    print("=" * 60)

    save_dir = cfg["save_dir"]
    save_steps = cfg.get("save_steps", 200)
    resume = cfg.get("resume", True)

    # --- Auto-resume ---
    start_epoch, global_step = 0, 0
    ckpt_path = cfg.get("model_path")

    if resume and not ckpt_path:
        ckpt_path, start_epoch, global_step = find_latest_checkpoint(save_dir)

    if ckpt_path:
        cfg["model_path"] = ckpt_path
        print(f"[INFO] Resuming from epoch={start_epoch}, step={global_step}")

    model_cfg = get_model_identifiers(cfg["model_family"])
    model, tokenizer = load_model_and_tokenizer(cfg, model_cfg)
    dataloader = get_forget_retain_dataloaders(cfg, tokenizer)

    # Oracle model (NPO/DPO: reference during training | task_vector: SFT snapshot for negation)
    oracle_model = None
    if cfg["forget_loss"] in NEEDS_ORACLE or cfg["forget_loss"] == "task_vector":
        print("Loading oracle (reference) model...")
        oracle_cfg = copy.deepcopy(cfg)
        oracle_cfg["lora"] = {"r": 0}
        oracle_model, _ = load_model_and_tokenizer(oracle_cfg, model_cfg)
        oracle_model.eval()
        for p in oracle_model.parameters():
            p.requires_grad = False

    loss_fn = LOSS_REGISTRY[cfg["forget_loss"]]

    # Task Vector requires retain_weight=0 — override + warn if misconfigured
    if cfg["forget_loss"] == "task_vector":
        rw = cfg.get("retain_weight", 0.0)
        if rw != 0.0:
            print(f"[WARN] Task Vector: retain_weight={rw} will pollute the task vector.")
            print(f"[WARN] Overriding retain_weight → 0.0 for correct Task Vector behavior.")
            cfg["retain_weight"] = 0.0

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01),
    )

    if ckpt_path:
        opt_path = os.path.join(ckpt_path, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            print(f"[INFO] Optimizer state restored")

    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    num_epochs = cfg.get("num_epochs", 8)
    max_steps = cfg.get("max_steps")

    model.train()
    for epoch in range(start_epoch, num_epochs):
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

                if global_step % save_steps == 0:
                    save_checkpoint(model, tokenizer, optimizer, epoch, global_step, save_dir)

            total_loss += loss.item() * grad_accum
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}", step=global_step)

            if max_steps and global_step >= max_steps:
                break

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

        # Save cuối epoch
        save_checkpoint(model, tokenizer, optimizer, epoch + 1, global_step, save_dir)

        if max_steps and global_step >= max_steps:
            print(f"Reached max_steps={max_steps}, stopping.")
            break

    # Task Vector negation: θ_unlearn = θ_SFT - α*(θ_forget - θ_SFT) = (1+α)*θ_SFT - α*θ_forget
    if cfg["forget_loss"] == "task_vector":
        tv_alpha = cfg.get("tv_alpha", 1.0)
        print(f"[Task Vector] Applying negation: θ_unlearn = {1+tv_alpha}×θ_SFT - {tv_alpha}×θ_forget (alpha={tv_alpha}) ...")
        with torch.no_grad():
            for (name, param), (_, sft_param) in zip(
                model.named_parameters(), oracle_model.named_parameters()
            ):
                if param.requires_grad:
                    # task_vector = θ_forget - θ_SFT
                    # negated     = θ_SFT - α * task_vector
                    param.data = sft_param.data - tv_alpha * (param.data - sft_param.data)
        print(f"[Task Vector] Negation done (alpha={tv_alpha})")

    save_model(model, tokenizer, save_dir)

    config_save = {k: str(v) if not isinstance(v, (int, float, bool, type(None), list, dict)) else v
                   for k, v in cfg.items()}
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(config_save, f, indent=2)
    print(f"Unlearned model saved to {save_dir}")


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