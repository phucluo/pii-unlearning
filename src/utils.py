"""
Utility functions — config loading, model/tokenizer setup, helpers.
Simplified from UnlearnPII's utils.py + forget.py model creation logic.
"""
import yaml
import argparse
import os
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


# ========================= CONFIG =========================

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(config_path, overrides=None):
    """Load YAML config with optional CLI overrides."""
    cfg = load_yaml(config_path)
    if overrides:
        for kv in overrides:
            key, val = kv.split("=", 1)
            # Auto-cast types
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            elif val.replace(".", "").replace("-", "").replace("e", "").isdigit():
                val = float(val) if "." in val or "e" in val.lower() else int(val)
            elif val.lower() == "null" or val.lower() == "none":
                val = None
            cfg[key] = val
    return cfg


def get_model_identifiers(model_family, config_dir="configs"):
    """Load model chat template tags from model_config.yaml."""
    model_configs = load_yaml(os.path.join(config_dir, "model_config.yaml"))
    if model_family not in model_configs:
        raise ValueError(f"Unknown model_family '{model_family}'. Available: {list(model_configs.keys())}")
    return model_configs[model_family]


def parse_args():
    """Parse --config and any --key=value overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args, unknown = parser.parse_known_args()
    # Parse --key=value pairs
    overrides = []
    for u in unknown:
        if u.startswith("--") and "=" in u:
            overrides.append(u[2:])  # strip --
    return args.config, overrides


def resolve_save_dir(cfg):
    """Resolve ${variable} placeholders in save_dir."""
    save_dir = cfg["save_dir"]
    for key, val in cfg.items():
        if isinstance(val, str):
            save_dir = save_dir.replace(f"${{{key}}}", val)
    cfg["save_dir"] = save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return cfg


# ========================= MODEL =========================

def find_all_linear_names(model):
    """Find all linear layer names for LoRA (same as UnlearnPII)."""
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parts = name.split(".")
            names.add(parts[-1])
    names.discard("lm_head")
    return list(names)


def _is_peft_checkpoint(path):
    """Check if path is a saved PEFT/LoRA adapter (not a full model)."""
    return os.path.isdir(path) and os.path.exists(
        os.path.join(path, "adapter_config.json")
    )


def load_model_and_tokenizer(cfg, model_cfg, is_eval=False):
    """
    Load model + tokenizer with optional quantization + LoRA.
    Supports:
      - Fresh load from HuggingFace (default)
      - Resume from PEFT/LoRA checkpoint via cfg["model_path"]

    Args:
        is_eval: if True, load for inference only (no gradient checkpointing,
                 PEFT adapter loaded as non-trainable). Use for evaluate.py
                 and oracle model in train.py.
    """
    model_id = model_cfg["hf_key"]
    torch_dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16

    # --- Quantization ---
    bnb_config = None
    quant = cfg.get("quantization", "none")
    if quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # --- Load model ---
    model_path = cfg.get("model_path", model_id)

    # Flash Attention 2: chỉ dùng với bf16/fp16, không dùng với quantization
    use_flash_attn = model_cfg.get("use_flash_attn", False) and not bnb_config
    attn_kwargs = {"attn_implementation": "flash_attention_2"} if use_flash_attn else {}
    if use_flash_attn:
        print("[INFO] Flash Attention 2 enabled")
    else:
        print("[INFO] Using default attention (SDPA if available)")

    if _is_peft_checkpoint(model_path):
        # Resume từ LoRA checkpoint đã save:
        # Load base model từ HF trước, sau đó load LoRA adapter
        print(f"[INFO] Detected PEFT checkpoint at '{model_path}'. Loading base + adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            **attn_kwargs,
        )
        if bnb_config:
            base_model = prepare_model_for_kbit_training(base_model)
        # is_trainable=True để tiếp tục train; False cho eval/oracle (tiết kiệm VRAM)
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=not is_eval)
        print(f"[INFO] Resumed LoRA adapter from '{model_path}' (is_eval={is_eval})")
    else:
        # Load fresh từ HuggingFace hoặc local full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            **attn_kwargs,
        )
        if bnb_config:
            model = prepare_model_for_kbit_training(model)

        # --- LoRA (chỉ apply khi load fresh, không apply khi resume) ---
        lora_cfg = cfg.get("lora", {})
        if lora_cfg and lora_cfg.get("r", 0) > 0:
            target_modules = find_all_linear_names(model)
            peft_config = LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["alpha"],
                lora_dropout=lora_cfg.get("dropout", 0.05),
                target_modules=target_modules,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

    # print_trainable_parameters() chỉ có trên PEFT model
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"all params: {total:,} (non-PEFT model)")
    model.config.use_cache = False
    if not is_eval:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def save_model(model, tokenizer, save_dir):
    """Save model + tokenizer + training config."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")