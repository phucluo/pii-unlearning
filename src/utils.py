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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


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


def load_model_and_tokenizer(cfg, model_cfg):
    """
    Load model + tokenizer with optional quantization + LoRA.
    Mirrors UnlearnPII's forget.py create_model() but simplified.
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # --- LoRA ---
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
        model.print_trainable_parameters()

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

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
