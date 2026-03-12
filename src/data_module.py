"""
Data loading & tokenization — on-the-fly, no pre-tokenized JSONL needed.
Mirrors UnlearnPII's data_module.py but simplified.

Key design: JSON files are read raw, tokenization happens in __getitem__().
This means switching model_family only requires changing the config, not regenerating data.
"""
import json
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.utils import get_model_identifiers


# ========================= TOKENIZATION (ON-THE-FLY) =========================

def convert_to_model_format(tokenizer, max_length, question, answer, model_configs):
    """
    Tokenize a single QA pair with completion-only loss masking.
    Mirrors UnlearnPII's convert_raw_data_to_model_format() exactly.

    Returns: (input_ids, labels, attention_mask) as tensors
    """
    q_start = model_configs["question_start_tag"]
    q_end = model_configs["question_end_tag"]
    a_start = model_configs["answer_tag"]
    a_end = model_configs["answer_end_tag"]

    new_question = q_start + question + q_end
    new_answer = a_start + answer + a_end
    full_text = new_question + new_answer

    # Count question tokens → will be masked in labels
    num_q_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )

    pad_length = max_length - len(encoded.input_ids)

    # Pad input_ids and attention_mask
    input_ids = encoded["input_ids"] + [tokenizer.pad_token_id] * pad_length
    attention_mask = encoded["attention_mask"] + [0] * pad_length

    # Labels: mask question tokens with -100, pad with -100
    if len(encoded.input_ids) == max_length:
        labels = list(encoded.input_ids)
    else:
        labels = encoded["input_ids"] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)

    for i in range(num_q_tokens):
        labels[i] = -100

    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(attention_mask),
    )


# ========================= DATASETS =========================

class SFTDataset(Dataset):
    """
    Dataset for SFT Exposed step.
    Reads full_with_qa.json and tokenizes on-the-fly.
    """
    def __init__(self, data_path, tokenizer, model_family, max_length=500,
                 question_key="question", answer_key="answer"):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.model_configs = get_model_identifiers(model_family)
        self.max_length = max_length
        self.question_key = question_key
        self.answer_key = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return convert_to_model_format(
            self.tokenizer, self.max_length,
            item[self.question_key], item[self.answer_key],
            self.model_configs,
        )


class ForgetRetainDataset(Dataset):
    """
    Dataset for unlearning step — returns (forget, retain, idk) triples.
    Mirrors UnlearnPII's CommonForgetQA.

    Each __getitem__ returns a tuple of 3 tensors groups:
      forget_batch: (input_ids, labels, attention_mask)
      retain_batch: (input_ids, labels, attention_mask)
      idk_batch:    (input_ids, labels, attention_mask) — for DPO/NPO
    """
    def __init__(self, forget_path, retain_path, idk_path,
                 tokenizer, model_family, max_length=500,
                 question_key="question", answer_key="answer"):

        with open(forget_path) as f:
            self.forget_data = json.load(f)
        with open(retain_path) as f:
            self.retain_data = json.load(f)

        # IDK responses (simple text lines)
        self.idk_responses = []
        if idk_path and os.path.exists(idk_path):
            with open(idk_path) as f:
                self.idk_responses = [line.strip() for line in f if line.strip()]

        self.tokenizer = tokenizer
        self.model_configs = get_model_identifiers(model_family)
        self.max_length = max_length
        self.question_key = question_key
        self.answer_key = answer_key

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        # Forget sample
        f_item = self.forget_data[idx]
        forget = convert_to_model_format(
            self.tokenizer, self.max_length,
            f_item[self.question_key], f_item[self.answer_key],
            self.model_configs,
        )

        # Retain sample (cycle through if shorter)
        r_idx = idx % len(self.retain_data)
        r_item = self.retain_data[r_idx]
        retain = convert_to_model_format(
            self.tokenizer, self.max_length,
            r_item[self.question_key], r_item[self.answer_key],
            self.model_configs,
        )

        # IDK sample (for DPO/NPO — same question, IDK answer)
        if self.idk_responses:
            idk_answer = random.choice(self.idk_responses)
            idk = convert_to_model_format(
                self.tokenizer, self.max_length,
                f_item[self.question_key], idk_answer,
                self.model_configs,
            )
        else:
            idk = forget  # fallback

        return forget, retain, idk


# ========================= COLLATORS =========================

def sft_collator(batch):
    """Collate for SFT — batch of (input_ids, labels, attention_mask)."""
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    attention_mask = torch.stack([b[2] for b in batch])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def forget_collator(batch):
    """Collate for unlearning — batch of (forget, retain, idk) triples."""
    forget = tuple(torch.stack([b[0][i] for b in batch]) for i in range(3))
    retain = tuple(torch.stack([b[1][i] for b in batch]) for i in range(3))
    idk = tuple(torch.stack([b[2][i] for b in batch]) for i in range(3))
    return forget, retain, idk


# ========================= DATALOADER HELPERS =========================

def get_sft_dataloader(cfg, tokenizer):
    """Create SFT dataloader from config."""
    ds = SFTDataset(
        cfg["data_path"], tokenizer, cfg["model_family"],
        max_length=cfg.get("max_length", 500),
    )
    return DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=sft_collator, num_workers=0,
    )


def get_forget_retain_dataloaders(cfg, tokenizer):
    """Create forget+retain dataloader from config."""
    # Determine split files
    split = cfg["split"]  # e.g. "forget10"
    retain_split = split.replace("forget", "retain")  # "retain90"

    forget_path = os.path.join(cfg["forget_data_path"], f"{split}.json")
    retain_path = os.path.join(cfg["retain_data_path"], f"{retain_split}.json")
    idk_path = cfg.get("idk_path", "data/raw/idontknow.jsonl")

    ds = ForgetRetainDataset(
        forget_path, retain_path, idk_path,
        tokenizer, cfg["model_family"],
        max_length=cfg.get("max_length", 500),
    )
    return DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=forget_collator, num_workers=0,
    )
