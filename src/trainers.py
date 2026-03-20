"""
Unlearning loss functions — GA, GD, NPO, DPO, Task Vector, AAU-PII.
Mirrors the compute_loss() switch-case in UnlearnPII's dataloader.py.

Each function takes model outputs and returns a scalar loss.
"""
import torch
import torch.nn.functional as F


def get_batch_loss(logits, labels):
    """
    Compute per-sample cross-entropy loss (completion-only).
    Same as UnlearnPII's get_batch_loss in data_module.py.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    # (batch, seq_len)
    loss = loss_fn(shift_logits.transpose(1, 2), shift_labels)

    # Mask padding (-100) and mean over valid tokens per sample
    mask = (shift_labels != -100).float()
    per_sample_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return per_sample_loss  # (batch,)


def compute_ce_loss(model, input_ids, labels, attention_mask):
    """Forward pass + CE loss."""
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    return outputs.loss, outputs


# ========================= LOSS FUNCTIONS =========================

def grad_ascent_loss(model, forget_batch, retain_batch, retain_weight=0.0, **kwargs):
    """
    Gradient Ascent: maximize loss on forget set = negate CE loss.
    With retain_weight > 0, becomes Gradient Difference (GA + retain regularization).
    """
    f_ids, f_labels, f_mask = [x.to(model.device) for x in forget_batch]

    # Forget: NEGATE loss (maximize → model "unlearns")
    forget_loss, outputs = compute_ce_loss(model, f_ids, f_labels, f_mask)
    loss = -forget_loss

    # Retain: standard CE (minimize → preserve knowledge)
    if retain_weight > 0 and retain_batch is not None:
        r_ids, r_labels, r_mask = [x.to(model.device) for x in retain_batch]
        retain_loss, _ = compute_ce_loss(model, r_ids, r_labels, r_mask)
        loss = loss + retain_weight * retain_loss

    return loss, outputs


def npo_loss(model, oracle_model, forget_batch, retain_batch,
             retain_weight=1.0, beta=0.1, **kwargs):
    """
    Negative Preference Optimization (NPO).
    Mirrors UnlearnPII's compute_npo_loss().
    """
    f_ids, f_labels, f_mask = [x.to(model.device) for x in forget_batch]

    # Current model loss on forget
    outputs = model(input_ids=f_ids, attention_mask=f_mask, labels=f_labels)
    forget_loss_current = get_batch_loss(outputs.logits, f_labels)

    # Oracle (reference) model loss on forget
    with torch.no_grad():
        oracle_out = oracle_model(input_ids=f_ids, attention_mask=f_mask, labels=f_labels)
        forget_loss_oracle = get_batch_loss(oracle_out.logits, f_labels)

    # NPO objective
    log_ratio = forget_loss_current - forget_loss_oracle
    loss = -F.logsigmoid(beta * log_ratio).mean() * 2 / beta

    # Retain regularization (GD-style)
    if retain_weight > 0 and retain_batch is not None:
        r_ids, r_labels, r_mask = [x.to(model.device) for x in retain_batch]
        retain_loss, _ = compute_ce_loss(model, r_ids, r_labels, r_mask)
        loss = loss + retain_weight * retain_loss

    return loss, outputs


def dpo_loss(model, oracle_model, forget_batch, idk_batch,
             retain_batch, retain_weight=1.0, beta=0.1, **kwargs):
    """
    DPO for unlearning: prefer IDK responses over PII-leaking responses.
    Mirrors UnlearnPII's compute_dpo_loss().
    """
    f_ids, f_labels, f_mask = [x.to(model.device) for x in forget_batch]
    idk_ids, idk_labels, idk_mask = [x.to(model.device) for x in idk_batch]

    # Current model
    forget_out = model(input_ids=f_ids, attention_mask=f_mask, labels=f_labels)
    idk_out = model(input_ids=idk_ids, attention_mask=idk_mask, labels=idk_labels)

    # Oracle model
    with torch.no_grad():
        forget_out_oracle = oracle_model(input_ids=f_ids, attention_mask=f_mask, labels=f_labels)
        idk_out_oracle = oracle_model(input_ids=idk_ids, attention_mask=idk_mask, labels=idk_labels)

    # Log-ratios
    idk_loss_current = -get_batch_loss(idk_out.logits, idk_labels)
    forget_loss_current = -get_batch_loss(forget_out.logits, f_labels)
    idk_loss_oracle = -get_batch_loss(idk_out_oracle.logits, idk_labels)
    forget_loss_oracle = -get_batch_loss(forget_out_oracle.logits, f_labels)

    pi_logratios = idk_loss_current - forget_loss_current
    ref_logratios = idk_loss_oracle - forget_loss_oracle

    loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

    # Retain
    if retain_weight > 0 and retain_batch is not None:
        r_ids, r_labels, r_mask = [x.to(model.device) for x in retain_batch]
        retain_loss, _ = compute_ce_loss(model, r_ids, r_labels, r_mask)
        loss = loss + retain_weight * retain_loss

    return loss, forget_out


def task_vector_loss(model, forget_batch, retain_batch, retain_weight=0.0, **kwargs):
    """
    Task Vector: train normally on forget set (CE loss), then SUBTRACT the learned
    LoRA weights from base. The subtraction is done AFTER training, not during.
    So the training loss here is standard SFT on forget set ONLY.

    IMPORTANT: retain_weight MUST be 0.0 for correct Task Vector behavior.
    If retain data is included here, the task vector will contain both forget AND
    retain knowledge changes. When negated, retain knowledge gets reversed too,
    causing collateral damage on the retain set.
    Correct formula: task_vec = theta_forget_only - theta_SFT (pure forget direction)
    """
    f_ids, f_labels, f_mask = [x.to(model.device) for x in forget_batch]
    loss, outputs = compute_ce_loss(model, f_ids, f_labels, f_mask)

    # NOTE: retain regularization intentionally disabled for Task Vector.
    # The task vector must capture ONLY forget-specific changes.
    if retain_weight > 0 and retain_batch is not None:
        import warnings
        warnings.warn(
            "[Task Vector] retain_weight > 0 will pollute the task vector with retain "
            "knowledge and degrade retain utility after negation. Use retain_weight=0.0.",
            UserWarning, stacklevel=2,
        )

    return loss, outputs


# ========================= DISPATCHER =========================

LOSS_REGISTRY = {
    "grad_ascent": grad_ascent_loss,
    "grad_diff": grad_ascent_loss,      # same fn, just retain_weight > 0
    "npo": npo_loss,
    "dpo": dpo_loss,
    "task_vector": task_vector_loss,
    # "aau_pii": aau_pii_loss,          # TODO: implement
}

NEEDS_ORACLE = {"npo", "dpo"}  # methods that require a reference model
