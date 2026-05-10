import torch
import torch.nn.functional as F


def get_batch_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.transpose(1, 2), shift_labels)

    mask = (shift_labels != -100).float()
    return (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def compute_ce_loss(model, input_ids, labels, attention_mask):
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    return outputs.loss, outputs


def grad_ascent_loss(model, forget_batch, retain_batch, retain_weight=0.0, **kwargs):
    f_ids, f_labels, f_mask = [x.to(model.device) for x in forget_batch]
    forget_loss, outputs = compute_ce_loss(model, f_ids, f_labels, f_mask)
    loss = -forget_loss

    if retain_weight > 0 and retain_batch is not None:
        r_ids, r_labels, r_mask = [x.to(model.device) for x in retain_batch]
        retain_loss, _ = compute_ce_loss(model, r_ids, r_labels, r_mask)
        loss = loss + retain_weight * retain_loss

    return loss, outputs


def npo_loss(model, oracle_model, forget_batch, retain_batch,
             retain_weight=1.0, beta=0.1, **kwargs):
    f_ids, f_labels, f_mask = [x.to(model.device) for x in forget_batch]

    outputs = model(input_ids=f_ids, attention_mask=f_mask, labels=f_labels)
    forget_loss_current = get_batch_loss(outputs.logits, f_labels)

    with torch.no_grad():
        oracle_out = oracle_model(input_ids=f_ids, attention_mask=f_mask, labels=f_labels)
        forget_loss_oracle = get_batch_loss(oracle_out.logits, f_labels)

    log_ratio = forget_loss_current - forget_loss_oracle
    loss = -F.logsigmoid(beta * log_ratio).mean() * 2 / beta

    if retain_weight > 0 and retain_batch is not None:
        r_ids, r_labels, r_mask = [x.to(model.device) for x in retain_batch]
        retain_loss, _ = compute_ce_loss(model, r_ids, r_labels, r_mask)
        loss = loss + retain_weight * retain_loss

    return loss, outputs


def dpo_loss(model, oracle_model, forget_batch, idk_batch,
             retain_batch, retain_weight=1.0, beta=0.1, **kwargs):
    f_ids, f_labels, f_mask = [x.to(model.device) for x in forget_batch]
    idk_ids, idk_labels, idk_mask = [x.to(model.device) for x in idk_batch]

    forget_out = model(input_ids=f_ids, attention_mask=f_mask, labels=f_labels)
    idk_out = model(input_ids=idk_ids, attention_mask=idk_mask, labels=idk_labels)

    with torch.no_grad():
        forget_out_oracle = oracle_model(
            input_ids=f_ids, attention_mask=f_mask, labels=f_labels
        )
        idk_out_oracle = oracle_model(
            input_ids=idk_ids, attention_mask=idk_mask, labels=idk_labels
        )

    idk_loss_current = -get_batch_loss(idk_out.logits, idk_labels)
    forget_loss_current = -get_batch_loss(forget_out.logits, f_labels)
    idk_loss_oracle = -get_batch_loss(idk_out_oracle.logits, idk_labels)
    forget_loss_oracle = -get_batch_loss(forget_out_oracle.logits, f_labels)

    pi_logratios = idk_loss_current - forget_loss_current
    ref_logratios = idk_loss_oracle - forget_loss_oracle

    loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

    if retain_weight > 0 and retain_batch is not None:
        r_ids, r_labels, r_mask = [x.to(model.device) for x in retain_batch]
        retain_loss, _ = compute_ce_loss(model, r_ids, r_labels, r_mask)
        loss = loss + retain_weight * retain_loss

    return loss, forget_out


def task_vector_loss(model, forget_batch, retain_batch, retain_weight=0.0, **kwargs):
    # Standard CE on forget set; the task-vector subtraction happens after training.
    f_ids, f_labels, f_mask = [x.to(model.device) for x in forget_batch]
    loss, outputs = compute_ce_loss(model, f_ids, f_labels, f_mask)

    if retain_weight > 0 and retain_batch is not None:
        import warnings
        warnings.warn(
            "[Task Vector] retain_weight > 0 will pollute the task vector with retain "
            "knowledge and degrade retain utility after negation. Use retain_weight=0.0.",
            UserWarning, stacklevel=2,
        )

    return loss, outputs


LOSS_REGISTRY = {
    "grad_ascent": grad_ascent_loss,
    "grad_diff": grad_ascent_loss,
    "npo": npo_loss,
    "dpo": dpo_loss,
    "task_vector": task_vector_loss,
}

NEEDS_ORACLE = {"npo", "dpo"}
