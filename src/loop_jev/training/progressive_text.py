"""Explicit rationale-prefix pruning with direct-answer training at every update."""
import math

from loop_jev.training.process_anchors import split_reasoning_steps


def curriculum_stage(update_step, total_steps):
    if total_steps < 4 or not 0 <= update_step < total_steps:
        raise ValueError("Four-stage curriculum requires >=4 updates and an in-range update index")
    return min(3, 4 * update_step // total_steps)


def retained_steps(steps, stage):
    if stage not in range(4):
        raise ValueError("Stage must be 0, 1, 2 or 3")
    if stage == 3:
        return []
    return steps[math.floor(stage * len(steps) / 3):]


def encode_trajectories(tokenizer, prompt_ids, rationale, gold_label, max_length):
    """Removed prefixes are absent from inputs and labels; complete suffixes are targets.

    Reject overlength trajectories instead of silently corrupting arithmetic steps.
    The final gold letter is encoded separately, matching direct-answer readout.
    """
    steps = split_reasoning_steps(rationale)
    result = []
    for stage in range(3):
        suffix = "\n".join(retained_steps(steps, stage))
        response = f"Reasoning: {suffix}\nAnswer: (" if suffix else "Answer: ("
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        response_ids += tokenizer.encode(gold_label, add_special_tokens=False)
        full = prompt_ids + response_ids
        if len(full) > max_length:
            raise ValueError("Progressive trajectory exceeds max-length; increase max-length")
        result.append({"input_ids": full[:-1],
                       "labels": ([-100] * len(prompt_ids) + response_ids)[1:]})
    return result


def collate_trajectories(examples, pad_id, max_targets, device, generator=None):
    """Left padding and already shifted next-token labels.

    Bound vocabulary logits, without shortening the teacher-forced text. Uniform
    target subsampling is an unbiased estimate of each example's token-mean CE.
    """
    import torch
    width = max(len(example["input_ids"]) for example in examples)
    ids, masks, labels = [], [], []
    for example in examples:
        padding = width - len(example["input_ids"])
        ids.append([pad_id] * padding + example["input_ids"])
        masks.append([0] * padding + [1] * len(example["input_ids"]))
        labels.append([-100] * padding + example["labels"])
    labels = torch.tensor(labels, device=device)
    keep = labels != -100
    if max_targets < 1:
        raise ValueError("max_targets must be positive")
    for row in range(len(examples)):
        positions = keep[row].nonzero().flatten()
        if len(positions) > max_targets:
            selected = torch.randperm(len(positions), generator=generator)[:max_targets].to(device)
            keep[row] = False
            keep[row, positions[selected]] = True
    return {
        "input_ids": torch.tensor(ids, device=device),
        "attention_mask": torch.tensor(masks, device=device),
        "logits_to_keep": keep,
    }, labels[keep], keep.nonzero()[:, 0]


def trajectory_loss(logits, targets, owners, batch_size):
    import torch.nn.functional as F
    token_losses = F.cross_entropy(logits.float(), targets, reduction="none")
    sums = token_losses.new_zeros(batch_size).scatter_add(0, owners, token_losses)
    counts = owners.bincount(minlength=batch_size)
    return (sums / counts.clamp_min(1)).mean()


def main():
    # Reuse the established FSDP2 loading, precision, answer KL and checkpoint IO.
    from loop_jev.training.answer_slot_distill import main as train
    train()


if __name__ == "__main__":
    main()
