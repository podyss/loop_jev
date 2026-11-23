"""Answer losses for independently evaluated layerwise depths."""

import torch
from torch.nn import functional as F


def depth_answer_ce(logits, candidate_ids, valid_options, targets):
    options = logits[:, -1, candidate_ids].float()
    options = options.masked_fill(~valid_options, torch.finfo(options.dtype).min)
    return F.cross_entropy(options, targets, reduction="none")


def adjacent_penalty(shallow_ce, deep_ce, margin=0.05):
    # Make the deeper path improve instead of raising the shallower path's loss.
    target = (shallow_ce.detach() - margin).clamp_min(0)
    return (deep_ce - target).relu().mean()
