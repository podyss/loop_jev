"""Forward and backward arithmetic messages at a shared intermediate quantity."""

import math
from fractions import Fraction

import torch
from torch import nn
from torch.nn import functional as F

from loop_jev.training.program_composition import MAX_COEFFICIENT, _affine, _verified_trace


def compile_targets(row):
    if len(row["options"]) != 4:
        return None
    trace = _verified_trace(row)
    if trace is None:
        return None
    try:
        options = [Fraction(str(option).replace(",", "")) for option in row["options"]]
    except (ValueError, ZeroDivisionError):
        return None
    if len(set(options)) != len(options):
        return None
    steps, values, _, _ = trace
    last = len(steps) - 1
    for pivot in sorted(range(last), key=lambda i: abs(i - (last - 1) / 2)):
        try:
            _, forward, _ = _affine(steps, pivot)
            slope, intercept, _ = _affine(steps, last, pivot=pivot)
        except ValueError:
            continue
        if not slope:
            continue
        backward = [(option - intercept) / slope for option in options]
        if forward != values[pivot] or backward[row["gold_index"]] != forward:
            continue
        if max(abs(value) for value in [forward, *backward]) > MAX_COEFFICIENT:
            continue
        messages = [math.asinh(value) for value in [forward, *backward]]
        distances = [abs(messages[0] - value) for value in messages[1:]]
        # Exclude near ties that are indistinguishable after target compression.
        if min(distance for i, distance in enumerate(distances) if i != row["gold_index"]) < 1e-4:
            continue
        return [*messages, *distances]
    return None


class ArithmeticReadout(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.heads = nn.ModuleList(nn.Sequential(nn.RMSNorm(hidden_size), nn.Linear(hidden_size, width))
                                   for width in (1, 4, 4))

    def forward(self, states):
        return torch.cat([head(state.float()) for head, state in zip(self.heads, states, strict=True)], dim=-1)


def arithmetic_loss(predictions, targets, valid):
    predictions = predictions.float()
    message_loss = F.smooth_l1_loss(predictions, targets, reduction="none")
    # Average within each stage before averaging the three stages.
    fit = (message_loss[:, 0] + message_loss[:, 1:5].mean(-1) + message_loss[:, 5:].mean(-1)) / 3
    comparison = (predictions[:, :1] - predictions[:, 1:5]).abs()
    consistency = F.smooth_l1_loss(predictions[:, 5:], comparison, reduction="none").mean(-1)
    stats = torch.stack(((fit.detach() * valid).sum(), (consistency.detach() * valid).sum(), valid.sum()))
    return ((fit + 0.25 * consistency) * valid).mean(), stats
