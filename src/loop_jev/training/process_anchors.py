"""Semantic CoT prefixes for loop-wise teacher supervision."""

import math
import re


def split_reasoning_steps(rationale):
    """Prefer dataset line boundaries; fall back to sentence boundaries.

    Never split decimals or the interior of GSM8K arithmetic annotations.
    A rationale without a safe boundary remains a single step.
    """
    lines = [line.strip() for line in rationale.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", rationale.strip())
            if part.strip()]


def make_process_prefixes(steps, loop_count):
    """Map loops to ceil(k * step_count / loop_count) whole-step prefixes."""
    if loop_count < 1:
        raise ValueError("loop_count must be positive")
    return ["\n".join(steps[:math.ceil(k * len(steps) / loop_count)])
            for k in range(1, loop_count + 1)]


def process_alignment_loss(student_states, teacher_states, valid, discount=1.0):
    """Weighted mean cosine distance at normalized final-layer readouts.

    Each tensor is [batch, hidden]. Targets are always detached. Empty
    rationales contribute zero, including in batches with no valid anchors.
    """
    import torch.nn.functional as F

    if not student_states or len(student_states) != len(teacher_states):
        raise ValueError("Expected matching nonempty student and teacher trajectories")
    if not 0 < discount <= 1:
        raise ValueError("discount must be in (0, 1]")
    mask = valid.float()
    weights = [discount ** k for k in range(len(student_states))]
    distances = [1 - F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1)
                 for student, teacher in zip(student_states, teacher_states)]
    weighted = sum(weight * distance for weight, distance in zip(weights, distances))
    return (weighted * mask).sum() / (mask.sum().clamp_min(1) * sum(weights))


def delta_alignment_loss(student_updates, teacher_states, valid):
    """Relative MSE of whole updates, scaled per example by teacher trajectory RMS.

    A shared scale across steps avoids amplifying repeated (zero-delta) prefixes.
    Teacher states and scales are detached; empty rationales are masked.
    """
    import torch
    if not student_updates or len(teacher_states) != len(student_updates) + 1:
        raise ValueError("K updates require K+1 teacher states")
    targets = torch.stack([b.detach().float() - a.detach().float()
                           for a, b in zip(teacher_states[:-1], teacher_states[1:])])
    predictions = torch.stack([state.float() for state in student_updates])
    scale2 = targets.square().mean(dim=(0, 2)).clamp_min(1e-4)
    per_example = (predictions - targets).square().mean(dim=(0, 2)) / scale2
    mask = valid.to(per_example.dtype)
    return (per_example * mask).sum() / mask.sum().clamp_min(1)
