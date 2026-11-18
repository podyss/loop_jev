"""Token targets and shared LM-head readout for recurrent CoT supervision."""

import hashlib
from itertools import permutations


IGNORED = -100
_SHUFFLED_ORDERS = tuple(order for order in permutations(range(3)) if order != (0, 1, 2))


def round_targets(token_ids, positions, *, order, row_id, seed):
    """Divide the CoT into contiguous thirds and sample each third in order."""
    segments = []
    for index in range(3):
        start = index * len(token_ids) // 3
        stop = (index + 1) * len(token_ids) // 3
        count = min(positions, stop - start)
        targets = [token_ids[start + (2 * i + 1) * (stop - start) // (2 * count)]
                   for i in range(count)]
        segments.append(targets + [IGNORED] * (positions - count))
    if order == "ordered":
        return segments
    if order != "shuffled":
        raise ValueError(f"Unknown CoT round order: {order}")
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    assignment = _SHUFFLED_ORDERS[int.from_bytes(digest[:8], "big") % len(_SHUFFLED_ORDERS)]
    return [segments[index] for index in assignment]


def add_workspace_readout(model, positions, answer_length):
    """Project only recurrent slots and the answer in the ordinary head call."""
    import torch
    import torch.nn.functional as F

    def prepend_workspace(_module, inputs):
        states = model.model.loop_workspace_states
        workspace = torch.cat(states, dim=1)
        workspace = F.rms_norm(workspace, (workspace.shape[-1],), eps=model.config.rms_norm_eps)
        return (torch.cat((workspace, inputs[0]), dim=1),)

    return (slice(-positions - answer_length, -answer_length),
            model.lm_head.register_forward_pre_hook(prepend_workspace))


def token_loss(logits, targets):
    import torch.nn.functional as F

    targets = targets.flatten()
    logits = logits.reshape(-1, logits.shape[-1])
    total = F.cross_entropy(logits.float(), targets, ignore_index=IGNORED, reduction="sum")
    return total / (targets != IGNORED).sum().clamp_min(1)
