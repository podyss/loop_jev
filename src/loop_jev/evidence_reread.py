"""Route fresh evidence into the latent workspace between block rounds."""

import torch
import torch.nn.functional as F


def evidence_bank(prefix, valid):
    values = prefix.float().masked_fill(~valid[..., None], 0)
    center = values.sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1)[..., None]
    values = (values - center).masked_fill(~valid[..., None], 0)
    return F.normalize(values, dim=-1), values


def reread_evidence(state, previous, end, slots, keys, values, valid, coverage):
    start = end - slots
    workspace = state[:, start:end].float()
    update = workspace - previous[:, start:end].float()
    # Query with both the current state and what changed in this round.
    query = F.normalize(F.normalize(workspace, dim=-1) + F.normalize(update, dim=-1), dim=-1)
    scores = 8.0 * (query @ keys.transpose(1, 2)).float()
    if coverage is not None:
        # Subtract a penalty per slot relative to its previous peak attention.
        scores = scores - 2.0 * coverage / coverage.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scores = scores.masked_fill(~valid[:, None, :], torch.finfo(scores.dtype).min)
    attention = scores.softmax(dim=-1).masked_fill(~valid[:, None, :], 0)
    message = (attention @ values).float()
    # Bound the write by half the norm of the existing damped slot update.
    scale = update.detach().norm(dim=-1, keepdim=True) * 0.5
    message = message * (scale / message.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp_max(1.0)
    workspace = (workspace + message).to(state.dtype)
    return torch.cat((state[:, :start], workspace, state[:, end:]), dim=1), attention.detach()
