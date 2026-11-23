"""Two-scale residual exchange between rounds of the recurrent block."""

import torch
import torch.nn.functional as F

LOCAL_SLOTS = 18
COARSE_SLOTS = 6
SLOT_COUNT = LOCAL_SLOTS + COARSE_SLOTS


def exchange_workspace(state, previous, suffix, *, upward):
    start = state.shape[1] - suffix - SLOT_COUNT
    split, end = start + LOCAL_SLOTS, state.shape[1] - suffix
    local, coarse = state[:, start:split], state[:, split:end]
    local_delta = local.float() - previous[:, start:split].float()
    if upward:
        restricted = local_delta.reshape(*local_delta.shape[:1], COARSE_SLOTS, -1,
                                         local_delta.shape[-1]).mean(dim=2)
        coarse = (coarse.float() + restricted).to(coarse.dtype)
    else:
        # Route the second-round coarse residual by current local/coarse content.
        coarse_delta = coarse.float() - previous[:, split:end].float()
        affinity = F.normalize(local.float(), dim=-1) @ F.normalize(coarse.float(), dim=-1).transpose(1, 2)
        local = (local.float() + (4.0 * affinity).softmax(dim=-1) @ coarse_delta).to(local.dtype)
    return torch.cat((state[:, :start], local, coarse, state[:, end:]), dim=1)
