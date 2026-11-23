"""A recurrent reasoning core over slots and a fixed decoder memory."""

from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


class LatentCore(nn.Module):
    def __init__(self, hidden_size, width, output_projection):
        super().__init__()
        self.heads = 4
        self.slot_in = nn.Linear(hidden_size, width, bias=False)
        self.memory_kv = nn.Linear(hidden_size, 2 * width, bias=False)
        self.cross_norm = nn.LayerNorm(width)
        self.cross_q = nn.Linear(width, width, bias=False)
        self.cross_out = nn.Linear(width, width, bias=False)
        self.self_norm = nn.LayerNorm(width)
        self.self_qkv = nn.Linear(width, 3 * width, bias=False)
        self.self_out = nn.Linear(width, width, bias=False)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width, bias=False),
                                 nn.GELU(), nn.Linear(2 * width, width, bias=False))
        self.readout_norm = nn.LayerNorm(width)
        self.output = output_projection(width, hidden_size, bias=False)
        nn.init.zeros_(self.output.weight)

    def attend(self, q, k, v, mask=None):
        def split(x):
            return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)

        out = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=mask)
        return out.transpose(1, 2).flatten(2)

    def forward(self, memory, valid, slots, rounds, reset_last=False):
        initial = self.slot_in(slots)
        state = initial
        k, v = self.memory_kv(memory).chunk(2, dim=-1)
        mask = valid[:, None, None, :].bool() if valid is not None else None
        for index in range(rounds):
            if reset_last and index == rounds - 1:
                state = initial
            state = state + self.cross_out(self.attend(self.cross_q(self.cross_norm(state)), k, v, mask))
            q, sk, sv = self.self_qkv(self.self_norm(state)).chunk(3, dim=-1)
            state = state + self.self_out(self.attend(q, sk, sv))
            state = state + self.ffn(state)
        return self.output(self.readout_norm(state).mean(dim=1, keepdim=True))


def enable_latent_core(model, *, rounds=None, reset_last=False):
    """Use a single native Qwen forward, then update only the final answer position."""
    config = model.config
    rounds = config.loop_count if rounds is None else rounds
    if not 1 <= rounds <= config.loop_count or (reset_last and rounds != config.loop_count):
        raise ValueError("Invalid core depth or reset mode")
    suffix = config.latent_core_suffix
    end = config.latent_slots + suffix
    backbone = model.model

    def core_forward(this, input_ids=None, attention_mask=None, **kwargs):
        kwargs["use_cache"] = False
        output = this._unlooped_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = output.last_hidden_state
        if hidden.shape[1] <= end or (attention_mask is not None and attention_mask.ndim != 2):
            raise ValueError("Latent core requires a prompt prefix and a 2D padding mask")
        valid = attention_mask[:, :-end] if attention_mask is not None else None
        delta = model.latent_core(hidden[:, :-end], valid, hidden[:, -end:-suffix], rounds, reset_last)
        answer = hidden[:, -1:] + delta.to(hidden.dtype)
        this.loop_hidden_states = [answer]
        output.last_hidden_state = torch.cat((hidden[:, :-1], answer), dim=1)
        return output

    if not hasattr(backbone, "_unlooped_forward"):
        backbone._unlooped_forward = backbone.forward
    backbone.forward = MethodType(core_forward, backbone)
    return model
