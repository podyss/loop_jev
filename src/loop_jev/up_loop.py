"""Qwen3 decoder with UpLoop's branch-specific residual writes."""

import math

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer


GATE_SCALE = 8.0
GATE_INITIAL = 0.3
INITIAL_LOGIT = math.log(GATE_INITIAL / (1 - GATE_INITIAL))


class ResidualGates(nn.Module):
    def __init__(self, rounds, hidden_size):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(rounds - 1, 2, hidden_size))

    def forward(self, local_round):
        return (self.raw[local_round - 1].float() * GATE_SCALE + INITIAL_LOGIT).sigmoid().unbind(0)


class UpLoopQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.loop_gates = ResidualGates(config.loop_count, config.hidden_size)

    def forward(self, hidden_states, *, local_round=0, **kwargs):
        if local_round == 0:
            return super().forward(hidden_states, **kwargs)

        attention_gate, mlp_gate = self.loop_gates(local_round)
        attention, _ = self.self_attn(hidden_states=self.input_layernorm(hidden_states), **kwargs)
        state = hidden_states + attention * attention_gate.to(attention.dtype)
        update = self.mlp(self.post_attention_layernorm(state))
        return state + update * mlp_gate.to(update.dtype)
