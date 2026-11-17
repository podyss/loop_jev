"""Qwen3 with a checkpointed, shared residual transition between loop passes."""

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3ForCausalLM
from transformers import initialization as init
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


class ZeroInitLinear(nn.Linear):
    """Marker for HF initialization of the residual output projection."""


class LoopTransition(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.loop_transition_rank
        self.scope = getattr(config, "loop_transition_scope", "all-tokens")
        if self.scope not in ("all-tokens", "answer-slot"):
            raise ValueError(f"Unknown transition scope: {self.scope}")
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.down = nn.Linear(config.hidden_size, width, bias=False)
        self.up = ZeroInitLinear(width, config.hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        # Answer-slot training/evaluation left-pad inputs and put the prediction
        # slot at the final position. Restrict the MLP before allocating activations.
        state = hidden_states[:, -1:, :] if self.scope == "answer-slot" else hidden_states
        updated = state + self.up(torch.nn.functional.silu(self.down(self.norm(state))))
        if self.scope == "answer-slot":
            return torch.cat((hidden_states[:, :-1, :], updated), dim=1)
        return updated


class LinearLoopCorrection(nn.Module):
    """Zero-initialized answer-slot correction W h, applied inside every update."""
    def __init__(self, config):
        super().__init__()
        self.projection = ZeroInitLinear(config.hidden_size, config.hidden_size, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, hidden_states):
        return self.projection(hidden_states[:, -1:, :])


class Qwen3LoopForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "loop_transition_rank", 0) > 0:
            self.loop_transition = (LinearLoopCorrection(config)
                if getattr(config, "loop_transition_type", "mlp") == "linear-delta"
                else LoopTransition(config))
            # Register before from_pretrained applies FSDP or loads checkpoint tensors.
            self.post_init()

    def _init_weights(self, module):
        if isinstance(module, ZeroInitLinear):
            init.zeros_(module.weight)
        else:
            super()._init_weights(module)


def load_loop_model(path, *, config=None, **kwargs):
    """Load adapter tensors with their owning class (AutoModel alone drops them)."""
    if config is None:
        config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    model_class = AutoModelForCausalLM
    if getattr(config, "loop_transition_rank", 0) > 0:
        if config.model_type != "qwen3":
            raise ValueError("Loop transitions currently support Qwen3 only")
        model_class = Qwen3LoopForCausalLM
    return model_class.from_pretrained(path, config=config, **kwargs)
