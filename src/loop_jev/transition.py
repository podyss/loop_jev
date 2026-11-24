"""Qwen3 with a checkpointed, shared residual transition between loop passes."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3ForCausalLM
from transformers import initialization as init
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import slice_logits_to_keep
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from loop_jev.up_loop import UpLoopQwen3DecoderLayer
from loop_jev.option_workspace import SLOT_COUNT, SLOTS_PER_OPTION


@dataclass
class OptionWorkspaceOutput(CausalLMOutputWithPast):
    answer_token_logits: torch.Tensor | None = None


@dataclass
class DualReadoutOutput(CausalLMOutputWithPast):
    mix: torch.Tensor | None = None


class ZeroInitLinear(nn.Linear):
    """Marker for HF initialization of the residual output projection."""


class DualReadoutMixHead(nn.Module):
    def __init__(self, hidden_size, initial_mix):
        super().__init__()
        self.initial_mix = initial_mix
        self.input_projection = nn.Linear(2 * hidden_size, 64)
        self.output_projection = ZeroInitLinear(64, 1, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, readout_states):
        features = torch.nn.functional.gelu(self.input_projection(readout_states.flatten(-2)))
        log_multiplier = self.output_projection(features).float()
        return self.initial_mix * (2 * log_multiplier.tanh()).exp()


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


class DirectAnswerHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        width = 64
        self.query = nn.Linear(hidden_size, width, bias=False)
        self.key = nn.Linear(hidden_size, width, bias=False)
        self.scale = width ** -0.5

    def forward(self, hidden, option_positions):
        indices = option_positions.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        options = hidden.gather(1, indices)
        query = self.query(hidden[:, -1]).float()
        keys = self.key(options).float()
        scores = (query[:, None] * keys).sum(dim=-1) * self.scale
        return scores.masked_fill(option_positions < 0, float("-inf"))


class Qwen3LoopForCausalLM(Qwen3ForCausalLM):
    _no_split_modules = [*Qwen3ForCausalLM._no_split_modules, "UpLoopQwen3DecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "loop_schedule", "block") == "layerwise-gated":
            if not (0 <= config.loop_start_layer < config.loop_end_layer <= len(self.model.layers)
                    and config.loop_count >= 2):
                raise ValueError("Invalid span or round count for gated looping")
            for index in range(config.loop_start_layer, config.loop_end_layer):
                self.model.layers[index] = UpLoopQwen3DecoderLayer(config, index)
        if getattr(config, "loop_transition_rank", 0) > 0:
            self.loop_transition = (LinearLoopCorrection(config)
                if getattr(config, "loop_transition_type", "mlp") == "linear-delta"
                else LoopTransition(config))
        option_workspace = getattr(config, "option_workspace", False)
        if option_workspace:
            self.option_slot_offsets = nn.Embedding(SLOTS_PER_OPTION, config.hidden_size)
            self.option_scorer = nn.Linear(config.hidden_size, 1, bias=False)
        refinement_ids = getattr(config, "loop_refinement_token_ids", None)
        if refinement_ids:
            self.loop_answer_refinement = ZeroInitLinear(config.hidden_size, len(refinement_ids), bias=False)
            nn.init.zeros_(self.loop_answer_refinement.weight)
        core_width = getattr(config, "latent_core_width", 0)
        if core_width:
            from loop_jev.latent_core import LatentCore
            self.latent_core = LatentCore(config.hidden_size, core_width, ZeroInitLinear)
        direct_answer_head = getattr(config, "direct_answer_head", False)
        if direct_answer_head == "shared":
            self.direct_answer_head = DirectAnswerHead(config.hidden_size)
        elif direct_answer_head:
            self.direct_answer_head = nn.Linear(config.hidden_size, 4)
        mix_initial = getattr(config, "dual_readout_mix_head_initial", None)
        if mix_initial is not None:
            self.dual_readout_mix_head = DualReadoutMixHead(config.hidden_size, mix_initial)
        if (getattr(config, "loop_schedule", "block") == "layerwise-gated"
                or getattr(config, "loop_transition_rank", 0) > 0 or option_workspace or refinement_ids
                or core_width or direct_answer_head or mix_initial is not None):
            # Register before from_pretrained applies FSDP or loads checkpoint tensors.
            self.post_init()

    def forward(self, *args, option_spans=None, option_vocab_logits=False, refine_answer_scores=False,
                direct_answer_scores=False, direct_answer_positions=None, mix_head_positions=None, **kwargs):
        if mix_head_positions is not None:
            logits_to_keep = kwargs.pop("logits_to_keep", 0)
            hidden = self.model(*args, **kwargs).last_hidden_state
            mix_states = slice_logits_to_keep(hidden, mix_head_positions)
            logits = self.lm_head(slice_logits_to_keep(hidden, logits_to_keep))
            return DualReadoutOutput(logits=logits, mix=self.dual_readout_mix_head(mix_states))
        if direct_answer_scores:
            kwargs.pop("logits_to_keep", None)
            outputs = self.model(*args, **kwargs)
            hidden = outputs.last_hidden_state
            state = hidden[:, -1:, :]
            if self.config.direct_answer_head == "shared":
                if direct_answer_positions is None:
                    raise ValueError("The shared answer head requires the option end positions")
                scores = self.direct_answer_head(hidden, direct_answer_positions)
            else:
                scores = self.direct_answer_head(state[:, 0, :])
            return OptionWorkspaceOutput(
                logits=scores,
                answer_token_logits=self.lm_head(state) if option_vocab_logits else None,
            )
        if refine_answer_scores:
            if not hasattr(self, "loop_answer_refinement") or not hasattr(self.model, "_unlooped_forward"):
                raise ValueError("Answer refinement needs an enabled refinement checkpoint")
            kwargs.pop("logits_to_keep", None)
            kwargs.pop("output_hidden_states", None)
            kwargs["use_cache"] = False
            outputs = self.model(*args, **kwargs, refine_answer_scores=True)
            first = self.model.loop_hidden_states[0]
            base_logits = self.lm_head(first)
            delta = self.loop_answer_refinement(outputs.last_hidden_state - first.detach())
            base_options = base_logits[:, -1, self.config.loop_refinement_token_ids].float()
            return OptionWorkspaceOutput(logits=base_options.detach() + delta[:, -1, :].float(),
                                         answer_token_logits=base_logits)
        if not getattr(self.config, "option_workspace", False):
            return super().forward(*args, **kwargs)
        if option_spans is None or not hasattr(self.model, "_unlooped_forward"):
            raise ValueError("Option workspace requires option spans and an enabled block forward")
        offsets = self.option_slot_offsets(torch.arange(SLOTS_PER_OPTION, device=option_spans.device))
        kwargs.pop("logits_to_keep", None)
        kwargs.pop("output_hidden_states", None)
        kwargs["use_cache"] = False
        outputs = self.model(*args, **kwargs, option_workspace_input=(option_spans, offsets))
        hidden = outputs.last_hidden_state
        suffix = self.config.option_workspace_suffix
        states = hidden[:, -SLOT_COUNT - suffix:-suffix][:, SLOTS_PER_OPTION - 1::SLOTS_PER_OPTION]
        logits = self.option_scorer(states).squeeze(-1)
        vocabulary = self.lm_head(hidden[:, -1:, :]) if option_vocab_logits else None
        return OptionWorkspaceOutput(logits=logits, answer_token_logits=vocabulary)

    def _init_weights(self, module):
        if isinstance(module, ZeroInitLinear):
            init.zeros_(module.weight)
        else:
            super()._init_weights(module)


def load_loop_model(path, *, config=None, **kwargs):
    """Load adapter tensors with their owning class (AutoModel alone drops them)."""
    if config is None:
        config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    gated = getattr(config, "loop_schedule", "block") == "layerwise-gated"
    model_class = AutoModelForCausalLM
    if (getattr(config, "loop_transition_rank", 0) > 0 or gated or getattr(config, "option_workspace", False)
            or getattr(config, "loop_refinement_token_ids", None) or getattr(config, "latent_core_width", 0)
            or getattr(config, "direct_answer_head", False)
            or getattr(config, "dual_readout_mix_head_initial", None) is not None):
        if config.model_type != "qwen3":
            raise ValueError("Loop extensions currently support Qwen3 only")
        model_class = Qwen3LoopForCausalLM
    if gated:
        return_loading_info = kwargs.pop("output_loading_info", False)
        model, info = model_class.from_pretrained(path, config=config, output_loading_info=True, **kwargs)
        # Missing weights inside the nested HF backbone bypass the outer initializer.
        with torch.no_grad():
            for key in info["missing_keys"]:
                if key.endswith(".loop_gates.raw"):
                    model.get_parameter(key).zero_()
        return (model, info) if return_loading_info else model
    return model_class.from_pretrained(path, config=config, **kwargs)
