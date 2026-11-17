"""Shared-layer looping for Qwen3 decoder models."""

from types import MethodType


def enable_looping(model, start_layer, end_layer, loop_count, damping=False, collect_all_readouts=True,
                   use_transition=True, collect_updates=False):
    """Repeat decoder block [start_layer, end_layer) with shared parameters."""
    if loop_count < 1:
        raise ValueError("loop_count must be positive")
    backbone = getattr(model, "model", None)
    if backbone is None or "Qwen3Model" not in backbone.__class__.__name__:
        raise TypeError("Looping currently supports Qwen3ForCausalLM models")
    layer_count = len(backbone.layers)
    if not (0 <= start_layer < end_layer <= layer_count):
        raise ValueError(
            f"Loop range [{start_layer}, {end_layer}) must fit {layer_count} decoder layers"
        )

    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.qwen3.modeling_qwen3 import (
        DynamicCache,
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    def looped_forward(
        this, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs,
    ):
        if loop_count > 1:
            use_cache = False
            past_key_values = None
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = this.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=this.config)
        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (
                __import__("torch").arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + seen
            ).unsqueeze(0)
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": this.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            masks = {"full_attention": create_causal_mask(**mask_kwargs)}
            if this.has_sliding_layers:
                masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            masks = attention_mask
        hidden_states = inputs_embeds
        position_embeddings = this.rotary_emb(hidden_states, position_ids)
        layers = this.layers[: this.config.num_hidden_layers]

        def run_layer(layer_index, state):
            return layers[layer_index](
                state,
                attention_mask=masks[this.config.layer_types[layer_index]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = inputs_embeds
        for index in range(start_layer):
            hidden_states = run_layer(index, hidden_states)

        loop_readouts = []
        loop_updates = []
        linear_delta = getattr(model.config, "loop_transition_type", "mlp") == "linear-delta"

        def transition(state):
            if this.training and model.is_gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(model.loop_transition, state, use_reentrant=False)
            return model.loop_transition(state)

        loop_input = hidden_states
        for loop_index in range(loop_count):
            previous = loop_input
            block_output = loop_input
            for index in range(start_layer, end_layer):
                block_output = run_layer(index, block_output)
            loop_input = (
                loop_input + (block_output - loop_input) / loop_count
                if damping else block_output
            )

            if use_transition and linear_delta and hasattr(model, "loop_transition"):
                correction = transition(previous)
                loop_input = __import__("torch").cat(
                    (loop_input[:, :-1, :], loop_input[:, -1:, :] + correction), dim=1)
            if collect_updates:
                loop_updates.append(loop_input[:, -1, :].float() - previous[:, -1, :].float())

            if collect_all_readouts or loop_index + 1 == loop_count:
                readout = loop_input
                for index in range(end_layer, len(layers)):
                    readout = run_layer(index, readout)
                loop_readouts.append(this.norm(readout))
            # The final readout precedes the transition: K loops use K-1 updates.
            if use_transition and not linear_delta and loop_index + 1 < loop_count and hasattr(model, "loop_transition"):
                if this.training and model.is_gradient_checkpointing:
                    from torch.utils.checkpoint import checkpoint
                    loop_input = checkpoint(model.loop_transition, loop_input, use_reentrant=False)
                else:
                    loop_input = model.loop_transition(loop_input)

        this.loop_updates = loop_updates
        this.loop_hidden_states = loop_readouts
        hidden_states = loop_readouts[-1]
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    backbone.forward = MethodType(looped_forward, backbone)
    model.config.loop_start_layer = start_layer
    model.config.loop_end_layer = end_layer
    model.config.loop_count = loop_count
    model.config.loop_damping = bool(damping)
    return model


def enable_configured_looping(model, *, use_transition=True):
    config = model.config
    if getattr(config, "loop_count", 1) > 1:
        enable_looping(
            model,
            config.loop_start_layer,
            config.loop_end_layer,
            config.loop_count,
            getattr(config, "loop_damping", False),
            collect_all_readouts=False,
            use_transition=use_transition,
        )
    return model
