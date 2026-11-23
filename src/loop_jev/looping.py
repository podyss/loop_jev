"""Shared-layer looping for Qwen3 decoder models."""

from contextlib import contextmanager
from types import MethodType


@contextmanager
def without_looping(model):
    """Temporarily run the normal Qwen3 forward with the same model weights."""
    backbone = model.model
    if ("Qwen3Model" not in type(backbone).__name__
            or getattr(model.config, "loop_count", 1) < 2
            or not hasattr(backbone, "_unlooped_forward")):
        raise ValueError("without_looping requires a loop-enabled Qwen3 model")
    looped_forward = backbone.forward
    backbone.forward = backbone._unlooped_forward
    try:
        yield
    finally:
        backbone.forward = looped_forward


def damped_update(previous, block_output, loop_count, step=None):
    delta = block_output - previous
    return previous + (delta / loop_count if step is None else step * delta)


def enable_looping(model, start_layer, end_layer, loop_count, damping=False, collect_all_readouts=True,
                   use_transition=True, collect_updates=False, workspace_slice=None,
                   carry_workspace=None, readout_round=None, schedule="block", bypass_looping=False,
                   multiscale_workspace=None, staged_handoff=None, evidence_reread=None, damping_step=None):
    """Repeat a decoder block or each layer in that block with shared parameters."""
    import torch

    if loop_count < 1:
        raise ValueError("loop_count must be positive")
    if damping_step is not None and (not damping or not 0 < damping_step <= 1):
        raise ValueError("damping_step must be in (0, 1] and requires damping")
    if schedule not in ("block", "layerwise", "layerwise-gated"):
        raise ValueError(f"Unknown loop schedule: {schedule}")
    layerwise = schedule != "block"
    gated = schedule == "layerwise-gated"
    if layerwise and (
        damping or collect_all_readouts or collect_updates or workspace_slice is not None
        or carry_workspace is not None or (use_transition and hasattr(model, "loop_transition"))
    ):
        raise ValueError("Layerwise looping supports only a final readout without block-level extras")
    rounds = loop_count if readout_round is None else readout_round
    if not 1 <= rounds <= loop_count:
        raise ValueError("readout_round must be between 1 and the trained loop count")
    if carry_workspace is not None:
        slots, suffix = carry_workspace
        if slots < 1 or suffix < 1 or loop_count < 2:
            raise ValueError("Workspace carry requires slots, an answer suffix, and multiple loops")
    if multiscale_workspace is not None:
        from loop_jev.multiscale_workspace import SLOT_COUNT, exchange_workspace
        scale_mode, scale_suffix = multiscale_workspace
        if (scale_mode not in ("up", "up-down") or scale_suffix < 1 or layerwise
                or loop_count != 3 or not damping or carry_workspace is not None):
            raise ValueError("Multiscale workspace requires three damped block rounds and an answer suffix")
    if evidence_reread is not None:
        from loop_jev.evidence_reread import evidence_bank, reread_evidence
        evidence_slots, evidence_suffix = evidence_reread
        if (min(evidence_reread) < 1 or layerwise or loop_count != 3 or not damping
                or carry_workspace is not None or staged_handoff is not None):
            raise ValueError("Evidence rereading requires three damped block rounds and a retained workspace")
    backbone = getattr(model, "model", None)
    if backbone is None or "Qwen3Model" not in backbone.__class__.__name__:
        raise TypeError("Looping currently supports Qwen3ForCausalLM models")
    if staged_handoff is not None:
        from loop_jev.staged_handoff import handoff_masks, handoff_state
        if (min(staged_handoff) < 1 or layerwise or loop_count != 3 or rounds < 2 or not damping
                or collect_all_readouts or collect_updates or workspace_slice is not None
                or carry_workspace is not None or multiscale_workspace is not None
                or model.config._attn_implementation != "sdpa" or backbone.has_sliding_layers
                or (use_transition and hasattr(model, "loop_transition"))):
            raise ValueError("Staged handoff requires plain damped block looping with full SDPA attention")
    layer_count = len(backbone.layers)
    if not (0 <= start_layer < end_layer <= layer_count):
        raise ValueError(
            f"Loop range [{start_layer}, {end_layer}) must fit {layer_count} decoder layers"
        )
    if gated and any(
        not hasattr(layer, "loop_gates") or layer.loop_gates.raw.size(0) != loop_count - 1
        for layer in backbone.layers[start_layer:end_layer]
    ):
        raise ValueError("Load a Qwen3 model with matching UpLoop gates before enabling gated looping")

    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.qwen3.modeling_qwen3 import (
        DynamicCache,
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    def looped_forward(
        this, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, use_cache=None, loop_depth=None,
        dependency_input=None, option_workspace_input=None, handoff_option_start=None,
        swap_handoff=False, refine_answer_scores=False, disable_evidence_reread=False,
        block_state_observer=None, block_layer_observation=None, block_round_observer=None,
        block_early_exit=None, block_final_transform=None, **kwargs,
    ):
        if block_final_transform is not None:
            if (this.training or torch.is_grad_enabled() or layerwise or bypass_looping or rounds < 2
                    or not damping or collect_all_readouts or use_cache or past_key_values is not None
                    or block_state_observer is not None or block_layer_observation is not None
                    or block_round_observer is not None or block_early_exit is not None
                    or dependency_input is not None or refine_answer_scores or workspace_slice is not None
                    or carry_workspace is not None or multiscale_workspace is not None or staged_handoff is not None
                    or evidence_reread is not None or (use_transition and hasattr(model, "loop_transition"))
                    or (attention_mask is not None and (not torch.is_tensor(attention_mask) or attention_mask.ndim != 2))):
                raise ValueError("Final transforms require plain damped block inference without a cache")
            block_path = []
        if block_state_observer is not None and (this.training or layerwise or bypass_looping):
            raise ValueError("Block state observation requires block looping in eval mode")
        if block_early_exit is not None:
            exit_round, should_exit = block_early_exit
            if (this.training or torch.is_grad_enabled() or layerwise or bypass_looping or collect_all_readouts
                    or use_cache or past_key_values is not None or not 1 <= exit_round < rounds
                    or block_state_observer is not None or block_layer_observation is not None
                    or block_round_observer is not None or dependency_input is not None or refine_answer_scores
                    or carry_workspace is not None or multiscale_workspace is not None or staged_handoff is not None
                    or evidence_reread is not None or (use_transition and hasattr(model, "loop_transition"))):
                raise ValueError("Early exit requires inference without cache or extensions on a plain loop block")
        if block_round_observer is not None and (
            this.training or torch.is_grad_enabled() or layerwise or bypass_looping or not damping
            or rounds < 2 or carry_workspace is not None or multiscale_workspace is not None
            or evidence_reread is not None or staged_handoff is not None
            or (use_transition and hasattr(model, "loop_transition"))
        ):
            raise ValueError("Round replay requires a plain damped block with gradients disabled")
        observed_layers = ()
        if block_layer_observation is not None:
            observed_layers, observe_layer = block_layer_observation
            if (this.training or layerwise or bypass_looping or rounds < 2 or not damping
                    or (use_transition and hasattr(model, "loop_transition")) or staged_handoff is not None
                    or any(not start_layer < layer <= end_layer for layer in observed_layers)):
                raise ValueError("Layer observation requires a plain damped block in eval mode")
        if refine_answer_scores and (layerwise or rounds < 2 or not damping or collect_all_readouts):
            raise ValueError("Answer refinement requires multiple damped block rounds and a final readout")
        if dependency_input is not None and (not this.training or layerwise or workspace_slice is None or rounds < 2):
            raise ValueError("Dependency training requires a recurrent block and a workspace slice")
        if loop_depth is not None and (not layerwise or not 1 <= loop_depth <= rounds):
            raise ValueError("loop_depth requires layerwise looping and must fit the configured depth")
        if loop_count > 1:
            use_cache = False
            past_key_values = None
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = this.embed_tokens(input_ids)
        if dependency_input is not None:
            reader_ids, reader_mask, probe = dependency_input
            reader_embeds = this.embed_tokens(reader_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=this.config)
        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (
                __import__("torch").arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + seen
            ).unsqueeze(0)
        if staged_handoff is not None:
            if (handoff_option_start is None or attention_mask is None or attention_mask.ndim != 2
                    or (swap_handoff and this.training)):
                raise ValueError("Staged handoff needs option offsets and a padding mask; swapping is eval-only")
            write_mask, read_mask = handoff_masks(attention_mask, handoff_option_start, *staged_handoff)
            masks = {"full_attention": write_mask}
        elif handoff_option_start is not None or swap_handoff:
            raise ValueError("Handoff arguments require a staged handoff checkpoint")
        elif not isinstance(attention_mask, dict):
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

        def run_layer(layer_index, state, **layer_kwargs):
            return layers[layer_index](
                state,
                attention_mask=masks[this.config.layer_types[layer_index]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
                **layer_kwargs,
            )

        hidden_states = inputs_embeds
        for index in range(start_layer):
            hidden_states = run_layer(index, hidden_states)

        if option_workspace_input is not None:
            from loop_jev.option_workspace import seed_option_slots
            spans, offsets = option_workspace_input
            hidden_states = seed_option_slots(hidden_states, spans, offsets, this.config.option_workspace_suffix)

        if layerwise:
            for index in range(start_layer, end_layer):
                hidden_states = run_layer(index, hidden_states)
                for local_round in range(1, rounds if loop_depth is None else loop_depth):
                    if gated:
                        hidden_states = run_layer(index, hidden_states, local_round=local_round)
                    else:
                        hidden_states = run_layer(index, hidden_states)
            for index in range(end_layer, len(layers)):
                hidden_states = run_layer(index, hidden_states)
            hidden_states = this.norm(hidden_states)
            this.loop_hidden_states = [hidden_states]
            this.loop_updates = []
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            )

        loop_readouts = []
        loop_updates = []
        workspace_states = []
        linear_delta = getattr(model.config, "loop_transition_type", "mlp") == "linear-delta"

        def transition(state):
            if this.training and model.is_gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(model.loop_transition, state, use_reentrant=False)
            return model.loop_transition(state)

        loop_input = hidden_states
        use_evidence = evidence_reread is not None and not disable_evidence_reread and rounds > 1
        if use_evidence:
            evidence_end = hidden_states.shape[1] - evidence_suffix
            prefix_end = evidence_end - evidence_slots
            if prefix_end < 1 or (attention_mask is not None and attention_mask.ndim != 2):
                raise ValueError("Evidence rereading needs a nonempty prefix and a 2D padding mask")
            evidence_valid = (attention_mask[:, :prefix_end].bool() if attention_mask is not None else
                              hidden_states.new_ones(hidden_states.shape[0], prefix_end, dtype=torch.bool))
            keys, values = evidence_bank(hidden_states[:, :prefix_end], evidence_valid)
            coverage = None
        if carry_workspace is not None and hidden_states.shape[1] < slots + suffix:
            raise ValueError("Input is too short for the configured workspace and answer suffix")
        if multiscale_workspace is not None and hidden_states.shape[1] < SLOT_COUNT + scale_suffix:
            raise ValueError("Input is too short for the multiscale workspace and answer suffix")
        def read_suffix(state):
            for index in range(end_layer, len(layers)):
                state = run_layer(index, state)
            return state

        def evaluate_block(states):
            if loop_input.shape[0] != 1 or states.shape[1:] != loop_input.shape[1:]:
                raise ValueError("Parallel time probes require a single input and matching state shapes")
            for index in range(start_layer, end_layer):
                states = run_layer(index, states)
            return states

        def replay_rounds(state, remaining):
            for _ in range(remaining):
                output = state
                for index in range(start_layer, end_layer):
                    output = run_layer(index, output)
                state = damped_update(state, output, loop_count, damping_step)
            return read_suffix(state)

        for loop_index in range(rounds):
            previous = loop_input
            if block_final_transform is not None:
                block_path.append(previous)
            block_output = loop_input
            observing = bool(observed_layers) and loop_index in (0, rounds - 1)
            for index in range(start_layer, end_layer):
                block_output = run_layer(index, block_output)
                layer = index + 1
                if observing and layer < end_layer and layer in observed_layers:
                    def replay(state, next_layer=layer, round_input=previous):
                        for remaining in range(next_layer, end_layer):
                            state = run_layer(remaining, state)
                        return read_suffix(damped_update(round_input, state, loop_count, damping_step))

                    observe_layer(loop_index + 1, layer, block_output, replay)
            loop_input = (
                damped_update(loop_input, block_output, loop_count, damping_step)
                if damping and not bypass_looping else block_output
            )

            if use_transition and linear_delta and hasattr(model, "loop_transition"):
                correction = transition(previous)
                loop_input = __import__("torch").cat(
                    (loop_input[:, :-1, :], loop_input[:, -1:, :] + correction), dim=1)
            if collect_updates:
                loop_updates.append(loop_input[:, -1, :].float() - previous[:, -1, :].float())
            if workspace_slice is not None:
                workspace_states.append(loop_input[:, workspace_slice, :].clone())

            if block_state_observer is not None:
                block_state_observer(loop_index + 1, loop_input, read_suffix)
            if block_round_observer is not None:
                def replay_update(gate, before=previous, raw=block_output, remaining=rounds - loop_index - 1):
                    step = (1 / loop_count if damping_step is None else damping_step) * gate
                    return replay_rounds(damped_update(before, raw, loop_count, step), remaining)

                block_round_observer(loop_index + 1, previous, loop_input, read_suffix, replay_update)
            if observing and end_layer in observed_layers:
                observe_layer(loop_index + 1, end_layer, loop_input, read_suffix)
            if refine_answer_scores and loop_index == 0:
                # Use the undamped output so the anchor matches a standard single pass.
                first = read_suffix(block_output)
                loop_readouts.append(this.norm(first[:, -1:, :]))
            if block_early_exit is not None and loop_index + 1 == exit_round:
                exit_state = this.norm(read_suffix(loop_input))
                if should_exit(exit_state):
                    loop_readouts.append(exit_state)
                    break
            if collect_all_readouts or loop_index + 1 == rounds:
                state = (loop_input if block_final_transform is None else
                         block_final_transform((*block_path, loop_input), attention_mask, evaluate_block))
                readout = read_suffix(state)
                if refine_answer_scores:
                    readout = readout[:, -1:, :]
                loop_readouts.append(this.norm(readout))
            # The final readout precedes the transition: K loops use K-1 updates.
            if use_transition and not linear_delta and loop_index + 1 < rounds and hasattr(model, "loop_transition"):
                if this.training and model.is_gradient_checkpointing:
                    from torch.utils.checkpoint import checkpoint
                    loop_input = checkpoint(model.loop_transition, loop_input, use_reentrant=False)
                else:
                    loop_input = model.loop_transition(loop_input)
            if carry_workspace is not None and loop_index + 1 < rounds:
                loop_input = torch.cat((hidden_states[:, :-(slots + suffix), :],
                                       loop_input[:, -(slots + suffix):-suffix, :],
                                       hidden_states[:, -suffix:, :]), dim=1)
            if multiscale_workspace is not None and loop_index + 1 < rounds:
                if loop_index == 0 or scale_mode == "up-down":
                    loop_input = exchange_workspace(loop_input, previous, scale_suffix,
                                                    upward=loop_index == 0)
            if use_evidence and loop_index + 1 < rounds:
                loop_input, coverage = reread_evidence(loop_input, previous, evidence_end, evidence_slots,
                                                        keys, values, evidence_valid, coverage)
            if staged_handoff is not None and loop_index == 0:
                loop_input = handoff_state(hidden_states, loop_input, *staged_handoff, swap=swap_handoff)
                masks = {"full_attention": read_mask}

        this.loop_updates = loop_updates
        this.loop_hidden_states = loop_readouts
        if workspace_slice is not None:
            this.loop_workspace_states = workspace_states
        if dependency_input is not None:
            workspace = workspace_states[0]
            reader_embeds = reader_embeds.to(workspace.dtype)
            batch_size, slot_count, _ = workspace.shape
            state = torch.cat((workspace, reader_embeds), dim=1)
            mask = torch.cat((reader_mask.new_ones(batch_size, slot_count), reader_mask), dim=1)
            read_index = slot_count + reader_mask.sum(dim=1) - 1
            if probe:
                # Same reader tokens and position indices; only remove the incoming state.
                blank = torch.cat((torch.zeros_like(workspace), reader_embeds.detach()), dim=1)
                state = torch.cat((state, blank), dim=0)
                mask = mask.repeat(2, 1)
                read_index = read_index.repeat(2)
            position_ids = torch.arange(state.shape[1], device=state.device).unsqueeze(0)
            reader_mask_kwargs = dict(config=this.config, inputs_embeds=state, attention_mask=mask,
                                      past_key_values=None, position_ids=position_ids)
            masks = {"full_attention": create_causal_mask(**reader_mask_kwargs)}
            if this.has_sliding_layers:
                masks["sliding_attention"] = create_sliding_window_causal_mask(**reader_mask_kwargs)
            position_embeddings = this.rotary_emb(state, position_ids)
            reader = state
            for index in range(start_layer, end_layer):
                reader = run_layer(index, reader)
            if damping:
                reader = damped_update(state, reader, loop_count, damping_step)
            reader = reader[torch.arange(reader.shape[0], device=reader.device), read_index]
            this.loop_dependency_states = (workspace[:, -1], reader[:batch_size],
                                           reader[batch_size:].detach() if probe else None)
        hidden_states = loop_readouts[-1]
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    if not hasattr(backbone, "_unlooped_forward"):
        backbone._unlooped_forward = backbone.forward
    backbone.forward = MethodType(looped_forward, backbone)
    model.config.loop_start_layer = start_layer
    model.config.loop_end_layer = end_layer
    model.config.loop_count = loop_count
    model.config.loop_schedule = schedule
    model.config.loop_damping = bool(damping)
    model.config.loop_damping_step = damping_step
    model.config.loop_carry_workspace = carry_workspace is not None
    if carry_workspace is not None:
        model.config.loop_workspace_suffix = suffix
    if multiscale_workspace is not None:
        model.config.loop_multiscale_workspace = scale_mode
        model.config.loop_multiscale_suffix = scale_suffix
    if staged_handoff is not None:
        model.config.loop_staged_handoff = True
        model.config.loop_handoff_suffix = staged_handoff[1]
    model.config.loop_evidence_reread = evidence_reread is not None
    if evidence_reread is not None:
        model.config.loop_evidence_suffix = evidence_suffix
    return model


def enable_configured_looping(model, *, use_transition=True, readout_round=None, bypass_looping=False,
                              reset_latent_core=False):
    config = model.config
    if getattr(config, "latent_core_width", 0):
        if not bypass_looping:
            from loop_jev.latent_core import enable_latent_core
            enable_latent_core(model, rounds=readout_round, reset_last=reset_latent_core)
        return model
    if getattr(config, "loop_count", 1) > 1 or getattr(config, "option_workspace", False):
        enable_looping(
            model,
            config.loop_start_layer,
            config.loop_end_layer,
            config.loop_count,
            getattr(config, "loop_damping", False),
            damping_step=getattr(config, "loop_damping_step", None),
            collect_all_readouts=False,
            use_transition=use_transition,
            carry_workspace=(config.latent_slots, config.loop_workspace_suffix)
            if getattr(config, "loop_carry_workspace", False) else None,
            readout_round=1 if bypass_looping else readout_round,
            schedule=getattr(config, "loop_schedule", "block"),
            bypass_looping=bypass_looping,
            multiscale_workspace=(config.loop_multiscale_workspace, config.loop_multiscale_suffix)
            if getattr(config, "loop_multiscale_workspace", None) else None,
            staged_handoff=(config.latent_slots, config.loop_handoff_suffix)
            if getattr(config, "loop_staged_handoff", False) else None,
            evidence_reread=(config.latent_slots, config.loop_evidence_suffix)
            if getattr(config, "loop_evidence_reread", False) else None,
        )
    return model
