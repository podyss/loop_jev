"""Supervised removal of recurrent computation from the block10 checkpoint."""

import torch

from loop_jev.looping import enable_looping


def removal_schedule(mode, steps_per_epoch, *, final_rounds=1, keep_slots=False):
    if final_rounds not in (1, 2, 3) or (final_rounds > 1 and not keep_slots):
        raise ValueError("Expected one to three final rounds; short loops keep the workspace")
    final = (final_rounds, 24 if keep_slots else 0)
    if mode == "abrupt":
        return ((0, *final),)
    if mode != "gradual" or steps_per_epoch < 4:
        raise ValueError("Gradual loop removal needs at least four updates per epoch")
    stages = [(0, 10, 24), (steps_per_epoch // 4, 5, 24 if keep_slots else 12)]
    if final_rounds == 1:
        stages.append((steps_per_epoch // 2, 2, 24 if keep_slots else 4))
    stages.append(((3 if final_rounds == 1 else 2) * steps_per_epoch // 4, *final))
    return tuple(stages)


def validate_checkpoint(config, marker_id):
    expected = dict(loop_schedule="block", loop_start_layer=9, loop_end_layer=18,
                    loop_count=10, loop_damping=True, loop_damping_step=0.1,
                    latent_slots=24, latent_marker_token_id=marker_id)
    extras = ("loop_transition_rank", "loop_carry_workspace", "loop_multiscale_workspace",
              "loop_staged_handoff", "loop_evidence_reread", "loop_refinement_token_ids",
              "latent_core_width", "option_workspace")
    if (any(getattr(config, key, None) != value for key, value in expected.items())
            or any(getattr(config, key, None) for key in extras)):
        raise ValueError("Loop removal expects the plain damped block10 checkpoint with 24 slots")


def configure_stage(model, rounds, slots, marker_id, *, start_layer=9, end_layer=18, damping_step=None):
    cfg = model.config
    if rounds > 1:
        # Keep the nominal integration horizon fixed as the number of steps shrinks.
        enable_looping(model, start_layer, end_layer, rounds, damping=True,
                       damping_step=1 / rounds if damping_step is None else damping_step,
                       collect_all_readouts=False)
    else:
        backbone = model.model
        if hasattr(backbone, "_unlooped_forward"):
            backbone.forward = backbone._unlooped_forward
        cfg.loop_count = 1
        cfg.loop_damping = False
        cfg.loop_damping_step = None
    cfg.latent_slots = slots
    cfg.latent_marker_token_id = marker_id if slots else None


def trim_workspace(batch, slots, suffix_length, original_slots=24):
    if slots == original_slots:
        return batch
    # Left padding makes the workspace span align across prompts of different lengths.
    offset = suffix_length + original_slots - slots
    return {key: torch.cat((value[:, :-offset], value[:, -suffix_length:]), dim=1)
            for key, value in batch.items()}
