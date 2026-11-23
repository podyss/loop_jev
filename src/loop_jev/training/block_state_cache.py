"""Cache frozen teacher answer states or loop deltas for distillation."""

import argparse
import hashlib
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


def cache_metadata(*, train, model, teacher, layer, seed, rounds, damping_step):
    return dict(train_sha256=hashlib.sha256(Path(train).read_bytes()).hexdigest(),
                model=model, teacher=str(Path(teacher).resolve()) if Path(teacher).exists() else teacher,
                layer=layer, seed=seed, rounds=rounds, damping_step=damping_step)


def capture_answer(model, layer):
    state = {}

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        state["answer"] = hidden[:, -1, :].clone()

    return state, model.model.layers[layer - 1].register_forward_hook(capture)


def state_loss(student, base, target, projection, scale):
    with torch.autocast(device_type=student.device.type, enabled=False):
        state = student.float() if base is None else student.float() - base.float()
        predicted = state @ projection
        return F.smooth_l1_loss(predicted / scale, target.float() / scale)


def load_cache(path, metadata, selected_ids, target_rounds=None, *, target_mode="delta"):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if cache["metadata"] != metadata or set(cache["ids"]) != set(selected_ids):
        raise ValueError("Block-state cache does not match the teacher, base model or training subset")
    if len(cache["ids"]) != len(selected_ids):
        raise ValueError("Block-state cache contains duplicate IDs")
    if cache.get("target_rounds") != target_rounds:
        raise ValueError("Block-state cache does not match the requested teacher rounds")
    if cache.get("target_mode", "delta") != target_mode:
        raise ValueError("Block-state cache does not match the requested target mode")
    base, targets, projection = (cache[key] for key in ("base", "targets", "projection"))
    target_shape = ((len(selected_ids), len(target_rounds), projection.shape[1])
                    if target_rounds else (len(selected_ids), projection.shape[1]))
    valid_base = (base is None if target_mode == "absolute" else
                  base is not None and base.shape == (len(selected_ids), projection.shape[0])
                  and base.isfinite().all())
    if (not valid_base or targets.shape != target_shape
            or not all(t.isfinite().all() for t in (targets, projection))
            or not 0 < cache["scale"] < float("inf")):
        raise ValueError("Invalid block-state cache tensors")
    if base is None:
        base = [None] * len(selected_ids)
    return cache, dict(zip(cache["ids"], zip(base, targets, strict=True), strict=True))


def shuffle_targets_within_label(cached_states, labels, seed):
    if cached_states.keys() != labels.keys():
        raise ValueError("Block-state labels do not match the cached IDs")
    groups = {}
    for item_id, label in sorted(labels.items()):
        groups.setdefault(label, []).append(item_id)

    rng = random.Random(seed)
    shuffled_states = {}
    for label, ids in groups.items():
        if len(ids) < 2:
            raise ValueError(f"Cannot shuffle block-state targets for singleton label {label}")
        rng.shuffle(ids)
        for item_id, donor_id in zip(ids, ids[1:] + ids[:1], strict=True):
            shuffled_states[item_id] = (cached_states[item_id][0], cached_states[donor_id][1])
    return shuffled_states


@torch.inference_mode()
def extract(model, tokenizer, rows, batch_size, device, *, teacher_layer=None, base_layer=None,
            target_rounds=None, target_mode="delta"):
    from tqdm.auto import tqdm

    from loop_jev.evaluation.zero_shot import render_prompt

    cfg = model.config
    capture, handle = capture_answer(model, base_layer) if base_layer else ({}, None)
    anchors = target_rounds or [getattr(cfg, "loop_count", 1)]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def observe(round_number, state, _read_suffix):
        if target_mode == "absolute":
            if round_number == anchors[-1]:
                capture[round_number] = state[:, -1, :].float().clone()
        elif round_number == 1:
            capture["early"] = state[:, -1, :].float().clone()
        elif round_number in anchors:
            capture[round_number] = state[:, -1, :].float() - capture["early"]

    output = []
    try:
        for offset in tqdm(range(0, len(rows), batch_size), desc="Teacher targets" if teacher_layer else "Base states"):
            prompts = [render_prompt(tokenizer, row, cfg.latent_slots, cfg.latent_marker_token_id)
                       if teacher_layer else render_prompt(tokenizer, row)
                       for row in rows[offset:offset + batch_size]]
            width = max(map(len, prompts))
            ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=device)
            mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
            kwargs = {"block_state_observer": observe} if teacher_layer else {}
            model.model(input_ids=ids, attention_mask=mask, use_cache=False, **kwargs)
            if teacher_layer:
                capture.pop("early", None)
                states = [capture.pop(round_number) for round_number in anchors]
                output.append((torch.stack(states, dim=1) if target_rounds else states[0]).cpu())
            else:
                output.append(capture.pop("answer").cpu())
    finally:
        if handle is not None:
            handle.remove()
    shape = (0, len(anchors), cfg.hidden_size) if target_rounds else (0, cfg.hidden_size)
    return torch.cat(output) if output else torch.empty(shape)


def valid_target_rounds(rounds, final):
    return bool(rounds) and 1 < rounds[0] and rounds[-1] == final and sorted(set(rounds)) == rounds


def main():
    from transformers import AutoTokenizer

    from loop_jev.evaluation.zero_shot import read_rows
    from loop_jev.looping import enable_configured_looping
    from loop_jev.training.latent_kv import marker_token_id
    from loop_jev.transition import load_loop_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-sample-seed", type=int, default=42)
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--target-rounds", type=int, nargs="+",
                        help="Increasing teacher rounds ending at the final round; all use round 1 as the anchor")
    parser.add_argument("--target-mode", choices=("delta", "absolute"), default="delta")
    args = parser.parse_args()
    if args.target_mode == "absolute" and args.target_rounds:
        parser.error("Absolute targets support only the final teacher round")
    if min(args.batch_size, args.max_train_examples) < 1:
        parser.error("Batch size and sample count must be positive")
    rows = read_rows(args.train)
    if len(rows) > args.max_train_examples:
        indices = random.Random(args.train_sample_seed).sample(range(len(rows)), args.max_train_examples)
        rows = [rows[i] for i in indices]
    if not rows or len({row["id"] for row in rows}) != len(rows) or any(row.get("split") != "train" for row in rows):
        parser.error("Expected unique training rows")

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    shard = rows[rank::world]
    load_options = dict(dtype=torch.bfloat16, device_map={"": device},
                        attn_implementation="sdpa", trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    teacher = load_loop_model(args.teacher, **load_options).eval()
    cfg = teacher.config
    if (getattr(cfg, "loop_schedule", None) != "block" or cfg.loop_count < 2 or not cfg.loop_damping
            or not cfg.latent_slots or cfg.latent_marker_token_id != marker_token_id(tokenizer)
            or any(getattr(cfg, key, None) for key in (
                "loop_transition_rank", "loop_carry_workspace", "loop_multiscale_workspace",
                "loop_staged_handoff", "loop_evidence_reread", "loop_refinement_token_ids",
                "latent_core_width", "option_workspace"))):
        parser.error("Expected a plain damped block teacher with a matching workspace marker")
    layer = cfg.loop_end_layer
    if args.target_rounds and not valid_target_rounds(args.target_rounds, cfg.loop_count):
        parser.error("Target rounds must be unique, increasing, above 1, and end at the final teacher round")
    metadata = cache_metadata(train=args.train, model=args.model, teacher=args.teacher, layer=layer,
                              seed=args.train_sample_seed, rounds=cfg.loop_count, damping_step=cfg.loop_damping_step)
    enable_configured_looping(teacher)
    teacher_states = extract(teacher, tokenizer, shard, args.batch_size, device, teacher_layer=layer,
                             target_rounds=args.target_rounds, target_mode=args.target_mode)
    # The patched forward closes over the model. Restore it to release the teacher before loading the base.
    teacher.model.forward = teacher.model._unlooped_forward
    del teacher
    torch.cuda.empty_cache()
    model = load_loop_model(args.model, **load_options).eval()
    base = extract(model, tokenizer, shard, args.batch_size, device, base_layer=layer)
    if base.shape != (teacher_states.shape[0], teacher_states.shape[-1]):
        raise ValueError("Teacher and base answer states must have matching width")

    generator = torch.Generator().manual_seed(43)
    width = base.shape[1]
    projection = (torch.randint(0, 2, (width, min(256, width)), generator=generator).float() * 2 - 1) / width**0.5
    # Gather plain lists for Gloo, so deserialization never depends on a worker's device.
    projected = (torch.stack([stage.contiguous() @ projection for stage in teacher_states.unbind(dim=1)], dim=1)
                 if args.target_rounds else teacher_states @ projection)
    base_values = base.float() @ projection if args.target_mode == "absolute" else base.float()
    local = ([row["id"] for row in shard], base_values.tolist(), projected.tolist())
    if world > 1:
        dist.init_process_group("gloo")
        parts = [None] * world if rank == 0 else None
        dist.gather_object(local, parts, dst=0)
        dist.destroy_process_group()
        if rank != 0:
            return
    else:
        parts = [local]
    ids, base_values, target_values = ([item for part in parts for item in part[i]] for i in range(3))
    targets = torch.tensor(target_values, dtype=torch.float32)
    # Keep the final-stage scale for all epochs, matching the fixed-target control.
    final_targets = targets[:, -1, :].contiguous() if args.target_rounds else targets
    base = torch.tensor(base_values, dtype=torch.float32)
    # Normalize both modes by the initial error against the untrained student's state.
    initial_error = final_targets - base if args.target_mode == "absolute" else final_targets
    scale = initial_error.square().mean().sqrt().item()
    if not 0 < scale < float("inf"):
        raise ValueError("Block-state targets have invalid initial-error RMS")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(dict(metadata=metadata, ids=ids, target_rounds=args.target_rounds, target_mode=args.target_mode,
                    base=None if args.target_mode == "absolute" else base.to(torch.bfloat16),
                    targets=targets, projection=projection, scale=scale), temporary)
    temporary.replace(args.output)
    print(f"Saved {len(ids)} {args.target_mode} targets to {args.output}; initial-error RMS={scale:.5f}")


if __name__ == "__main__":
    main()
