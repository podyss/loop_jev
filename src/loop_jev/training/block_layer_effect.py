"""Cache option-score effects of replacing late answer states with early ones."""

import argparse
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from loop_jev.training.block_state_cache import cache_metadata


def layer_effect_targets(path, metadata, ids, layers, mode):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    cached_ids = cache["ids"]
    effects = cache["effects"]
    if (cache["metadata"] != metadata or cache["layers"] != layers
            or len(cached_ids) != len(ids) or set(cached_ids) != set(ids)
            or effects.shape != (len(ids), len(layers), 4) or not effects.isfinite().all()):
        raise ValueError("Layer effect cache does not match the requested experiment")
    if mode == "shared":
        effects = effects[:, -1:, :].expand_as(effects)
    return dict(zip(cached_ids, effects, strict=True)), cache["scale"]


class LayerEffectReadout(nn.Module):
    def __init__(self, hidden_size, count):
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(hidden_size, 4, bias=False) for _ in range(count))

    def forward(self, states, target, scale):
        predictions = torch.stack([head(F.rms_norm(state.float(), (state.shape[-1],)))
                                   for head, state in zip(self.heads, states, strict=True)], dim=1).float()
        predictions = predictions - predictions.mean(dim=-1, keepdim=True)
        return F.smooth_l1_loss(predictions, target.float() / scale)


@torch.inference_mode()
def extract_effects(model, input_ids, mask, layers, label_ids):
    final, patched = extract_scores(model, input_ids, mask, layers, label_ids)
    effects = final[:, None, :] - patched
    return effects - effects.mean(dim=-1, keepdim=True)


@torch.inference_mode()
def extract_scores(model, input_ids, mask, layers, label_ids):
    early, patched_scores = {}, {}

    def score(normalized):
        return model.lm_head(normalized[:, -1:, :])[:, 0, label_ids].float()

    def observe(round_number, layer, state, replay):
        if round_number == 1:
            early[layer] = state[:, -1, :].clone()
        else:
            patched = state.clone()
            patched[:, -1, :] = early[layer]
            patched_scores[layer] = score(model.model.norm(replay(patched)))

    final = model.model(input_ids=input_ids, attention_mask=mask, use_cache=False,
                        block_layer_observation=(layers, observe)).last_hidden_state
    return score(final), torch.stack([patched_scores[layer] for layer in layers], dim=1)


def main():
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    from loop_jev.evaluation.zero_shot import read_rows, render_prompt
    from loop_jev.looping import enable_configured_looping
    from loop_jev.training.latent_kv import marker_token_id
    from loop_jev.transition import load_loop_model

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "teacher"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("train", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--train-sample-seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_rows(args.train)
    if min(args.batch_size, args.max_train_examples) < 1:
        parser.error("Batch size and sample count must be positive")
    if len(rows) > args.max_train_examples:
        rows = [rows[i] for i in random.Random(args.train_sample_seed).sample(range(len(rows)), args.max_train_examples)]
    if not rows or any(row.get("split") != "train" or len(row["options"]) != 4 for row in rows):
        parser.error("Expected four-option training rows")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    teacher = load_loop_model(args.teacher, dtype=torch.bfloat16, device_map={"": device},
                              attn_implementation="sdpa", trust_remote_code=False).eval()
    cfg = teacher.config
    if (cfg.loop_schedule != "block" or cfg.loop_count < 2 or not cfg.loop_damping
            or not cfg.latent_slots or cfg.latent_marker_token_id != marker_token_id(tokenizer)
            or any(getattr(cfg, name, None) for name in (
                "loop_transition_rank", "loop_carry_workspace", "loop_multiscale_workspace",
                "loop_staged_handoff", "loop_evidence_reread", "loop_refinement_token_ids",
                "latent_core_width", "option_workspace"))):
        parser.error("Expected a plain damped block-loop teacher")
    if (args.layers != sorted(set(args.layers)) or args.layers[0] <= cfg.loop_start_layer
            or args.layers[-1] != cfg.loop_end_layer):
        parser.error("Expected increasing layer exits in the loop block, ending at its output")
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in "ABCD"]
    if any(len(ids) != 1 for ids in labels):
        parser.error("Expected single-token option labels")
    label_ids = [ids[0] for ids in labels]
    enable_configured_looping(teacher)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    shard = rows[rank::world]
    local = []
    for offset in tqdm(range(0, len(shard), args.batch_size), desc=f"Layer effects rank {rank}"):
        batch = shard[offset:offset + args.batch_size]
        prompts = [render_prompt(tokenizer, row, cfg.latent_slots, cfg.latent_marker_token_id) for row in batch]
        width = max(map(len, prompts))
        inputs = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
        effects = extract_effects(teacher, inputs, mask, args.layers, label_ids)
        local.extend(zip((row["id"] for row in batch), effects.tolist(), strict=True))
    if world > 1:
        dist.init_process_group("gloo")
        parts = [None] * world if rank == 0 else None
        dist.gather_object(local, parts, dst=0)
        dist.destroy_process_group()
        if rank != 0:
            return
        local = [item for part in parts for item in part]
    ids, values = zip(*local, strict=True)
    effects = torch.tensor(values, dtype=torch.float32)
    layer_rms = effects.square().mean(dim=(0, 2)).sqrt()
    scale = layer_rms[-1].item()
    if len(set(ids)) != len(rows) or not effects.isfinite().all() or scale <= 0:
        raise ValueError("Teacher effects are nonfinite, zero at the block output, or missing IDs")
    metadata = cache_metadata(train=args.train, model=args.model, teacher=args.teacher, layer=cfg.loop_end_layer,
                              seed=args.train_sample_seed, rounds=cfg.loop_count, damping_step=cfg.loop_damping_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(dict(metadata=metadata, layers=args.layers, ids=list(ids), effects=effects, scale=scale), temporary)
    temporary.replace(args.output)
    print(f"Saved {len(ids)} targets to {args.output}; per-layer RMS={layer_rms.tolist()}")


if __name__ == "__main__":
    main()
