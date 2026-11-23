"""Cache immediate and delayed four-option effects for each teacher update."""

import argparse
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from loop_jev.training.block_state_cache import cache_metadata

GATE = 0.75
MODES = ("immediate", "counterfactual")


def round_effect_targets(path, metadata, ids, mode):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if (cache["metadata"] != metadata or cache["gate"] != GATE
            or len(cache["ids"]) != len(ids) or set(cache["ids"]) != set(ids)):
        raise ValueError("Round effect cache does not match the teacher or training subset")
    targets = cache[mode]
    scale = targets.square().mean().sqrt().item()
    if targets.shape != (len(ids), metadata["rounds"], 4) or not 0 < scale < float("inf"):
        raise ValueError(f"Invalid {mode} round targets")
    return dict(zip(cache["ids"], targets, strict=True)), scale


@torch.inference_mode()
def extract_round_effects(model, inputs, mask, label_ids):
    weights = model.lm_head.weight[label_ids].float()
    scores, intervened = [], []

    def score(normalized):
        logits = F.linear(normalized[:, -1, :].float(), weights)
        return logits - logits.mean(dim=-1, keepdim=True)

    def observe(round_number, before, after, read_suffix, replay):
        if round_number == 1:
            scores.append(score(model.model.norm(read_suffix(before))))
        if round_number < model.config.loop_count:
            scores.append(score(model.model.norm(read_suffix(after))))
        intervened.append(score(model.model.norm(replay(GATE))))

    final = model.model(input_ids=inputs, attention_mask=mask, use_cache=False,
                        block_round_observer=observe).last_hidden_state
    scores.append(score(final))
    immediate = torch.stack(scores, dim=1).diff(dim=1)
    counterfactual = (scores[-1][:, None] - torch.stack(intervened, dim=1)) / (1 - GATE)
    return immediate.cpu(), counterfactual.cpu()


def main():
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    from loop_jev.evaluation.zero_shot import read_rows, render_prompt
    from loop_jev.looping import enable_configured_looping
    from loop_jev.training.latent_kv import marker_token_id
    from loop_jev.training.tiny_loop_distill import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "teacher"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("train", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--train-sample-seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_rows(args.train)
    if min(args.batch_size, args.max_train_examples) < 1:
        parser.error("Batch size and sample count must be positive")
    if len(rows) > args.max_train_examples:
        indices = random.Random(args.train_sample_seed).sample(range(len(rows)), args.max_train_examples)
        rows = [rows[i] for i in indices]
    if not rows or any(row.get("split") != "train" or len(row["options"]) != 4 for row in rows):
        parser.error("Expected four-option training rows")

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("gloo")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    teacher = load_model(args.teacher, device)
    cfg = teacher.config
    if ((cfg.loop_start_layer, cfg.loop_end_layer, cfg.loop_count, cfg.loop_damping_step) != (9, 18, 10, 0.1)
            or cfg.loop_schedule != "block" or not cfg.loop_damping or not cfg.latent_slots
            or cfg.latent_marker_token_id != marker_token_id(tokenizer)
            or any(getattr(cfg, name, None) for name in (
                "loop_transition_rank", "loop_carry_workspace", "loop_multiscale_workspace",
                "loop_staged_handoff", "loop_evidence_reread", "loop_refinement_token_ids",
                "latent_core_width", "option_workspace"))):
        parser.error("Expected the plain damped 9-layer, 10-round teacher with step 0.1")
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in "ABCD"]
    if any(len(ids) != 1 for ids in labels):
        parser.error("Expected single-token option labels")
    label_ids = [ids[0] for ids in labels]
    enable_configured_looping(teacher)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    shard = rows[rank::world]
    local = []
    for offset in tqdm(range(0, len(shard), args.batch_size), desc=f"Round effects rank {rank}"):
        batch = shard[offset:offset + args.batch_size]
        prompts = [render_prompt(tokenizer, row, cfg.latent_slots, cfg.latent_marker_token_id) for row in batch]
        width = max(map(len, prompts))
        inputs = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
        immediate, counterfactual = extract_round_effects(teacher, inputs, mask, label_ids)
        local.extend(zip((row["id"] for row in batch), immediate.tolist(), counterfactual.tolist(), strict=True))
    if world > 1:
        parts = [None] * world if rank == 0 else None
        dist.gather_object(local, parts, dst=0)
        dist.destroy_process_group()
        if rank != 0:
            return
        local = [item for part in parts for item in part]
    ids, *values = zip(*local, strict=True)
    if len(ids) != len(rows) or len(set(ids)) != len(rows):
        raise ValueError("Missing or duplicate training IDs")
    cache = dict(metadata=cache_metadata(train=args.train, model=args.model, teacher=args.teacher,
                                         layer=cfg.loop_end_layer, seed=args.train_sample_seed,
                                         rounds=cfg.loop_count, damping_step=cfg.loop_damping_step),
                 gate=GATE, ids=list(ids))
    cache["metadata"]["start_layer"] = cfg.loop_start_layer
    for mode, targets in zip(MODES, values, strict=True):
        tensor = torch.tensor(targets, dtype=torch.float32)
        rms = tensor.square().mean(dim=(0, 2)).sqrt()
        if not rms.isfinite().all() or not rms.gt(0).all():
            raise ValueError(f"Nonfinite or zero round targets for {mode}")
        cache[mode] = tensor
        print(f"{mode} RMS by round: {rms.tolist()}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    torch.save(cache, temp)
    temp.replace(args.output)
    print(f"Saved {len(ids)} training targets to {args.output}")


if __name__ == "__main__":
    main()
