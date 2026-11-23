"""Distill a frozen block teacher into a small recurrent module after frozen Qwen."""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import enable_configured_looping
from loop_jev.training.block_state_cache import cache_metadata
from loop_jev.transition import load_loop_model


ROUNDS = 10
WIDTH = 64


class TinyLoop(nn.Module):
    def __init__(self, hidden_size, width=WIDTH):
        super().__init__()
        self.context = nn.Linear(hidden_size, width, bias=False)
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width))
        self.readout = nn.Linear(width, 4, bias=False)

    def forward(self, answer, base_scores, steps=ROUNDS):
        context = self.context(F.rms_norm(answer, (answer.shape[-1],)))
        state = torch.zeros_like(context)
        updates, scores = [], []
        for _ in range(steps):
            delta = self.update(torch.cat((state, context), dim=-1))
            state = state + delta
            updates.append(delta)
            scores.append(base_scores + self.readout(state))
        return torch.stack(updates, dim=1), torch.stack(scores, dim=1)


def center(scores):
    return scores - scores.mean(dim=-1, keepdim=True)


@torch.inference_mode()
def teacher_trajectory(model, inputs, mask, label_weight, projection, *, include_initial_scores=False):
    cfg = model.config
    states, scores = [], []
    initial = entry = None

    def answer_state(state):
        answer = state[:, -1, :].float()
        return answer if projection is None else answer @ projection

    def capture_initial(_layer, args):
        nonlocal initial, entry
        if initial is None:
            initial = answer_state(args[0])
            if include_initial_scores:
                entry = args[0]

    def observe(round_number, state, read_suffix):
        nonlocal entry
        states.append(answer_state(state))
        if round_number == 1 and include_initial_scores:
            hidden = model.model.norm(read_suffix(entry))[:, -1, :]
            scores.append(F.linear(hidden, label_weight).float())
            entry = None
        if round_number < ROUNDS:
            hidden = model.model.norm(read_suffix(state))[:, -1, :]
            scores.append(F.linear(hidden, label_weight).float())

    handle = model.model.layers[cfg.loop_start_layer].register_forward_pre_hook(capture_initial)
    try:
        final = model.model(input_ids=inputs, attention_mask=mask, use_cache=False,
                            block_state_observer=observe).last_hidden_state
    finally:
        handle.remove()
    scores.append(F.linear(final[:, -1, :], label_weight).float())
    if initial is None or len(states) != ROUNDS:
        raise ValueError("Expected ten teacher states and one pre-block state")
    deltas = torch.stack([initial, *states], dim=1).diff(dim=1)
    return deltas.cpu(), center(torch.stack(scores, dim=1)).cpu()


@torch.inference_mode()
def encode(model, tokenizer, rows, batch_size, *, projection=None, include_initial_scores=False):
    from tqdm.auto import tqdm

    cfg = model.config
    weight = model.lm_head.weight
    device = weight.device
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in "ABCD"]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError("Expected four single-token option labels")
    label_weight = weight[[ids[0] for ids in labels]]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    first, second = [], []
    for offset in tqdm(range(0, len(rows), batch_size), desc="Teacher trajectory" if projection is not None else "Frozen Qwen"):
        batch = rows[offset:offset + batch_size]
        prompts = ([render_prompt(tokenizer, row, cfg.latent_slots, cfg.latent_marker_token_id) for row in batch]
                   if projection is not None else [render_prompt(tokenizer, row) for row in batch])
        width = max(map(len, prompts))
        inputs = torch.tensor([[pad] * (width - len(prompt)) + prompt for prompt in prompts], device=device)
        mask = torch.tensor([[0] * (width - len(prompt)) + [1] * len(prompt) for prompt in prompts], device=device)
        if projection is not None:
            a, b = teacher_trajectory(model, inputs, mask, label_weight, projection,
                                      include_initial_scores=include_initial_scores)
        else:
            hidden = model.model(input_ids=inputs, attention_mask=mask, use_cache=False).last_hidden_state[:, -1]
            a, b = hidden.float().cpu(), center(F.linear(hidden, label_weight).float()).cpu()
        first.append(a)
        second.append(b)
    return torch.cat(first), torch.cat(second)


def fit(mode, answer, base, deltas, scores, gold, args):
    torch.manual_seed(args.seed)
    core = TinyLoop(answer.shape[-1])
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.learning_rate)
    order = torch.Generator().manual_seed(args.seed)
    delta_scale = deltas.square().mean().sqrt().clamp_min(1e-6)
    final_state = deltas.sum(1)
    final_scale = final_state.square().mean().sqrt().clamp_min(1e-6)
    score_scale = (scores - base[:, None]).square().mean().sqrt().clamp_min(1e-6)
    history = []
    for epoch in range(args.epochs):
        means = torch.zeros(4)
        batches = torch.randperm(len(gold), generator=order).split(args.fit_batch_size)
        for batch in batches:
            pred_delta, pred_scores = core(answer[batch], base[batch])
            if mode == "trajectory":
                state_loss = F.smooth_l1_loss(pred_delta, deltas[batch] / delta_scale)
                score_loss = F.smooth_l1_loss(center(pred_scores) / score_scale, scores[batch] / score_scale)
            else:
                state_loss = F.smooth_l1_loss(pred_delta.sum(1) * delta_scale / final_scale,
                                              final_state[batch] / final_scale)
                score_loss = F.smooth_l1_loss(center(pred_scores[:, -1]) / score_scale,
                                              scores[batch, -1] / score_scale)
            ce = F.cross_entropy(pred_scores[:, -1], gold[batch])
            loss = ce + 0.5 * state_loss + 0.5 * score_loss
            if not loss.isfinite():
                raise ValueError(f"Nonfinite tiny-loop loss in {mode}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            optimizer.step()
            means += torch.stack((loss, ce, state_loss, score_loss)).detach() * len(batch)
        history.append(dict(zip(("loss", "ce", "state", "scores"), (means / len(gold)).tolist(), strict=True)))
        print(f"{mode} epoch {epoch + 1}: {history[-1]}", flush=True)
    return core.eval(), history


def load_model(path, device):
    return load_loop_model(str(path), dtype=torch.bfloat16, device_map={"": device},
                           attn_implementation="sdpa", trust_remote_code=False).eval()


def gather(local, world, rank):
    if world == 1:
        return [local]
    parts = [None] * world if rank == 0 else None
    dist.gather_object(local, parts, dst=0)
    return parts


def main():
    from datetime import timedelta

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    for name in ("teacher", "train", "test", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fit-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if min(args.batch_size, args.fit_batch_size, args.epochs) < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("Expected positive batch sizes, epochs and learning rate")
    experiment = json.loads((args.teacher / "experiment.json").read_text())
    train_pool, test = read_rows(args.train), read_rows(args.test)
    for split, rows, count in (("train", train_pool, 1024), ("test", test, 1319)):
        if (len(rows) < count or len({row["id"] for row in rows}) != len(rows)
                or any(row["split"] != split or len(row["options"]) != 4 for row in rows)):
            parser.error(f"Expected unique four-option {split} rows")
    if (len(test) != 1319 or experiment["examples"] != 1024 or experiment["model"] != args.model
            or Path(experiment["train"]).resolve() != args.train.resolve()):
        parser.error("Expected the original 1024-question teacher training pool and 1319 test rows")
    seed = experiment["train_sample_seed"]
    train = random.Random(seed).sample(train_pool, 1024) if len(train_pool) > 1024 else train_pool
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = int(os.environ.get("LOCAL_RANK", 0))
    if world > min(len(train), len(test)):
        parser.error("Each worker must receive training and test rows")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(hours=2))

    train_shard = train[rank::world]
    teacher = load_model(args.teacher, device)
    cfg = teacher.config
    if (cfg.loop_count != ROUNDS or not cfg.loop_damping or cfg.loop_schedule != "block" or not cfg.latent_slots
            or any(getattr(cfg, name, None) for name in (
                "loop_transition_rank", "loop_carry_workspace", "loop_multiscale_workspace",
                "loop_staged_handoff", "loop_evidence_reread", "loop_refinement_token_ids",
                "latent_core_width", "option_workspace"))):
        parser.error("Expected a plain ten-round damped block teacher")
    enable_configured_looping(teacher)
    projection = torch.empty(cfg.hidden_size, WIDTH).bernoulli_(0.5, generator=torch.Generator().manual_seed(43))
    projection = ((2 * projection - 1) / cfg.hidden_size**0.5).to(device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=False)
    deltas, scores = encode(teacher, teacher_tokenizer, train_shard, args.batch_size, projection=projection)
    teacher.model.forward = teacher.model._unlooped_forward
    del teacher, projection
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    base_model = load_model(args.model, device)
    answers, base_scores = encode(base_model, tokenizer, train_shard, args.batch_size)
    parts = gather((answers, base_scores, deltas, scores, torch.tensor([r["gold_index"] for r in train_shard])),
                   world, rank)
    modes = ("trajectory", "endpoint")
    checkpoint = [None]
    histories = {}
    if rank == 0:
        tensors = [torch.stack([part[i][j] for j in range(max(len(p[i]) for p in parts))
                                for part in parts if j < len(part[i])]) for i in range(5)]
        weights = {}
        for mode in modes:
            core, histories[mode] = fit(mode, *tensors, args)
            weights[mode] = core.state_dict()
        checkpoint[0] = weights
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save(dict(weights=weights, hidden_size=answers.shape[-1], width=WIDTH, rounds=ROUNDS,
                        model=args.model), args.output / "tiny-loop.pt")
    if world > 1:
        dist.broadcast_object_list(checkpoint, src=0)

    test_shard = test[rank::world]
    args.output.mkdir(parents=True, exist_ok=True)
    features, logits = encode(base_model, tokenizer, test_shard, args.batch_size)
    predictions = {"base_0": logits.argmax(-1).tolist()}
    with torch.no_grad():
        for mode in modes:
            core = TinyLoop(features.shape[-1]).eval()
            core.load_state_dict(checkpoint[0][mode])
            _, path = core(features, logits)
            for step in (1, 5, ROUNDS):
                predictions[f"{mode}_{step}"] = path[:, step - 1].argmax(-1).tolist()
    local_results = [dict(id=row["id"], gold_index=row["gold_index"],
                          predictions={key: value[i] for key, value in predictions.items()})
                     for i, row in enumerate(test_shard)]
    results = gather((local_results, torch.cuda.max_memory_reserved(device) / 2**30), world, rank)
    if world > 1:
        dist.destroy_process_group()
    if rank != 0:
        return
    rows = {row["id"]: row for part, _peak in results for row in part}
    if len(rows) != len(test) or rows.keys() != {row["id"] for row in test}:
        raise ValueError("Missing tiny-loop test results")
    stats = {}
    for key in predictions:
        correct = sum(row["predictions"][key] == row["gold_index"] for row in rows.values())
        stats[key] = dict(correct=correct, total=len(rows), accuracy=correct / len(rows))
        print(f"{key:15} {correct / len(rows):.5f} ({correct}/{len(rows)})")
    parameters = sum(p.numel() for p in core.parameters())
    delta_pp = 100 * (stats["trajectory_10"]["accuracy"] - stats["endpoint_10"]["accuracy"])
    peak = max(peak for _part, peak in results)
    summary = dict(results=stats, history=histories, parameters=parameters, peak_reserved_gib=peak,
                   trajectory_vs_endpoint_pp=delta_pp,
                   teacher=cache_metadata(train=args.train, model=args.model, teacher=str(args.teacher),
                                          layer=cfg.loop_end_layer, seed=seed, rounds=ROUNDS,
                                          damping_step=cfg.loop_damping_step))
    (args.output / "predictions.jsonl").write_text("".join(json.dumps(rows[row["id"]]) + "\n" for row in test))
    output = args.output / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"trajectory_vs_endpoint: {delta_pp:+.2f} pp; parameters: {parameters:,}; peak reserved: {peak:.3f} GiB")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
