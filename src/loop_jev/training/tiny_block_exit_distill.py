"""Distill ten teacher updates into a tiny recurrence inside a frozen Qwen."""

import argparse
import json
import os
import random
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import enable_configured_looping
from loop_jev.training.block_state_cache import cache_metadata
from loop_jev.training.tiny_loop_distill import ROUNDS, WIDTH, center, encode, gather, load_model

SCORE_DELTA = "score_delta"


def block_exit_basis(hidden_size, device):
    basis = torch.randn(hidden_size, WIDTH, generator=torch.Generator().manual_seed(43))
    return torch.linalg.qr(basis).Q.contiguous().to(device)


class BlockExitCore(nn.Module):
    def __init__(self, projection):
        super().__init__()
        hidden_size, width = projection.shape
        self.context = nn.Linear(hidden_size, width, bias=False)
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width))
        self.decode = nn.Linear(width, hidden_size, bias=False)
        with torch.no_grad():
            self.decode.weight.copy_(projection)
            self.update[-1].weight.zero_()
            self.update[-1].bias.zero_()

    def forward(self, answer, steps=ROUNDS):
        context = self.context(F.rms_norm(answer.float(), (answer.shape[-1],)))
        state = torch.zeros_like(context)
        path = []
        for _ in range(steps):
            state = state + self.update(torch.cat((state, context), dim=-1))
            path.append(state)
        return self.decode(torch.stack(path, dim=1))


class FrozenQwenReadout:
    def __init__(self, model, layer, label_ids):
        self.backbone = model.model
        self.layer = layer
        self.label_weight = model.lm_head.weight[label_ids].detach()

    @torch.no_grad()
    def prefix(self, inputs, mask):
        backbone = self.backbone
        state = backbone.embed_tokens(inputs)
        positions = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0)
        args = dict(config=backbone.config, inputs_embeds=state, attention_mask=mask,
                    past_key_values=None, position_ids=positions)
        masks = {"full_attention": create_causal_mask(**args)}
        if backbone.has_sliding_layers:
            masks["sliding_attention"] = create_sliding_window_causal_mask(**args)
        embeddings = backbone.rotary_emb(state, positions)
        context = masks, positions, embeddings
        for index in range(self.layer):
            state = self.run_layer(state, index, context)
        return state, context

    def run_layer(self, state, index, context):
        masks, positions, embeddings = context
        return self.backbone.layers[index](
            state, attention_mask=masks[self.backbone.config.layer_types[index]],
            position_ids=positions, position_embeddings=embeddings, use_cache=False,
        )

    def scores(self, prefix, context, answer=None, checkpoint_layers=False):
        state = prefix if answer is None else torch.cat((prefix[:, :-1], answer[:, None].to(prefix.dtype)), dim=1)
        for index in range(self.layer, self.backbone.config.num_hidden_layers):
            if checkpoint_layers:
                state = checkpoint(self.run_layer, state, index, context, use_reentrant=False,
                                   preserve_rng_state=False)
            else:
                state = self.run_layer(state, index, context)
        return F.linear(self.backbone.norm(state)[:, -1], self.label_weight).float()


def suffix_gradients(readout, prefix, context, patches, mode, teacher_scores, gold, score_scale):
    if mode == SCORE_DELTA:
        return score_delta_gradients(readout, prefix, context, patches, teacher_scores, gold, score_scale)
    steps = range(ROUNDS) if mode == "trajectory" else (ROUNDS - 1,)
    gradients = torch.zeros_like(patches)
    ce_value, score_value = patches.new_zeros(()), patches.new_zeros(())
    # Replay and differentiate one upper-stack readout at a time.
    for step in steps:
        answer = patches[:, step].detach().requires_grad_()
        logits = readout.scores(prefix, context, answer, checkpoint_layers=True)
        score = F.smooth_l1_loss(center(logits) / score_scale, teacher_scores[:, step] / score_scale) / len(steps)
        objective = 0.5 * score
        score_value += score.detach()
        if step == ROUNDS - 1:
            ce = F.cross_entropy(logits, gold)
            objective = objective + ce
            ce_value = ce.detach()
        gradients[:, step], = torch.autograd.grad(objective, answer)
    return gradients, ce_value, score_value


def option_gap_deltas(scores):
    updates = scores.diff(dim=1)
    left, right = (0, 0, 0, 1, 1, 2), (1, 2, 3, 2, 3, 3)
    return updates[..., left] - updates[..., right]


def score_delta_gradients(readout, prefix, context, patches, teacher_changes, gold, score_scale):
    # Adjacent differences couple two rounds. Differentiate in score space first,
    # then replay one suffix at a time with its share of the score gradient.
    with torch.no_grad():
        base = readout.scores(prefix, context)
        scores = torch.stack([readout.scores(prefix, context, answer) for answer in patches.unbind(1)], dim=1)
    scores.requires_grad_()
    changes = option_gap_deltas(torch.cat((base[:, None], scores), dim=1))
    score = F.smooth_l1_loss(changes / score_scale, teacher_changes / score_scale)
    ce = F.cross_entropy(scores[:, -1], gold)
    cotangents, = torch.autograd.grad(ce + 0.5 * score, scores)

    gradients = torch.empty_like(patches)
    for step in range(patches.shape[1]):
        answer = patches[:, step].detach().requires_grad_()
        logits = readout.scores(prefix, context, answer, checkpoint_layers=True)
        gradients[:, step], = torch.autograd.grad(logits, answer, grad_outputs=cotangents[:, step])
    return gradients, ce.detach(), score.detach()


def collect_targets(local, world, device):
    local = local.to(device)
    if world == 1:
        return local
    parts = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(parts, local)
    return torch.stack(parts, dim=1).flatten(0, 1)


def batch_inputs(prompts, indices, pad, device):
    rows = [prompts[index] for index in indices]
    width = max(map(len, rows))
    inputs = torch.tensor([[pad] * (width - len(row)) + row for row in rows], device=device)
    mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows], device=device)
    return inputs, mask


def train_core(mode, readout, prompts, pad, targets, projection, scales, args, rank, world):
    device = projection.device
    torch.manual_seed(args.seed)
    core = BlockExitCore(projection).to(device)
    distributed_core = DistributedDataParallel(core, device_ids=[device.index]) if world > 1 else core
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.learning_rate)
    deltas, teacher_scores, labels = targets
    delta_scale, final_scale, score_scale = scales
    order = torch.Generator().manual_seed(args.seed)
    history = []
    for epoch in range(args.epochs):
        total = torch.zeros(4, device=device)
        batches = torch.randperm(len(prompts), generator=order)[rank::world].split(args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(enumerate(batches), total=len(batches), desc=f"{mode} epoch {epoch + 1}", disable=rank != 0)
        for index, batch in progress:
            inputs, mask = batch_inputs(prompts, batch.tolist(), pad, device)
            prefix, context = readout.prefix(inputs, mask)
            group = args.gradient_accumulation_steps
            group_size = min(group, len(batches) - (index // group) * group)
            sync_now = (index + 1) % group == 0 or index + 1 == len(batches)
            sync = distributed_core.no_sync() if world > 1 and not sync_now else nullcontext()
            with sync:
                answer = prefix[:, -1].float()
                path = distributed_core(answer) * delta_scale
                patches = answer[:, None] + path
                projected = path @ projection
                if mode in ("trajectory", SCORE_DELTA):
                    predicted = torch.cat((torch.zeros_like(projected[:, :1]), projected), dim=1).diff(dim=1)
                    state = F.smooth_l1_loss(predicted / delta_scale, deltas[batch] / delta_scale)
                else:
                    state = F.smooth_l1_loss(projected[:, -1] / final_scale, deltas[batch].sum(1) / final_scale)
                gradients, ce, score = suffix_gradients(readout, prefix, context, patches, mode,
                                                         teacher_scores[batch], labels[batch], score_scale)
                torch.autograd.backward((state, patches), (state.new_tensor(0.5 / group_size), gradients / group_size))
            if sync_now:
                nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total[:3] += torch.stack((ce, state.detach(), score)) * len(batch)
            total[3] += len(batch)
        if world > 1:
            dist.all_reduce(total)
        losses = (total[:3] / total[3]).tolist()
        history.append(dict(zip(("ce", "state", "scores"), losses, strict=True)))
        if rank == 0:
            print(f"{mode} epoch {epoch + 1}: {history[-1]}", flush=True)
    return core.eval(), history


@torch.no_grad()
def evaluate(readout, cores, prompts, pad, delta_scale, batch_size, rank, world):
    device = readout.label_weight.device
    local = []
    indices = list(range(rank, len(prompts), world))
    for begin in tqdm(range(0, len(indices), batch_size), desc="GSM-MC test", disable=rank != 0):
        batch = indices[begin:begin + batch_size]
        inputs, mask = batch_inputs(prompts, batch, pad, device)
        prefix, context = readout.prefix(inputs, mask)
        answer = prefix[:, -1].float()
        outputs = {"base_0": readout.scores(prefix, context).argmax(-1).tolist()}
        for mode, core in cores.items():
            path = core(answer) * delta_scale
            for step in (1, 5, ROUNDS):
                outputs[f"{mode}_{step}"] = readout.scores(prefix, context, answer + path[:, step - 1]).argmax(-1).tolist()
        local.extend((idx, {name: value[i] for name, value in outputs.items()}) for i, idx in enumerate(batch))
    return local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "student", "train", "test", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--modes", nargs="+", choices=("trajectory", "endpoint", SCORE_DELTA),
                        default=("trajectory", "endpoint"))
    args = parser.parse_args()
    if min(args.batch_size, args.gradient_accumulation_steps, args.epochs) < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("Expected positive batch sizes, epochs and learning rate")
    if len(set(args.modes)) != len(args.modes):
        parser.error("Expected each training mode at most once")
    experiment = json.loads((args.teacher / "experiment.json").read_text())
    pool, test = read_rows(args.train), read_rows(args.test)
    for split, rows in (("train", pool), ("test", test)):
        if len({row["id"] for row in rows}) != len(rows) or any(
            row["split"] != split or len(row["options"]) != 4 for row in rows
        ):
            parser.error(f"Expected unique four-option {split} rows")
    if (len(test) != 1319 or len(pool) < 1024 or experiment["examples"] != 1024
            or Path(experiment["train"]).resolve() != args.train.resolve()):
        parser.error("Expected the teacher's original training pool and 1319 test rows")
    sample_seed = experiment["train_sample_seed"]
    train = random.Random(sample_seed).sample(pool, 1024) if len(pool) > 1024 else pool
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    if len(train) % world:
        parser.error("Training rows must divide evenly across ranks")
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))

    teacher = load_model(args.teacher, device)
    cfg = teacher.config
    if (cfg.loop_count != ROUNDS or cfg.loop_end_layer != 18 or not cfg.loop_damping
            or cfg.loop_schedule != "block" or not cfg.latent_slots or any(getattr(cfg, name, None) for name in (
                "loop_transition_rank", "loop_carry_workspace", "loop_multiscale_workspace",
                "loop_staged_handoff", "loop_evidence_reread", "loop_refinement_token_ids",
                "latent_core_width", "option_workspace"))):
        parser.error("Expected a plain ten-round damped teacher ending at layer 18")
    enable_configured_looping(teacher)
    projection = block_exit_basis(cfg.hidden_size, device)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=False)
    include_initial = SCORE_DELTA in args.modes
    local_deltas, local_scores = encode(teacher, tokenizer, train[rank::world], args.batch_size,
                                        projection=projection, include_initial_scores=include_initial)
    teacher.model.forward = teacher.model._unlooped_forward
    del teacher
    torch.cuda.empty_cache()

    deltas = collect_targets(local_deltas, world, device)
    score_path = collect_targets(local_scores, world, device)
    if not deltas.isfinite().all() or not score_path.isfinite().all():
        raise ValueError("Expected finite teacher trajectories")
    scores = score_path[:, 1:] if include_initial else score_path
    teacher_changes = option_gap_deltas(score_path) if include_initial else None
    labels = torch.tensor([r["gold_index"] for r in train], device=device)
    delta_scale, final_scale = (x.square().mean().sqrt().clamp_min(1e-6) for x in (deltas, deltas.sum(1)))

    model = load_model(args.student, device).requires_grad_(False)
    if (model.config.model_type != "qwen3" or model.config.hidden_size != cfg.hidden_size
            or model.config.num_hidden_layers <= cfg.loop_end_layer or getattr(model.config, "loop_count", 1) != 1):
        parser.error("Expected a matching single-pass Qwen student with layers above the block")
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=False)
    label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in "ABCD"]
    if any(len(ids) != 1 for ids in label_ids):
        parser.error("Expected four single-token labels")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    readout = FrozenQwenReadout(model, cfg.loop_end_layer, [ids[0] for ids in label_ids])
    prompts = [render_prompt(tokenizer, row) for row in train]
    cores, history = {}, {}
    for mode in args.modes:
        target_scores = teacher_changes if mode == SCORE_DELTA else scores
        score_scale = target_scores.square().mean().sqrt().clamp_min(1e-6)
        scales = delta_scale, final_scale, score_scale
        cores[mode], history[mode] = train_core(mode, readout, prompts, pad, (deltas, target_scores, labels),
                                               projection, scales, args, rank, world)

    predictions = evaluate(readout, cores, [render_prompt(tokenizer, row) for row in test],
                           pad, delta_scale, args.batch_size, rank, world)
    peak = torch.cuda.max_memory_reserved(device) / 2**30
    parts = gather((predictions, peak), world, rank)
    if world > 1:
        dist.destroy_process_group()
    if rank != 0:
        return

    by_index = dict(item for shard, _ in parts for item in shard)
    if len(by_index) != sum(len(shard) for shard, _ in parts) or set(by_index) != set(range(len(test))):
        raise ValueError("Test predictions do not cover every question")
    results = {}
    for name in by_index[0]:
        correct = sum(by_index[i][name] == row["gold_index"] for i, row in enumerate(test))
        results[name] = dict(correct=correct, total=len(test), accuracy=correct / len(test))
        print(f"{name:15} {correct / len(test):.5f} ({correct}/{len(test)})")
    deltas_pp = {}
    for mode, references in (("trajectory", ("endpoint_10", "base_0")),
                             (SCORE_DELTA, ("trajectory_10", "endpoint_10", "base_0"))):
        if mode in cores:
            for name in references:
                if name in results:
                    deltas_pp[f"{mode}_vs_{name}_pp"] = 100 * (results[f"{mode}_10"]["accuracy"] -
                                                                results[name]["accuracy"])
    for key, value in deltas_pp.items():
        print(f"{key}: {value:+.2f} pp")
    metadata = cache_metadata(train=args.train, model=experiment["model"], teacher=str(args.teacher),
                              layer=cfg.loop_end_layer, seed=sample_seed, rounds=ROUNDS, damping_step=cfg.loop_damping_step)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(weights={mode: {key: value.cpu() for key, value in core.state_dict().items()}
                             for mode, core in cores.items()}, layer=cfg.loop_end_layer, rounds=ROUNDS,
                    student=str(args.student), width=WIDTH, delta_scale=delta_scale.item()), args.output / "tiny-loop.pt")
    rows = [dict(id=row["id"], gold_index=row["gold_index"], predictions=by_index[i]) for i, row in enumerate(test)]
    (args.output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = dict(results=results, history=history, **deltas_pp, teacher=metadata, student=str(args.student),
                   parameters=sum(p.numel() for p in next(iter(cores.values())).parameters()),
                   peak_reserved_gib=max(peak for _, peak in parts))
    path = args.output / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Peak reserved: {summary['peak_reserved_gib']:.3f} GiB. Saved: {path}")


if __name__ == "__main__":
    main()
