"""Compile teacher time into three causal groups of existing workspace tokens."""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from loop_jev.evaluation.coarse_state import Backbone, pack, reduce_gradients
from loop_jev.evaluation.graph_short_loop import evaluate, prepare_prompts
from loop_jev.evaluation.time_fold import install, merge_projections
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import damped_update
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.transition import load_loop_model


PHASES = ((12, 2), (15, 5), (18, 10))
SLOTS = 24


def workspace_positions(lengths, answer_width):
    return lengths[:, None] - answer_width - SLOTS + torch.arange(SLOTS, device=lengths.device)[None]


def take_slots(hidden, positions):
    return hidden.gather(1, positions[..., None].expand(-1, -1, hidden.size(-1)))


def coarsen(hidden):
    return hidden.float().reshape(hidden.size(0), 8, 3, hidden.size(-1)).mean(2)


def rotate_options(row, shift):
    permutation = [(i + shift) % 4 for i in range(4)]
    return {**row, 'options': [row['options'][i] for i in permutation],
            'gold_index': permutation.index(row['gold_index'])}, permutation


def consensus_logits(original, rotated, mapping):
    aligned = torch.zeros_like(rotated).scatter(1, mapping, rotated)
    log_probs = torch.stack(((original / 2).log_softmax(-1), (aligned / 2).log_softmax(-1)))
    consensus = torch.logsumexp(log_probs, dim=0) * 2
    return consensus, consensus.gather(1, mapping)


class PhaseSlots(nn.Module):
    def __init__(self, hidden_size, answer_width):
        super().__init__()
        self.answer_width = answer_width
        self.roles = nn.Parameter(torch.zeros(3, hidden_size))

    def project(self, base, before, context, lengths, *, supervise=False):
        positions = workspace_positions(lengths, self.answer_width)
        indices = positions[..., None].expand(-1, -1, before.size(-1))
        offsets = self.roles.to(before.dtype).repeat_interleave(8, dim=0)[None].expand(before.size(0), -1, -1)
        state = before.scatter(1, indices, take_slots(before, positions) + offsets)
        artifacts, lo = [], 9
        for phase, (hi, _) in enumerate(PHASES):
            state = base.stack(state, context, lo, hi)
            if supervise:
                artifacts.append(take_slots(state, positions[:, phase * 8:(phase + 1) * 8]))
            lo = hi
        return base.score(state, context, lengths), artifacts

    def predict(self, base, ids, lengths):
        before, context = base.prefix(ids)
        return self.project(base, before, context, lengths)[0].argmax(-1)


@torch.no_grad()
def capture(base, examples, pad, device, answer_width):
    records = []
    for offset in range(0, len(examples), 4):
        prompts, gold = zip(*examples[offset:offset + 4], strict=True)
        ids, lengths, valid = pack(prompts, pad, device)
        before, context = base.prefix(ids, valid)
        positions = workspace_positions(lengths, answer_width)
        origin = coarsen(take_slots(before, positions))
        state = before
        knots = []
        for rnd in range(1, 11):
            hidden, lo = state, 9
            for hi, phase in PHASES:
                hidden = base.stack(hidden, context, lo, hi)
                if rnd == phase and hi != 18:
                    knots.append(coarsen(take_slots(hidden, positions)).to(before.dtype))
                lo = hi
            state = damped_update(state, hidden, 10, .1)
        knots.append(coarsen(take_slots(state, positions)).to(before.dtype))
        logits = base.score(state, context, lengths)
        records.append(tuple(t.cpu() for t in (before, origin, *knots, logits, lengths, valid)) + (torch.tensor(gold),))
        if (offset // 4 + 1) % 32 == 0:
            print(f'{device}: captured {offset // 4 + 1}/{len(examples) // 4} batches', flush=True)
    return records


def phase_objective(predicted, targets, origin):
    losses = []
    for state, target in zip(predicted, targets, strict=True):
        scale = (target.float() - origin).square().mean(-1, keepdim=True).clamp_min(.01).sqrt()
        error = (state.float() - target.float()) / scale
        losses.append(F.smooth_l1_loss(error, torch.zeros_like(error)))
    return torch.stack(losses).mean()


def paired_records(base, chosen, render, pad, device, answer_width):
    rng = random.Random(537)
    examples = []
    for row in chosen:
        rotated, mapping = rotate_options(row, rng.randrange(1, 4))
        examples.append(((render(row), row['gold_index']), (render(rotated), rotated['gold_index']), mapping))
    examples.sort(key=lambda ex: max(len(ex[0][0]), len(ex[1][0])))
    identity, rotated, mappings = zip(*examples, strict=True)
    first = capture(base, identity, pad, device, answer_width)
    second = capture(base, rotated, pad, device, answer_width)
    for i, (a, b) in enumerate(zip(first, second, strict=True)):
        mapping = torch.tensor(mappings[4 * i:4 * (i + 1)])
        primary, alternate = consensus_logits(a[5], b[5], mapping)
        first[i], second[i] = (a[:5] + (primary,) + a[6:], b[:5] + (alternate,) + b[6:])
    return first, second


def train(args, base, render, pad, device, rank, world, answer_width):
    pool = read_rows(args.train)
    if (world not in (1, 2) or args.examples % 16 or not 0 < args.examples <= len(pool) or
            len({row['id'] for row in pool}) != len(pool) or
            any(row['split'] != 'train' or len(row['options']) != 4 or row['gold_index'] not in range(4) for row in pool)):
        raise ValueError('Expected unique training rows and an effective batch size of 16')
    chosen = random.Random(args.sample_seed).sample(pool, args.examples)
    if args.paired:
        records, alternate = paired_records(base, chosen[rank::world], render, pad, device, answer_width)
    else:
        examples = sorted(((render(row), row['gold_index']) for row in chosen), key=lambda item: len(item[0]))
        records = capture(base, examples[rank::world], pad, device, answer_width)
    torch.manual_seed(43)
    core = PhaseSlots(base.model.config.hidden_size, answer_width).to(device)
    layers = install(base, args.rank, lo=9, hi=args.train_end)
    params = [*core.parameters(), *(p for layer in layers for p in (layer.down, layer.up))]
    optimizer = torch.optim.AdamW(params, lr=args.lr, fused=True)
    accumulation = 4 // world
    rng, order = random.Random(43), list(range(len(records)))
    history = []
    for epoch in range(args.epochs):
        totals = torch.zeros(3, device=device)
        rng.shuffle(order)
        for i, index in enumerate(order, 1):
            batch = alternate[index] if args.paired and (index + epoch) % 2 else records[index]
            before, origin, early, middle, late, teacher, lengths, valid, gold = (t.to(device) for t in batch)
            context = base.context(before, valid)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, stages = core.project(base, before, context, lengths, supervise=args.phase_weight > 0)
            phase = phase_objective(stages, (early, middle, late), origin) if stages else logits.new_zeros(())
            kl = 4 * F.kl_div((logits / 2).log_softmax(-1), (teacher / 2).softmax(-1), reduction='batchmean')
            ce = F.cross_entropy(logits, gold)
            ((args.phase_weight * phase + kl + args.ce_weight * ce) / accumulation).backward()
            if i % accumulation == 0:
                if world > 1:
                    reduce_gradients(params, world)
                nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            totals += torch.stack((phase.detach(), kl.detach(), ce.detach()))
        totals = totals.cpu()
        if world > 1:
            dist.all_reduce(totals)
        metrics = dict(zip(('phase', 'kl', 'ce'), (totals / (world * len(records))).tolist(), strict=True))
        history.append(metrics)
        if rank == 0:
            print(f'epoch {epoch + 1}: {metrics}', flush=True)
            args.output.mkdir(parents=True, exist_ok=True)
            torch.save(dict(weights=[(m.down.detach().cpu(), m.up.detach().cpu()) for m in layers],
                            roles=core.roles.detach().cpu(), teacher=str(Path(args.model).resolve()),
                            rank=args.rank, train_end=args.train_end, phase_weight=args.phase_weight,
                            ce_weight=args.ce_weight, sample_seed=args.sample_seed, examples=args.examples,
                            paired=args.paired, history=history), args.output / 'adapter.pt')


@torch.no_grad()
def benchmark(args, base, render, pad, device, answer_width):
    prompts = prepare_prompts(read_rows(args.test), render, pad, device)
    saved = torch.load(args.adapter, map_location='cpu', weights_only=True)
    if saved['teacher'] != str(Path(args.model).resolve()):
        raise ValueError('Adapter does not match the teacher')
    merge_projections(base, saved['rank'], saved['weights'], lo=9, hi=saved['train_end'])
    core = PhaseSlots(base.model.config.hidden_size, answer_width).to(device=device, dtype=torch.bfloat16).eval()
    core.roles.copy_(saved['roles'])
    result = evaluate(base, prompts, 'time-nodes', eager=False, core=core)
    print(result, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'test.json').write_text(json.dumps(dict(model=args.model, adapter=str(args.adapter),
        input=str(args.test), device=torch.cuda.get_device_name(), batch_size=1, bucket=16,
        results={'phase-slots': result}), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--train', type=Path, default=Path('data/gsm-mc/train.jsonl'))
    parser.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--examples', type=int, default=1024)
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--train-end', type=int, default=18)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--phase-weight', type=float, default=.5)
    parser.add_argument('--ce-weight', type=float, default=1)
    parser.add_argument('--paired', action='store_true', help='Distill a consensus of canonical and rotated option prompts')
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if (args.rank < 1 or args.epochs < 1 or args.lr <= 0 or args.phase_weight < 0 or args.ce_weight < 0 or
            not 18 <= args.train_end <= 28 or (args.adapter and world != 1)):
        parser.error('Invalid training options or multi-device latency benchmark')
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('gloo')
    try:
        model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={'': device.index},
                                attn_implementation='sdpa', trust_remote_code=False).eval().requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        marker = marker_token_id(tokenizer)
        validate_checkpoint(model.config, marker)
        labels = [tokenizer.encode(x, add_special_tokens=False) for x in 'ABCD']
        if any(len(x) != 1 for x in labels) or model.config.latent_slots != SLOTS:
            raise ValueError('Expected single-token labels and exactly 24 latent slots')
        base = Backbone(model, [x[0] for x in labels])
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        answer_width = len(tokenizer.encode('Answer: (', add_special_tokens=False))

        def render(row):
            return render_prompt(tokenizer, row, SLOTS, marker)

        if args.adapter:
            benchmark(args, base, render, pad, device, answer_width)
        else:
            train(args, base, render, pad, device, rank, world, answer_width)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
