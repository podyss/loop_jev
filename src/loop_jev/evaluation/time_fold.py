"""Fold a damped teacher trajectory into depth-aligned single-pass targets."""

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

from loop_jev.evaluation.coarse_state import Backbone, pack, reduce_gradients, state_objective
from loop_jev.evaluation.graph_short_loop import evaluate, prepare_prompts
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import damped_update
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.transition import load_loop_model


ANCHORS = (12, 15, 18)


class FoldLinear(nn.Module):
    def __init__(self, linear, rank):
        super().__init__()
        self.linear = linear
        options = dict(device=linear.weight.device, dtype=torch.float32)
        self.down = nn.Parameter(torch.empty(rank, linear.in_features, **options))
        self.up = nn.Parameter(torch.zeros(linear.out_features, rank, **options))
        nn.init.kaiming_uniform_(self.down, a=5**0.5)

    def forward(self, x):
        return self.linear(x) + F.linear(F.linear(x, self.down), self.up)

    @torch.no_grad()
    def merge(self):
        weight = self.linear.weight
        weight.copy_(weight.float() + self.up @ self.down)
        return self.linear


def projections(base, lo=9, hi=18):
    for idx in range(lo, hi):
        layer = base.model.layers[idx]
        for group, names in ((layer.self_attn, ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
                             (layer.mlp, ('gate_proj', 'up_proj', 'down_proj'))):
            for name in names:
                yield group, name


def fold_targets(run_segment, before, *, rounds=10, step=.1):
    # At a given depth d, integrate F_{<=d}(H_r) - H_r across time.
    # The last anchor gets the actual BF16 Euler endpoint, including its rounding.
    targets = [before.float().clone() for _ in ANCHORS[:-1]]
    state = before
    for _ in range(rounds):
        origin, hidden, lo = state, state, 9
        for i, hi in enumerate(ANCHORS):
            hidden = run_segment(hidden, lo, hi)
            if i < len(targets):
                targets[i].add_(hidden.float() - origin.float(), alpha=step)
            lo = hi
        state = damped_update(origin, hidden, rounds, step)
    return (*[t.to(before.dtype) for t in targets], state)


@torch.no_grad()
def capture(base, examples, pad, device):
    records = []
    for start in range(0, len(examples), 4):
        prompts, gold = zip(*examples[start:start + 4], strict=True)
        ids, lengths, valid = pack(prompts, pad, device)
        before, context = base.prefix(ids, valid)
        targets = fold_targets(lambda x, lo, hi: base.stack(x, context, lo, hi), before)
        logits = base.score(targets[-1], context, lengths)
        records.append(tuple(x.cpu() for x in (before, *targets, logits, lengths, valid)) + (torch.tensor(gold),))
        if (start // 4 + 1) % 32 == 0:
            print(f'{device}: captured {start // 4 + 1}/{len(examples) // 4} batches', flush=True)
    return records


def install(base, rank, lo=9, hi=18):
    layers = []
    for group, name in projections(base, lo, hi):
        layer = FoldLinear(getattr(group, name), rank)
        setattr(group, name, layer)
        layers.append(layer)
    return layers


@torch.no_grad()
def merge_projections(base, rank, weights, lo=9, hi=18):
    for (group, name), (down, up) in zip(projections(base, lo, hi), weights, strict=True):
        layer = FoldLinear(getattr(group, name), rank)
        layer.down.copy_(down)
        layer.up.copy_(up)
        setattr(group, name, layer.merge())


def train(args, base, render, pad, device, rank, world):
    pool = read_rows(args.train)
    if (world not in (1, 2) or args.examples % 16 or not 0 < args.examples <= len(pool) or
            len({row['id'] for row in pool}) != len(pool) or
            any(row['split'] != 'train' or len(row['options']) != 4 or row['gold_index'] not in range(4) for row in pool)):
        raise ValueError('Expected unique training rows and an effective batch size of 16')
    chosen = random.Random(42).sample(pool, args.examples)
    examples = sorted(((render(row), row['gold_index']) for row in chosen), key=lambda item: len(item[0]))
    records = capture(base, examples[rank::world], pad, device)
    torch.manual_seed(43)
    layers = install(base, args.rank)
    params = [p for layer in layers for p in (layer.down, layer.up)]
    optimizer = torch.optim.AdamW(params, lr=args.lr, fused=True)
    accumulation = 4 // world
    rng, order = random.Random(43), list(range(len(records)))
    history = []
    for epoch in range(args.epochs):
        totals = torch.zeros(3, device=device)
        rng.shuffle(order)
        for i, index in enumerate(order, 1):
            before, *targets, teacher, lengths, valid, gold = [x.to(device) for x in records[index]]
            context = base.context(before, valid)
            hidden, lo = before, 9
            hidden_states = []
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for hi in ANCHORS:
                    hidden = base.stack(hidden, context, lo, hi)
                    hidden_states.append(hidden)
                    lo = hi
                logits = base.score(hidden, context, lengths)
            state_loss = torch.stack([state_objective(x, target, before, valid)
                                      for x, target in zip(hidden_states, targets, strict=True)]).mean()
            kl = 4 * F.kl_div((logits / 2).log_softmax(-1), (teacher / 2).softmax(-1), reduction='batchmean')
            ce = F.cross_entropy(logits, gold)
            ((args.state_weight * state_loss + kl + ce) / accumulation).backward()
            if i % accumulation == 0:
                if world > 1:
                    reduce_gradients(params, world)
                nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            totals += torch.stack((state_loss.detach(), kl.detach(), ce.detach()))
        totals = totals.cpu()
        if world > 1:
            dist.all_reduce(totals)
        metrics = dict(zip(('state', 'kl', 'ce'), (totals / (world * len(records))).tolist(), strict=True))
        history.append(metrics)
        if rank == 0:
            print(f'epoch {epoch + 1}: {metrics}', flush=True)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save(dict(weights=[(m.down.detach().cpu(), m.up.detach().cpu()) for m in layers],
                        teacher=str(Path(args.model).resolve()), rank=args.rank, state_weight=args.state_weight,
                        examples=args.examples, history=history), args.output / 'adapter.pt')


@torch.no_grad()
def benchmark(args, base, render, pad, device):
    prompts = prepare_prompts(read_rows(args.test), render, pad, device)
    saved = torch.load(args.adapter, map_location='cpu', weights_only=True)
    if saved['teacher'] != str(Path(args.model).resolve()):
        raise ValueError('Adapter does not match the teacher')
    merge_projections(base, saved['rank'], saved['weights'])
    result = evaluate(base, prompts, 'native', eager=False)
    print(result, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'test.json').write_text(json.dumps(dict(model=args.model, adapter=str(args.adapter),
        input=str(args.test), device=torch.cuda.get_device_name(), batch_size=1, bucket=16,
        results={'time-fold': result}), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--train', type=Path, default=Path('data/gsm-mc/train.jsonl'))
    parser.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--examples', type=int, default=1024)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--state-weight', type=float, default=.5)
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if args.rank < 1 or args.epochs < 1 or args.lr <= 0 or args.state_weight < 0 or (args.adapter and world != 1):
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
        if any(len(x) != 1 for x in labels):
            raise ValueError('Expected one token per answer label')
        base = Backbone(model, [x[0] for x in labels])
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        def render(row):
            return render_prompt(tokenizer, row, model.config.latent_slots, marker)

        if args.adapter:
            benchmark(args, base, render, pad, device)
        else:
            train(args, base, render, pad, device, rank, world)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
