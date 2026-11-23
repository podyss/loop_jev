"""Predict intermediate activations for two temporal quadrature nodes in parallel."""

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
from loop_jev.evaluation.time_fold import install, merge_projections
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import damped_update
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.transition import load_loop_model


WEIGHTS = tuple(.1 * .9**(9 - r) for r in range(10))
GROUPS = (range(1, 4), range(4, 10))
MASSES = tuple(sum(WEIGHTS[r] for r in group) for group in GROUPS)
TIMES = tuple(sum(r * WEIGHTS[r] for r in group) / mass for group, mass in zip(GROUPS, MASSES, strict=True))


class TimeNodes(nn.Module):
    def __init__(self, hidden, width, cut):
        super().__init__()
        self.cut = cut
        self.input = nn.Linear(3 * hidden, width, bias=False)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.mix = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, 4 * width, bias=False)
        self.reduce = nn.Linear(2 * width, width, bias=False)
        self.outputs = nn.Linear(width, 2 * hidden, bias=False)
        nn.init.zeros_(self.outputs.weight)

    def forward(self, before, middle, first, mask):
        h = before.size(-1)
        delta = first.float() - before.float()
        scale = delta.square().mean(-1, keepdim=True).clamp_min(.01).sqrt()
        inputs = torch.cat((F.rms_norm(before.float(), (h,)), F.rms_norm(middle.float(), (h,)), delta / scale), -1)
        x = self.input(inputs.to(before.dtype))
        b, n, w = x.shape
        q, k, v = (t.reshape(b, n, 8, w // 8).transpose(1, 2) for t in self.qkv(x).chunk(3, -1))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        x = x + self.mix(attended.transpose(1, 2).reshape(b, n, w))
        gate, value = self.gate(x).chunk(2, -1)
        corrections = self.outputs(x + self.reduce(F.silu(gate) * value)).chunk(2, -1)
        return tuple((middle.float() + (1 - .9**t) * delta + scale * corr.float()).to(before.dtype)
                     for t, corr in zip(TIMES, corrections, strict=True))

    def advance(self, base, before, middle, first, context):
        nodes = self(before, middle, first, context[-1])
        positions, rope, mask = context
        if mask is not None:
            mask = torch.cat((mask, mask), 0)
        estimates = base.stack(torch.cat(nodes), (positions, rope, mask), self.cut, 18).chunk(2, 0)
        h0 = before.float()
        result = h0 + WEIGHTS[0] * (first.float() - h0)
        for mass, estimate in zip(MASSES, estimates, strict=True):
            result = result + mass * (estimate.float() - h0)
        return result.to(before.dtype), nodes

    def predict(self, base, ids, lengths):
        before, context = base.prefix(ids)
        middle = base.stack(before, context, 9, self.cut)
        first = base.stack(middle, context, self.cut, 18)
        state, _ = self.advance(base, before, middle, first, context)
        return base.score(state, context, lengths).argmax(-1)


@torch.no_grad()
def capture(base, examples, cut, pad, device):
    records = []
    needed = {int(t) + offset for t in TIMES for offset in (0, 1)}
    for start in range(0, len(examples), 4):
        prompts, gold = zip(*examples[start:start + 4], strict=True)
        ids, lengths, valid = pack(prompts, pad, device)
        before, context = base.prefix(ids, valid)
        state, snapshots = before, {}
        for r in range(10):
            middle = base.stack(state, context, 9, cut)
            raw = base.stack(middle, context, cut, 18)
            if r == 0:
                initial, first = middle, raw
            if r in needed:
                snapshots[r] = middle.float()
            state = damped_update(state, raw, 10, .1)
        nodes = [torch.lerp(snapshots[int(t)], snapshots[int(t) + 1], t % 1).to(before.dtype) for t in TIMES]
        logits = base.score(state, context, lengths)
        records.append(tuple(x.cpu() for x in (before, initial, first, *nodes, state, logits, lengths, valid))
                       + (torch.tensor(gold),))
        if (start // 4 + 1) % 32 == 0:
            print(f'{device}: captured {start // 4 + 1}/{len(examples) // 4} batches', flush=True)
    return records


def train(args, base, examples, pad, device, rank, world):
    records = capture(base, examples[rank::world], args.cut, pad, device)
    torch.manual_seed(43)
    core = TimeNodes(base.model.config.hidden_size, args.width, args.cut).to(device)
    suffix_start, suffix_end = 18, len(base.model.layers)
    suffix = install(base, args.suffix_rank, suffix_start, suffix_end) if args.suffix_rank else []
    params = list(core.parameters())
    suffix_params = [p for layer in suffix for p in (layer.down, layer.up)]
    groups = [{'params': params}]
    if suffix_params:
        groups.append({'params': suffix_params, 'lr': 1e-4})
    optimizer = torch.optim.AdamW(groups, lr=3e-4, fused=True)
    params = params + suffix_params
    accumulation = 4 // world
    rng, order = random.Random(43), list(range(len(records)))
    history = []
    for epoch in range(args.epochs):
        totals = torch.zeros(4, device=device)
        rng.shuffle(order)
        for i, index in enumerate(order, 1):
            before, initial, first, near, far, end, teacher, lengths, valid, gold = [x.to(device) for x in records[index]]
            context = base.context(before, valid)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                state, (early, late) = core.advance(base, before, initial, first, context)
                logits = base.score(state, context, lengths)
            stage_loss = (state_objective(early, near, initial, valid) + state_objective(late, far, initial, valid)) / 2
            end_loss = state_objective(state, end, before, valid)
            kl = 4 * F.kl_div((logits / 2).log_softmax(-1), (teacher / 2).softmax(-1), reduction='batchmean')
            ce = F.cross_entropy(logits, gold)
            ((stage_loss + .5 * end_loss + kl + .35 * ce) / accumulation).backward()
            if i % accumulation == 0:
                if world > 1:
                    reduce_gradients(params, world)
                nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            totals += torch.stack((stage_loss.detach(), end_loss.detach(), kl.detach(), ce.detach()))
        totals = totals.cpu()
        if world > 1:
            dist.all_reduce(totals)
        metrics = dict(zip(('nodes', 'end', 'kl', 'ce'), (totals / (world * len(records))).tolist(), strict=True))
        history.append(metrics)
        if rank == 0:
            print(f'epoch {epoch + 1}: {metrics}', flush=True)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save(dict(weights=core.cpu().state_dict(), cut=args.cut, width=args.width, history=history,
                        suffix_rank=args.suffix_rank,
                        suffix=[(m.down.detach().cpu(), m.up.detach().cpu()) for m in suffix],
                        teacher=str(Path(args.model).resolve())), args.output / 'adapter.pt')


@torch.no_grad()
def benchmark(args, base, render, pad, device):
    prompts = prepare_prompts(read_rows(args.test), render, pad, device)
    saved = torch.load(args.adapter, map_location='cpu', weights_only=True)
    if saved['teacher'] != str(Path(args.model).resolve()):
        raise ValueError('Adapter does not match the teacher')
    if saved.get('suffix_rank'):
        merge_projections(base, saved['suffix_rank'], saved['suffix'], 18, len(base.model.layers))
    core = TimeNodes(base.model.config.hidden_size, saved['width'], saved['cut']).to(device, torch.bfloat16).eval()
    core.load_state_dict(saved['weights'])
    result = evaluate(base, prompts, 'time-nodes', eager=False, core=core)
    print(result, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'test.json').write_text(json.dumps(dict(model=args.model, adapter=str(args.adapter),
        input=str(args.test), device=torch.cuda.get_device_name(), batch_size=1, bucket=16, cut=saved['cut'],
        suffix_rank=saved.get('suffix_rank', 0),
        results={'time-nodes': result}), indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--train', type=Path, default=Path('data/gsm-mc/train.jsonl'))
    p.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    p.add_argument('--cut', type=int, default=15)
    p.add_argument('--width', type=int, default=384)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--suffix-rank', type=int, default=0, help='Co-train the readout stack via mergeable rank-r updates')
    args = p.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if (args.cut not in range(10, 18) or args.width < 8 or args.width % 8 or args.epochs < 1 or args.suffix_rank < 0
            or world not in (1, 2) or (args.adapter and world != 1)):
        p.error('Invalid training options or distributed latency benchmark')
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
            pool = read_rows(args.train)
            if (len(pool) < 1024 or len({row['id'] for row in pool}) != len(pool) or
                    any(row['split'] != 'train' or len(row['options']) != 4 or row['gold_index'] not in range(4) for row in pool)):
                raise ValueError('Expected at least 1024 unique GSM-MC training rows')
            chosen = random.Random(42).sample(pool, 1024)
            examples = sorted(((render(row), row['gold_index']) for row in chosen), key=lambda entry: len(entry[0]))
            train(args, base, examples, pad, device, rank, world)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
