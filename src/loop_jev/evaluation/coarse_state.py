"""Replace nine shared-block calls with one learned full-sequence correction."""

import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from loop_jev.evaluation.short_loop import validate_test_rows
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import damped_update
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.transition import load_loop_model


ENDPOINT_MASS = 1 - 0.9**10
WEIGHTED_TIME = sum(i * 0.1 * 0.9**(9 - i) for i in range(10)) / ENDPOINT_MASS


class CoarseCorrection(nn.Module):
    def __init__(self, hidden, width=256, *, midpoint=False):
        super().__init__()
        self.down = nn.Linear(hidden if midpoint else 2 * hidden, width, bias=False)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.mix = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, 4 * width, bias=False)
        self.up = nn.Linear(2 * width, width, bias=False)
        self.decode = nn.Linear(width, hidden, bias=False)
        nn.init.zeros_(self.decode.weight)

    def forward(self, before, after, causal_mask=None):
        anchor = before.float()
        features = F.rms_norm(anchor, (before.size(-1),))
        if after is None:
            scale = anchor.square().mean(-1, keepdim=True).clamp_min(1e-6).sqrt()
        else:
            delta = after.float() - anchor
            scale = delta.square().mean(-1, keepdim=True).clamp_min(1e-6).sqrt()
            anchor = anchor + 0.45 * delta
            features = torch.cat((features, delta / scale), -1)
        x = self.down(features.to(before.dtype))
        b, n, w = x.shape
        q, k, v = (t.reshape(b, n, 4, w // 4).transpose(1, 2) for t in self.qkv(x).chunk(3, -1))
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, is_causal=causal_mask is None)
        x = x + self.mix(h.transpose(1, 2).reshape(b, n, w))
        gate, value = self.gate(x).chunk(2, -1)
        return (anchor + scale * self.decode(x + self.up(F.silu(gate) * value)).float()).to(before.dtype)


class Backbone:
    def __init__(self, model, labels):
        self.model = model.model
        self.readout = model.lm_head.weight[labels].detach()

    def context(self, state, valid=None):
        n = state.size(1)
        positions = torch.arange(n, device=state.device)[None]
        mask = None
        if valid is not None:
            causal = positions[:, :, None] >= positions[:, None, :]
            mask = (causal & valid[:, None, :])[:, None]
        return positions, self.model.rotary_emb(state, positions), mask

    def stack(self, state, context, lo, hi):
        positions, rope, mask = context
        for i in range(lo, hi):
            state = self.model.layers[i](state, attention_mask=mask, position_ids=positions,
                                         position_embeddings=rope, use_cache=False)
        return state

    def prefix(self, ids, valid=None):
        embedding = self.model.embed_tokens(ids)
        context = self.context(embedding, valid)
        return self.stack(embedding, context, 0, 9), context

    def inputs(self, ids, valid=None):
        before, context = self.prefix(ids, valid)
        return before, self.stack(before, context, 9, 18), context

    def answer_state(self, state, context, lengths=None):
        state = self.stack(state, context, 18, len(self.model.layers))
        answer = state[:, -1] if lengths is None else state.gather(
            1, (lengths - 1)[:, None, None].expand(-1, 1, state.size(-1))).squeeze(1)
        return self.model.norm(answer)

    def score(self, state, context, lengths=None):
        return F.linear(self.answer_state(state, context, lengths), self.readout).float()

    def teacher(self, before, after, context, *, midpoint=False):
        state = before
        lower = math.floor(WEIGHTED_TIME)
        stages = []
        for step in range(10):
            if step:
                after = self.stack(state, context, 9, 18)
            state = damped_update(state, after, 10, 0.1)
            if midpoint and step + 1 in (lower, lower + 1):
                stages.append(state)
        if midpoint:
            stage = torch.lerp(stages[0].float(), stages[1].float(), WEIGHTED_TIME - lower).to(before.dtype)
            return state, stage
        return state


def collocate(base, core, before, context):
    stage = core(before, None, context[-1])
    raw = base.stack(stage, context, 9, 18)
    return torch.lerp(before.float(), raw.float(), ENDPOINT_MASS).to(before.dtype), stage


def pack(prompts, pad, device):
    lengths = torch.tensor([len(p) for p in prompts], device=device)
    width = max(map(len, prompts))
    ids = torch.tensor([p + [pad] * (width - len(p)) for p in prompts], device=device)
    valid = torch.arange(width, device=device)[None] < lengths[:, None]
    return ids, lengths, valid


@torch.no_grad()
def capture(base, batches, pad, device, *, midpoint=False):
    records = []
    for i, (prompts, labels) in enumerate(batches):
        ids, lengths, valid = pack(prompts, pad, device)
        before, after, context = base.inputs(ids, valid)
        if midpoint:
            target, after = base.teacher(before, after, context, midpoint=True)
        else:
            target = base.teacher(before, after, context)
        scores = base.score(target, context, lengths)
        records.append(tuple(t.cpu() for t in (before, after, target, scores, lengths, valid)) + (labels,))
        if (i + 1) % 32 == 0:
            print(f"captured {i + 1}/{len(batches)} batches on {device}", flush=True)
    return records


def state_objective(predicted, target, before, valid):
    scale = (target.float() - before.float()).square().mean(-1, keepdim=True).clamp_min(0.01).sqrt()
    error = F.smooth_l1_loss((predicted.float() - target.float()) / scale, torch.zeros_like(scale).expand_as(predicted),
                             reduction="none").mean(-1)
    return (error * valid).sum() / valid.sum()


def reduce_gradients(parameters, world):
    # The adapter is small; CPU collectives avoid querying the failed GPU via NCCL.
    gradients = [p.grad for p in parameters]
    flat = torch.cat([g.flatten() for g in gradients]).cpu()
    dist.all_reduce(flat)
    averaged = (flat / world).to(gradients[0].device)
    for grad, value in zip(gradients, averaged.split([g.numel() for g in gradients]), strict=True):
        grad.copy_(value.view_as(grad))


def train(args, model, base, tokenizer, device, rank, world):
    pool = read_rows(args.train)
    if world not in (1, 2, 4):
        raise ValueError('Expected one, two or four training ranks')
    accumulation = 4 // world
    if (len({row['id'] for row in pool}) != len(pool) or any(
            row['split'] != 'train' or len(row['options']) != 4 or row['gold_index'] not in range(4) for row in pool)
            or not 0 < args.examples <= len(pool) or args.examples % (4 * world * accumulation)):
        raise ValueError("Expected unique training questions and full batches for accumulation")
    chosen = random.Random(42).sample(pool, args.examples)
    cfg = model.config
    prompts = sorted([(render_prompt(tokenizer, r, cfg.latent_slots, cfg.latent_marker_token_id), r['gold_index'])
                      for r in chosen], key=lambda x: len(x[0]))
    local = prompts[rank::world]
    batches = [(list(ids), torch.tensor(gold)) for start in range(0, len(local), 4)
               for ids, gold in [zip(*local[start:start + 4], strict=True)]]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    data = capture(base, batches, pad, device, midpoint=args.midpoint)
    torch.manual_seed(43)
    core = CoarseCorrection(cfg.hidden_size, args.width, midpoint=args.midpoint).to(device)
    optimizer = torch.optim.AdamW(core.parameters(), lr=3e-4, fused=True)
    rng = random.Random(43)
    indices = list(range(len(data)))
    history = []
    for epoch in range(args.epochs):
        rng.shuffle(indices)
        totals = torch.zeros(3, device=device)
        optimizer.zero_grad(set_to_none=True)
        for i, idx in enumerate(indices):
            before, after, target, teacher, lengths, valid, gold = [t.to(device) for t in data[idx]]
            context = base.context(before, valid)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if args.midpoint:
                    pred, stage = collocate(base, core, before, context)
                else:
                    pred = core(before, after, context[-1])
                logits = base.score(pred, context, lengths)
            state = state_objective(pred, target, before, valid)
            if args.midpoint:
                state = state + state_objective(stage, after, before, valid)
            kl = 4 * F.kl_div((logits / 2).log_softmax(-1), (teacher / 2).softmax(-1), reduction='batchmean')
            ce = F.cross_entropy(logits, gold)
            ((0.5 * state + kl + 0.35 * ce) / accumulation).backward()
            if (i + 1) % accumulation == 0:
                if world > 1:
                    reduce_gradients(core.parameters(), world)
                nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            totals += torch.stack((state.detach(), kl.detach(), ce.detach()))
        totals = totals.cpu()
        if world > 1:
            dist.all_reduce(totals)
        metrics = dict(zip(('state', 'kl', 'ce'), (totals / (world * len(data))).tolist(), strict=True))
        history.append(metrics)
        if rank == 0:
            print(f"epoch {epoch + 1}: {metrics}", flush=True)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save(dict(weights=core.cpu().state_dict(), width=args.width, teacher=str(Path(args.model).resolve()),
                        examples=args.examples, history=history, midpoint=args.midpoint), args.output / 'adapter.pt')


@torch.inference_mode()
def evaluate(args, model, base, tokenizer, device, rows):
    cfg = model.config
    saved = torch.load(args.adapter, map_location='cpu', weights_only=True)
    if saved['teacher'] != str(Path(args.model).resolve()):
        raise ValueError('Adapter does not match this backbone')
    midpoint = saved.get('midpoint', False)
    core = CoarseCorrection(cfg.hidden_size, saved['width'], midpoint=midpoint).to(
        device=device, dtype=torch.bfloat16).eval()
    core.load_state_dict(saved['weights'])
    prompts = [(torch.tensor([render_prompt(tokenizer, r, cfg.latent_slots, cfg.latent_marker_token_id)],
                            device=device), r['gold_index']) for r in rows]
    results = {}
    modes = ('native', 'anchor', 'midpoint' if midpoint else 'coarse')
    for mode in modes:
        def predict(ids):
            if mode == 'midpoint':
                before, context = base.prefix(ids)
                after, _ = collocate(base, core, before, context)
            else:
                before, after, context = base.inputs(ids)
                if mode == 'anchor':
                    after = (before.float() + 0.45 * (after.float() - before.float())).to(before.dtype)
                elif mode == 'coarse':
                    after = core(before, after)
            return base.score(after, context).argmax(-1).item()

        for ids, _ in prompts[:20]:
            predict(ids)
        correct, times = 0, []
        for ids, gold in prompts:
            start = time.perf_counter()
            pred = predict(ids)
            times.append((time.perf_counter() - start) * 1000)
            correct += pred == gold
        results[mode] = dict(correct=correct, total=len(prompts), accuracy=correct / len(prompts),
                             wall_ms_p50=statistics.median(times), wall_ms_p95=sorted(times)[math.ceil(.95 * len(times))-1])
        print(mode, results[mode], flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'test.json').write_text(json.dumps(dict(model=args.model, adapter=str(args.adapter),
        parameters=sum(p.numel() for p in core.parameters()), batch_size=1, device=torch.cuda.get_device_name(),
        results=results), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', type=Path, default=Path('data/gsm-mc/train.jsonl'))
    parser.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    parser.add_argument('--adapter', type=Path, help='Evaluate this adapter instead of training')
    parser.add_argument('--examples', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--midpoint', action='store_true', help='Predict the sole block evaluation state')
    args = parser.parse_args()
    if args.epochs < 1 or args.width < 4 or args.width % 4:
        parser.error('Expected positive training epochs and a width divisible by four')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    rows = read_rows(args.test) if args.adapter else None
    if args.adapter:
        validate_test_rows(rows)
        if world != 1:
            parser.error('Latency evaluation requires one GPU')
    if world > 1:
        dist.init_process_group('gloo')
    try:
        model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={'': device.index},
                                attn_implementation='sdpa', trust_remote_code=False).eval().requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        validate_checkpoint(model.config, marker_token_id(tokenizer))
        ids = [tokenizer.encode(x, add_special_tokens=False) for x in 'ABCD']
        if any(len(x) != 1 for x in ids):
            raise ValueError('Expected one token per answer label')
        base = Backbone(model, [x[0] for x in ids])
        if args.adapter:
            evaluate(args, model, base, tokenizer, device, rows)
        else:
            train(args, model, base, tokenizer, device, rank, world)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
