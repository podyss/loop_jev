"""Run two real steps, then complete the trajectory in a small learned subspace."""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from loop_jev.evaluation.coarse_state import Backbone, pack, state_objective
from loop_jev.evaluation.graph_short_loop import evaluate, prepare_prompts
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import damped_update
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.transition import load_loop_model


def complete_pair(h0, h1, h2):
    x0, x1, x2 = h0.float(), h1.float(), h2.float()
    velocity = x2 - x1
    acceleration = velocity - (x1 - x0)
    rms = lambda x: x.square().mean(-1, keepdim=True).clamp_min(.0001).sqrt()
    scale = rms(velocity)
    # Match the weighted endpoint of a constant raw block output after step two.
    anchor = x2 + (9 * (1 - .9**8)) * velocity
    return x2, velocity, acceleration, scale, anchor


class TwoStepSurrogate(nn.Module):
    def __init__(self, hidden, width):
        super().__init__()
        self.project = nn.Linear(3 * hidden, width, bias=False)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.mix = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, 4 * width, bias=False)
        self.up = nn.Linear(2 * width, width, bias=False)
        self.correction = nn.Linear(width, hidden + 2, bias=False)
        nn.init.zeros_(self.correction.weight)

    def forward(self, h0, h1, h2, mask):
        current, v, a, scale, anchor = complete_pair(h0, h1, h2)
        features = torch.cat((F.rms_norm(current, (current.size(-1),)), v / scale, a / scale), dim=-1)
        x = self.project(features.to(h2.dtype))
        b, n, w = x.shape
        q, k, value = (p.reshape(b, n, 8, w // 8).transpose(1, 2) for p in self.qkv(x).chunk(3, -1))
        attended = F.scaled_dot_product_attention(q, k, value, attn_mask=mask, is_causal=mask is None)
        x = x + self.mix(attended.transpose(1, 2).reshape(b, n, w))
        gate, value = self.gate(x).chunk(2, -1)
        linear_v, linear_a, residual = self.correction(x + self.up(F.silu(gate) * value)).float().split(
            (1, 1, current.size(-1)), dim=-1)
        return (anchor + linear_v * v + linear_a * a + scale * residual).to(h2.dtype)

    def predict(self, base, ids, lengths):
        origin, first, context = base.inputs(ids)
        h1 = damped_update(origin, first, 10, .1)
        h2 = damped_update(h1, base.stack(h1, context, 9, 18), 10, .1)
        return base.score(self(origin, h1, h2, None), context, lengths).argmax(-1)


@torch.no_grad()
def capture(base, examples, pad, device):
    records = []
    for offset in range(0, len(examples), 4):
        prompts, gold = zip(*examples[offset:offset + 4], strict=True)
        ids, lengths, valid = pack(prompts, pad, device)
        h0, first, context = base.inputs(ids, valid)
        h1 = damped_update(h0, first, 10, .1)
        h2 = damped_update(h1, base.stack(h1, context, 9, 18), 10, .1)
        state = h2
        for _ in range(8):
            state = damped_update(state, base.stack(state, context, 9, 18), 10, .1)
        scores = base.score(state, context, lengths)
        records.append(tuple(t.cpu() for t in (h0, h1, h2, state, scores, lengths, valid)) + (torch.tensor(gold),))
        if (offset // 4 + 1) % 64 == 0:
            print(f'cached {offset // 4 + 1}/{len(examples) // 4}', flush=True)
    return records


def train(args, base, render, pad, device):
    pool = read_rows(args.train)
    if (len(pool) < 2048 or len({r['id'] for r in pool}) != len(pool) or
            any(r['split'] != 'train' or len(r['options']) != 4 or r['gold_index'] not in range(4) for r in pool)):
        raise ValueError('Expected unique GSM-MC training rows')
    seen = {r['id'] for r in random.Random(42).sample(pool, 1024)}
    chosen = random.Random(546).sample([r for r in pool if r['id'] not in seen], 1024)
    examples = sorted(((render(r), r['gold_index']) for r in chosen), key=lambda e: len(e[0]))
    records = capture(base, examples, pad, device)
    torch.manual_seed(43)
    core = TwoStepSurrogate(base.model.config.hidden_size, args.width).to(device)
    optimizer = torch.optim.AdamW(core.parameters(), lr=3e-4, fused=True)
    order, rng = list(range(len(records))), random.Random(43)
    history = []
    for epoch in range(args.epochs):
        losses = torch.zeros(3, device=device)
        rng.shuffle(order)
        for i, index in enumerate(order, 1):
            h0, h1, h2, target, teacher, lengths, valid, gold = (t.to(device) for t in records[index])
            context = base.context(h2, valid)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = core(h0, h1, h2, context[-1])
                logits = base.score(pred, context, lengths)
            state = state_objective(pred, target, h2, valid)
            kl = 4 * F.kl_div((logits / 2).log_softmax(-1), (teacher / 2).softmax(-1), reduction='batchmean')
            ce = F.cross_entropy(logits, gold)
            ((state + kl + .35 * ce) / 4).backward()
            if i % 4 == 0:
                nn.utils.clip_grad_norm_(core.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses += torch.stack((state.detach(), kl.detach(), ce.detach()))
        history.append(dict(zip(('state', 'kl', 'ce'), (losses / len(records)).tolist(), strict=True)))
        print(f'epoch {epoch + 1}: {history[-1]}', flush=True)
    torch.save(dict(weights=core.cpu().state_dict(), width=args.width, history=history,
                    teacher=str(Path(args.model).resolve()), train_examples=1024), args.output / 'adapter.pt')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train', type=Path, default=Path('data/gsm-mc/train.jsonl'))
    p.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    p.add_argument('--adapter', type=Path)
    p.add_argument('--width', type=int, default=384)
    p.add_argument('--epochs', type=int, default=5)
    args = p.parse_args()
    if args.width < 8 or args.width % 8 or args.epochs < 1:
        p.error('Expected positive epochs and a width divisible by eight')
    device = torch.device('cuda:0')
    model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={'': 0},
                            attn_implementation='sdpa', trust_remote_code=False).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    marker = marker_token_id(tokenizer)
    validate_checkpoint(model.config, marker)
    labels = [tokenizer.encode(x, add_special_tokens=False) for x in 'ABCD']
    if any(len(x) != 1 for x in labels):
        raise ValueError('Expected one token per label')
    base = Backbone(model, [x[0] for x in labels])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def render(row):
        return render_prompt(tokenizer, row, model.config.latent_slots, marker)

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.adapter:
        train(args, base, render, pad, device)
        return
    saved = torch.load(args.adapter, map_location='cpu', weights_only=True)
    if saved['teacher'] != str(Path(args.model).resolve()):
        raise ValueError('Adapter does not match the teacher')
    prompts = prepare_prompts(read_rows(args.test), render, pad, device)
    core = TwoStepSurrogate(model.config.hidden_size, saved['width']).to(device, dtype=torch.bfloat16).eval()
    results = {'constant-raw': evaluate(base, prompts, 'time-nodes', eager=False, core=core)}
    core.load_state_dict(saved['weights'])
    results['two-step-surrogate'] = evaluate(base, prompts, 'time-nodes', eager=False, core=core)
    print(results, flush=True)
    (args.output / 'test.json').write_text(json.dumps(dict(model=args.model, adapter=str(args.adapter),
        input=str(args.test), device=torch.cuda.get_device_name(), batch_size=1, bucket=16, results=results),
        indent=2) + '\n')


if __name__ == '__main__':
    main()
