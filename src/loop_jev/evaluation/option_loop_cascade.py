"""Measure when the real ten-round trajectory improves a single-pass option scorer."""

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from loop_jev.evaluation.coarse_state import Backbone
from loop_jev.evaluation.graph_short_loop import build_graphs, pack_prompts
from loop_jev.evaluation.option_trie import NumericReadout, make_prompt
from loop_jev.evaluation.short_loop import validate_test_rows
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.transition import load_loop_model


MIX = .75
BUDGETS = (.025, .05, .1, .2)
WEIGHTS = (.0, .125, .25, .5, 1., 2., 4.)


def fuse(fast, slow, weight):
    return fast + weight * slow


def fast_scores(numeric, labels):
    # Keep the absolute numeric likelihood available to the router.
    fused = fuse(numeric, labels, MIX)
    order = fused.argsort(dim=-1, descending=True)
    parts = [source.gather(-1, order) for source in (numeric, labels, fused)]
    return fused, torch.cat((parts[0][..., :1], *(x[..., :1] - x[..., 1:] for x in parts)), dim=-1)


class FastReadout(NumericReadout):
    def predict(self, base, ids, meta):
        fused, features = fast_scores(*self.scores(base, ids, meta))
        return torch.cat((fused, features))


class LoopReadout:
    def __init__(self, recurrent):
        self.recurrent = recurrent

    def predict(self, base, ids, lengths):
        before, first, context = base.inputs(ids)
        exit_state = base.teacher(before, first, context) if self.recurrent else first
        return base.score(exit_state, context, lengths).squeeze(0)


def replay(graphs, prompt):
    ids, meta, _ = prompt
    graph, static_ids, static_meta, result = graphs[(tuple(ids.shape), tuple(meta.shape))]
    static_ids.copy_(ids)
    static_meta.copy_(meta)
    graph.replay()
    return result


def gather(base, core, prompts):
    return torch.stack([core.predict(base, ids, meta) for ids, meta, _ in prompts])


def choose_weight(fast, slow, gold):
    return max(WEIGHTS, key=lambda w: (fuse(fast, slow, w).argmax(-1) == gold).sum().item())


def fit_router(features, gain):
    center = features.mean(0)
    scale = features.std(0).clamp_min(.05)
    x = ((features - center) / scale).clamp(-5, 5)
    penalty = .1 * x.size(0) * torch.eye(x.size(1), device=x.device)
    coef = torch.linalg.solve(x.T @ x + penalty, x.T @ (gain - gain.mean()))
    return center, scale, coef


def route_score(features, router):
    center, scale, coef = router
    return ((features - center) / scale).clamp(-5, 5) @ coef


class RoutedReadout(FastReadout):
    def __init__(self, model, router, thresholds):
        super().__init__(model, joint=True)
        self.router = router
        self.thresholds = thresholds

    def predict(self, base, ids, meta):
        output = super().predict(base, ids, meta)
        scores, features = output[:4], output[4:]
        mask = (route_score(features, self.router) > self.thresholds).long()
        bits = (mask * (1 << torch.arange(mask.numel(), device=mask.device))).sum()
        return torch.cat((scores, (scores.argmax() + 4 * bits).float().reshape(1)))


def benchmark(fast, slow, mode, weight, *, fast_prompts, slow_prompts, index=None):
    def run(quick, full):
        fast_logits = replay(fast, quick) if mode != 'slow-only' else None
        called = True
        if mode == 'adaptive':
            flag = int(fast_logits[4].item())
            if not (flag // 4) & (1 << index):
                return flag % 4, False
        elif mode == 'fast-only':
            return int(fast_logits[4].item()) % 4, False
        slow_logits = replay(slow, full)
        if mode != 'slow-only':
            slow_logits = fuse(fast_logits[:4], slow_logits, weight)
        return int(slow_logits.argmax().item()), called

    pairs = list(zip(fast_prompts, slow_prompts, strict=True))
    for quick, full in pairs[:20]:
        run(quick, full)
    correct, calls, samples = 0, 0, []
    for quick, full in pairs:
        start = time.perf_counter()
        guess, called = run(quick, full)
        samples.append((time.perf_counter() - start) * 1000)
        correct += guess == quick[2]
        calls += called
    n = len(samples)
    return dict(correct=correct, total=n, accuracy=correct / n, wall_ms_mean=statistics.mean(samples),
                wall_ms_p50=statistics.median(samples), wall_ms_p95=sorted(samples)[math.ceil(.95 * n) - 1],
                second_pass_rate=calls / n)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    p.add_argument('--train', type=Path, default=Path('data/gsm-mc/train.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-control-weight', type=float, help='Benchmark a non-recurrent fusion at a fixed weight')
    args = p.parse_args()
    rows = read_rows(args.test)
    validate_test_rows(rows)
    model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa',
                            trust_remote_code=False).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    marker = marker_token_id(tokenizer)
    validate_checkpoint(model.config, marker)
    labels = [tokenizer.encode(x, add_special_tokens=False) for x in 'ABCD']
    if any(len(x) != 1 for x in labels):
        raise ValueError('Expected single-token labels')
    base = Backbone(model, [x[0] for x in labels])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = torch.device('cuda:0')

    def prompts(rows):
        fast = [make_prompt(tokenizer, row, pad, device, config=model.config, layout='label-compact-noslots')
                for row in rows]
        slow = pack_prompts(rows, lambda row: render_prompt(tokenizer, row, model.config.latent_slots, marker),
                            pad, device)
        return fast, slow

    native_core, teacher_core = LoopReadout(False), LoopReadout(True)
    if args.native_control_weight is not None:
        fast_prompts, slow_prompts = prompts(rows)
        fast = build_graphs(base, fast_prompts, 'time-nodes', FastReadout(model, joint=True))
        native = build_graphs(base, slow_prompts, 'time-nodes', native_core)
        torch.cuda.synchronize()
        result = benchmark(fast, native, 'fusion', args.native_control_weight,
                           fast_prompts=fast_prompts, slow_prompts=slow_prompts)
        result['slow_weight'] = args.native_control_weight
        print('fixed-weight-native', result, flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(model=args.model, input=str(args.test),
            device=torch.cuda.get_device_name(), batch_size=1, bucket=16,
            results={'fixed-weight-native': result}), indent=2) + '\n')
        return

    train = read_rows(args.train)
    if any(r['split'] != 'train' for r in train) or len({r['id'] for r in train}) != len(train):
        raise ValueError('Expected unique training examples')
    seen = {r['id'] for r in random.Random(42).sample(train, 1024)}
    legacy = {r['id'] for r in random.Random(442).sample([r for r in train if r['id'] not in seen], 512)}
    sample = random.Random(721).sample([r for r in train if r['id'] not in seen | legacy], 1536)
    train_fast, train_slow = prompts(sample)
    fast_logits = gather(base, FastReadout(model, joint=True), train_fast)
    print('Collected fast scores', flush=True)
    native_logits = gather(base, native_core, train_slow)
    teacher_logits = gather(base, teacher_core, train_slow)
    print('Collected matched native and ten-round scores', flush=True)
    scores, features = fast_logits[:, :4], fast_logits[:, 4:]
    gold = torch.tensor([r['gold_index'] for r in sample], device=device)
    fit = slice(0, 1024)
    calib = slice(1024, None)
    weights = [choose_weight(scores[fit], branch[fit], gold[fit]) for branch in (native_logits, teacher_logits)]
    teacher_weight = weights[1]
    gain = ((fuse(scores[fit], teacher_logits[fit], teacher_weight).argmax(-1) == gold[fit]).float() -
            (scores[fit].argmax(-1) == gold[fit]).float())
    router = fit_router(features[fit], gain)
    quantiles = torch.tensor([1 - b for b in BUDGETS], device=device)
    thresholds = torch.quantile(route_score(features[calib], router), quantiles)
    print('Fit fusion weights:', weights, 'router thresholds:', thresholds.tolist(), flush=True)

    fast_prompts, slow_prompts = prompts(rows)
    fast = build_graphs(base, fast_prompts, 'time-nodes', RoutedReadout(model, router, thresholds))
    native = build_graphs(base, slow_prompts, 'time-nodes', native_core)
    teacher = build_graphs(base, slow_prompts, 'time-nodes', teacher_core)
    torch.cuda.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    experiments = [('fast-only', 'fast-only', teacher, 0., None),
                   ('canonical-native', 'slow-only', native, 0., None),
                   ('canonical-teacher', 'slow-only', teacher, 0., None),
                   ('full-fusion-native', 'fusion', native, weights[0], None),
                   ('full-fusion-teacher', 'fusion', teacher, teacher_weight, None)]
    experiments += [(f'adaptive-{b:g}', 'adaptive', teacher, teacher_weight, i) for i, b in enumerate(BUDGETS)]
    for name, mode, graphs, weight, index in experiments:
        result = benchmark(fast, graphs, mode, weight, fast_prompts=fast_prompts, slow_prompts=slow_prompts,
                           index=index)
        result['slow_weight'] = weight
        results[name] = result
        print(name, result, flush=True)
        args.output.write_text(json.dumps(dict(model=args.model, input=str(args.test),
            device=torch.cuda.get_device_name(), batch_size=1, bucket=16, results=results,
            fast_mix=MIX, router_thresholds=thresholds.tolist()), indent=2) + '\n')


if __name__ == '__main__':
    main()
