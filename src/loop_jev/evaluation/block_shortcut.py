"""Distill a ten-round recurrent teacher into one shared-block pass."""

import argparse
import json
import os
import random
from copy import deepcopy
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from loop_jev.data.shared_train import digest, shared_rows
from loop_jev.evaluation.coarse_state import Backbone, pack, reduce_gradients, state_objective
from loop_jev.evaluation.distributed_eval import eval_rank, eval_world_size
from loop_jev.evaluation.graph_short_loop import evaluate, prepare_prompts
from loop_jev.evaluation.time_fold import projections
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import damped_update
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.transition import load_loop_model


DIRECT, FIRST, SECOND = range(3)
INTERVALS = ((0, 10), (0, 5), (5, 10))
MODES = ('ce', 'endpoint', 'segments', 'shortcut')
PROGRESSIVE_MODES = ('rollout', 'progressive')
SUPPORTED_MODES = (*MODES, *PROGRESSIVE_MODES, 'consistency')
EMA_DECAY = 0.99


class IntervalLinear(nn.Module):
    def __init__(self, linear, rank, interval):
        super().__init__()
        self.linear = linear
        self.interval = interval
        options = dict(device=linear.weight.device, dtype=torch.float32)
        self.down = nn.Parameter(torch.empty(rank, linear.in_features, **options))
        self.up = nn.Parameter(torch.zeros(linear.out_features, rank, **options))
        self.gain = nn.Parameter(torch.zeros(len(INTERVALS), rank, **options))
        nn.init.kaiming_uniform_(self.down, a=5**0.5)

    def forward(self, x):
        hidden = F.linear(x, self.down)
        hidden = hidden * (1 + self.gain[self.interval[0]]).to(hidden.dtype)
        return self.linear(x) + F.linear(hidden, self.up)

    @torch.no_grad()
    def merged(self, interval):
        weight = self.linear.weight
        delta = (self.up * (1 + self.gain[interval])) @ self.down
        linear = nn.Linear(self.linear.in_features, self.linear.out_features, bias=self.linear.bias is not None,
                           device='meta', dtype=weight.dtype).requires_grad_(False)
        linear.weight = nn.Parameter((weight.float() + delta).to(weight.dtype), requires_grad=False)
        linear.bias = self.linear.bias
        return linear


class Shortcut:
    def __init__(self, base, rank):
        self.base = base
        self.interval = [DIRECT]
        self.sites = list(projections(base))
        self.layers = [IntervalLinear(getattr(group, name), rank, self.interval) for group, name in self.sites]

    def replace(self, layers):
        for (group, name), layer in zip(self.sites, layers, strict=True):
            setattr(group, name, layer)

    def restore(self):
        self.replace(layer.linear for layer in self.layers)

    def merged(self, interval):
        return [layer.merged(interval) for layer in self.layers]

    def hop(self, state, context, interval):
        self.interval[0] = interval
        return advance(self.base, state, context, interval)

    def parameters(self):
        return chain.from_iterable((layer.down, layer.up, layer.gain) for layer in self.layers)

    def save_weights(self):
        return [[tensor.detach().cpu() for tensor in (layer.down, layer.up, layer.gain)] for layer in self.layers]

    @torch.no_grad()
    def load_weights(self, weights):
        for layer, values in zip(self.layers, weights, strict=True):
            for param, value in zip((layer.down, layer.up, layer.gain), values, strict=True):
                param.copy_(value)


class MixedPrecisionLayer(nn.Module):
    def __init__(self, layer, dtype):
        super().__init__()
        self.layer = layer
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        # Keep optimizer weights in FP32 and run the entire decoder layer in BF16.
        weights = {name: param.to(self.dtype) for name, param in self.layer.named_parameters()}
        return torch.func.functional_call(self.layer, weights, args, kwargs)


class FullBlock:
    def __init__(self, base, *, trainable):
        self.base = base
        self.original = base.model.layers[9:18]
        dtype = next(self.original.parameters()).dtype
        self.block = deepcopy(self.original).requires_grad_(trainable)
        if trainable:
            self.block.float()
            self.layers = [MixedPrecisionLayer(layer, dtype) for layer in self.block]
        else:
            self.layers = self.block

    def replace(self, layers):
        for index, layer in enumerate(layers, 9):
            self.base.model.layers[index] = layer

    def restore(self):
        self.replace(self.original)

    def merged(self, interval):
        # All three intervals use the same full block and their own damping.
        return self.layers

    def hop(self, state, context, interval):
        return advance(self.base, state, context, interval)

    def parameters(self):
        return self.block.parameters()

    def save_weights(self):
        original = self.original.state_dict()
        return {name: value.to(device='cpu', dtype=original[name].dtype)
                for name, value in self.block.state_dict().items()}

    def load_weights(self, weights):
        self.block.load_state_dict(weights)


def advance(base, state, context, interval):
    start, end = INTERVALS[interval]
    return advance_rounds(base, state, context, end - start)


def advance_rounds(base, state, context, remaining):
    if remaining == 0:
        return state
    raw = base.stack(state, context, 9, 18)
    return damped_update(state, raw, 10, 1 - 0.9**remaining)


def predict_direct(base, ids, lengths):
    before, context = base.prefix(ids)
    return base.score(advance(base, before, context, DIRECT), context, lengths).argmax(-1)


def two_hop_predictor(core):
    first, second = core.merged(FIRST), core.merged(SECOND)

    def predict(base, ids, lengths):
        before, context = base.prefix(ids)
        # Swapping weights occurs at graph capture only; replay executes the two merged blocks.
        core.replace(first)
        middle = advance(base, before, context, FIRST)
        core.replace(second)
        end = advance(base, middle, context, SECOND)
        return base.score(end, context, lengths).argmax(-1)

    return predict


def teacher_states(base, before, context, *, rounds=(5, 10)):
    state = before
    states = [before] if 0 in rounds else []
    for step in range(1, 11):
        state = damped_update(state, base.stack(state, context, 9, 18), 10, 0.1)
        if step in rounds:
            states.append(state)
    return tuple(states)


@torch.no_grad()
def capture(base, vocab, examples, pad, device, *, capture_trajectory=False):
    records, trajectories = [], []
    rounds = tuple(range(11)) if capture_trajectory else (5, 10)
    for i in range(0, len(examples), 4):
        prompts, gold = zip(*examples[i:i + 4], strict=True)
        ids, lengths, valid = pack(prompts, pad, device)
        before, context = base.prefix(ids, valid)
        states = teacher_states(base, before, context, rounds=rounds)
        logits = F.linear(base.answer_state(states[-1], context, lengths), vocab)
        cached = tuple(state.cpu() for state in states)
        if capture_trajectory:
            trajectories.append(cached)
            before, middle, end = cached[0], cached[5], cached[10]
        else:
            before = before.cpu()
            middle, end = cached
        records.append((before, middle, end, logits.cpu(), lengths.cpu(), valid.cpu(), torch.tensor(gold)))
        if (i // 4 + 1) % 32 == 0:
            print(f'{device}: captured {i // 4 + 1}/{len(examples) // 4} teacher batches', flush=True)
    return records, trajectories


def kl_objective(student, teacher):
    return 4 * F.kl_div((student.float() / 2).log_softmax(-1), (teacher.float() / 2).softmax(-1),
                        reduction='batchmean')


def rollout_states(core, before, context):
    middle = core.hop(before, context, FIRST)
    return middle, core.hop(middle, context, SECOND)


def train_variant(args, mode, base, vocab, records, device, rank, world, *, trajectories=None):
    torch.manual_seed(43)
    core = FullBlock(base, trainable=True) if args.full_block else Shortcut(base, args.rank)
    ema = None
    if mode == 'consistency':
        ema = FullBlock(base, trainable=True)
        ema.block.requires_grad_(False)
        ema_params = list(ema.parameters())
    core.replace(core.layers)
    params = list(core.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, fused=True)
    accumulation = 4 // world
    rng = random.Random(43)
    order = list(range(len(records)))
    history = []
    for epoch in range(args.epochs):
        totals = torch.zeros(5, device=device)
        rng.shuffle(order)
        for i, index in enumerate(order, 1):
            before, middle, end, teacher, lengths, valid, gold = (x.to(device) for x in records[index])
            context = base.context(before, valid)
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                if ema is not None:
                    step = ((epoch * len(records) + i - 1) * world + rank) % 10
                    trajectory = trajectories[index]
                    next_state = end if step == 9 else trajectory[step + 1].to(device)
                    ema.replace(ema.layers)
                    with torch.no_grad():
                        target = advance_rounds(base, next_state, context, 9 - step)
                    core.replace(core.layers)
                if mode == 'rollout':
                    first, final = rollout_states(core, before, context)
                else:
                    final = core.hop(before, context, DIRECT)
                answer = base.answer_state(final, context, lengths)
                ce = F.cross_entropy(F.linear(answer, base.readout).float(), gold)
                state = kl = segment = consistency = ce.new_zeros(())
                if mode != 'ce':
                    kl = kl_objective(F.linear(answer, vocab), teacher)
                    state = state_objective(final, end, before, valid)
                if mode == 'rollout':
                    segment = state_objective(first, middle, before, valid)
                if mode in ('segments', 'shortcut'):
                    first = core.hop(before, context, FIRST)
                    second = core.hop(middle, context, SECOND)
                    segment = (state_objective(first, middle, before, valid) +
                               state_objective(second, end, middle, valid)) / 2
                    if mode == 'shortcut' and epoch > 0:
                        with torch.no_grad():
                            composed = core.hop(first.detach(), context, SECOND)
                        consistency = state_objective(final, composed, before, valid)
                if ema is not None:
                    current = before if step == 0 else trajectory[step].to(device)
                    prediction = final if step == 0 else advance_rounds(base, current, context, 10 - step)
                    consistency = state_objective(prediction, target, before, valid)
            loss = ce + kl + 0.5 * state + 0.5 * segment + 0.5 * consistency
            (loss / accumulation).backward()
            if i % accumulation == 0:
                if world > 1:
                    if args.full_block:
                        reductions = [dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
                                      for param in params]
                        for reduction in reductions:
                            reduction.wait()
                    else:
                        reduce_gradients(params, world)
                nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                if ema is not None:
                    with torch.no_grad():
                        torch._foreach_lerp_(ema_params, params, 1 - EMA_DECAY)
                optimizer.zero_grad(set_to_none=True)
            totals += torch.stack((ce.detach(), kl.detach(), state.detach(), segment.detach(), consistency.detach()))
        if world > 1:
            if not args.full_block:
                totals = totals.cpu()
            dist.all_reduce(totals)
        metrics = dict(zip(('ce', 'kl', 'state', 'segment', 'consistency'),
                           (totals / (world * len(records))).tolist(), strict=True))
        history.append(metrics)
        if rank == 0:
            print(f'{mode} epoch {epoch + 1}/{args.epochs}: {metrics}', flush=True)
    core.restore()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output) as temporary:
            checkpoint = Path(temporary) / f'{mode}.pt'
            trained = core if ema is None else ema
            metadata = {} if ema is None else dict(ema_decay=EMA_DECAY)
            torch.save(dict(mode=mode, teacher=str(Path(args.model).resolve()), weights=trained.save_weights(),
                            full_block=args.full_block, rank=None if args.full_block else args.rank,
                            epochs=args.epochs, examples=args.examples, lr=args.lr,
                            train_split=args.train_split, train_sha256=digest(args.train.read_bytes()),
                            parameters=sum(p.numel() for p in params), history=history, **metadata), checkpoint)
            checkpoint.replace(args.output / checkpoint.name)
    return core


def choose_train_rows(pool, examples, split):
    if split == 'fixed':
        if examples != len(pool):
            raise ValueError('Fixed training uses every row in the shared file')
        return shared_rows(pool)
    if split == 'original':
        return random.Random(42).sample(pool, examples)
    seen = {row['id'] for row in random.Random(42).sample(pool, 1024)}
    return random.Random(927).sample([row for row in pool if row['id'] not in seen], examples)


@torch.no_grad()
def capture_rollout(base, core, vocab, records, device):
    first, second = core.merged(FIRST), core.merged(SECOND)
    staged = []
    try:
        for before, _, _, _, lengths, valid, gold in records:
            h0, n, mask = (x.to(device) for x in (before, lengths, valid))
            context = base.context(h0, mask)
            core.replace(first)
            middle = advance(base, h0, context, FIRST)
            core.replace(second)
            end = advance(base, middle, context, SECOND)
            logits = F.linear(base.answer_state(end, context, n), vocab)
            staged.append((before, middle.cpu(), end.cpu(), logits.cpu(), lengths, valid, gold))
    finally:
        core.restore()
    print(f'{device}: cached {len(staged)} frozen two-hop batches', flush=True)
    return staged


def train(args, base, vocab, render, pad, device, rank, world):
    pool = read_rows(args.train)
    if (world not in (1, 2, 4) or args.examples % 16 or not 0 < args.examples <= len(pool)
            or len({row['id'] for row in pool}) != len(pool)
            or any(row['split'] != 'train' or len(row['options']) != 4
                   or row['gold_index'] not in range(4) for row in pool)):
        raise ValueError('Expected unique GSM-MC training rows and a complete effective batch of 16')
    chosen = choose_train_rows(pool, args.examples, args.train_split)
    examples = sorted(((render(row), row['gold_index']) for row in chosen), key=lambda item: len(item[0]))
    records, trajectories = capture(base, vocab, examples[rank::world], pad, device,
                                    capture_trajectory='consistency' in args.modes)
    bridge = None
    for mode in args.modes:
        targets = records if mode != 'progressive' else capture_rollout(base, bridge, vocab, records, device)
        core = train_variant(args, mode, base, vocab, targets, device, rank, world, trajectories=trajectories)
        if mode == 'rollout':
            bridge = core
        del core


@torch.inference_mode()
def benchmark(args, base, render, pad, device, *, base_ce_result=None):
    prompts = prepare_prompts(read_rows(args.test), render, pad, device)
    train_sha256 = digest(args.train.read_bytes()) if args.train_split == 'fixed' else None
    results = {}
    for mode in ('native', 'teacher', *args.modes):
        if mode in SUPPORTED_MODES:
            saved = torch.load(args.output / f'{mode}.pt', map_location='cpu', weights_only=True)
            if saved['teacher'] != str(Path(args.model).resolve()) or saved['mode'] != mode:
                raise ValueError(f'Mismatched {mode} checkpoint')
            if saved.get('full_block', False) != args.full_block:
                raise ValueError(f'{mode} checkpoint does not match the --full-block setting')
            if train_sha256 is not None and (saved.get('train_sha256') != train_sha256
                                             or saved.get('train_split') != 'fixed'):
                raise ValueError(f'{mode} checkpoint was not trained on the shared training sample')
            core = FullBlock(base, trainable=False) if args.full_block else Shortcut(base, saved['rank'])
            core.load_weights(saved['weights'])
            try:
                if mode == 'rollout':
                    result = evaluate(base, prompts, two_hop_predictor(core), eager=False)
                else:
                    core.replace(core.merged(DIRECT))
                    result = evaluate(base, prompts, predict_direct, eager=False)
            finally:
                core.restore()
        else:
            result = evaluate(base, prompts, mode, eager=False)
        results[mode] = result
        if eval_rank() == 0:
            print(mode, result, flush=True)
        if mode in ('segments', 'shortcut'):
            try:
                two_hop = evaluate(base, prompts, two_hop_predictor(core), eager=False)
            finally:
                core.restore()
            results[f'{mode}-two-hop'] = two_hop
            if eval_rank() == 0:
                print(f'{mode}-two-hop', two_hop, flush=True)
        if mode in SUPPORTED_MODES:
            del saved, core
    if eval_rank() != 0:
        return
    metadata = {}
    if base_ce_result is not None:
        results['base-ce'] = base_ce_result
        metadata['base_ce_model'] = str(args.base_ce_model)
    (args.output / 'test.json').write_text(json.dumps(dict(model=args.model, input=str(args.test), batch_size=1,
        bucket=16, dtype='bfloat16', cuda_graph=True, device=torch.cuda.get_device_name(),
        eval_world_size=eval_world_size(),
        train_sha256=train_sha256, full_block=args.full_block, results=results, **metadata),
        indent=2) + '\n')


def load_backbone(path, device, *, looped):
    model = load_loop_model(path, dtype=torch.bfloat16, device_map={'': device.index},
                            attn_implementation='sdpa', trust_remote_code=False).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    if looped:
        marker = marker_token_id(tokenizer)
        validate_checkpoint(model.config, marker)
        slots = model.config.latent_slots
    else:
        if getattr(model.config, 'loop_count', 1) != 1 or getattr(model.config, 'latent_slots', 0):
            raise ValueError('Expected a single-pass CE checkpoint without latent slots')
        marker, slots = None, 0
    encoded = [tokenizer.encode(label, add_special_tokens=False) for label in 'ABCD']
    if any(len(x) != 1 for x in encoded):
        raise ValueError('Expected single-token labels')
    base = Backbone(model, [x[0] for x in encoded])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def render(row):
        return render_prompt(tokenizer, row, slots, marker)

    return base, model.lm_head.weight, render, pad


def benchmark_base_ce(args, device):
    metadata = json.loads((args.base_ce_model / 'experiment.json').read_text(encoding='utf-8'))
    if metadata.get('status') != 'complete' or metadata.get('teacher') is not None:
        raise ValueError('Expected a completed CE checkpoint trained without a teacher')
    if args.train_split == 'fixed' and metadata.get('train_sha256') != digest(args.train.read_bytes()):
        raise ValueError('CE checkpoint was not trained on the shared training file')
    base, _, render, pad = load_backbone(args.base_ce_model, device, looped=False)
    prompts = prepare_prompts(read_rows(args.test), render, pad, device)
    result = evaluate(base, prompts, 'native', eager=False)
    if eval_rank() == 0:
        print('base-ce', result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', type=Path, default=Path('data/gsm-mc/train.jsonl'))
    parser.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    parser.add_argument('--examples', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=4e-4)
    tuning = parser.add_mutually_exclusive_group()
    tuning.add_argument('--rank', type=int, default=16)
    tuning.add_argument('--full-block', action='store_true', help='Fine-tune all nine shared layers.')
    parser.add_argument('--modes', choices=SUPPORTED_MODES, nargs='+', default=MODES)
    parser.add_argument('--train-split', choices=('original', 'disjoint', 'fixed'), default='original')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--base-ce-model', type=Path, help='Single-pass CE checkpoint to include in evaluation.')
    args = parser.parse_args()
    if args.base_ce_model is not None and not args.evaluate:
        parser.error('--base-ce-model requires --evaluate')
    if 'consistency' in args.modes and not args.full_block:
        parser.error('consistency requires --full-block')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if (args.rank < 1 or args.epochs < 1 or args.lr <= 0
            or len(set(args.modes)) != len(args.modes)
            or (not args.evaluate and 'progressive' in args.modes and
                ('rollout' not in args.modes or args.modes.index('rollout') > args.modes.index('progressive')))
            or (not args.evaluate and 'shortcut' in args.modes and args.epochs < 2)):
        parser.error('Invalid training options')
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if args.full_block and not args.evaluate else 'gloo')
    try:
        base_ce_result = benchmark_base_ce(args, device) if args.base_ce_model is not None else None
        base, vocab, render, pad = load_backbone(args.model, device, looped=True)
        if args.evaluate:
            benchmark(args, base, render, pad, device, base_ce_result=base_ce_result)
        else:
            train(args, base, vocab, render, pad, device, rank, world)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
