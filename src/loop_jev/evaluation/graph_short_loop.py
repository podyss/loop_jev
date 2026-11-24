"""Benchmark weighted time quadrature and CUDA Graph replay on the full GSM-MC test."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from loop_jev.evaluation.coarse_state import Backbone, CoarseCorrection, collocate
from loop_jev.evaluation.distributed_eval import eval_shard, eval_world_size
from loop_jev.evaluation.short_loop import validate_test_rows
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import damped_update
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.trajectory import parallel_finish
from loop_jev.transition import load_loop_model


def time_quadrature(evaluate, before, first, *, groups, horizon=10, step=0.1):
    """Approximate the geometrically weighted sum of raw Euler block outputs."""
    if sum(groups) != horizon - 1 or any(n <= 0 for n in groups):
        raise ValueError('Groups must partition all updates after the first')
    start, decay = 1, 1 - step
    origin, direction = before.float(), first.float() - before.float()
    coefficients, masses = [], []
    for count in groups:
        weights = [step * decay ** (horizon - 1 - t) for t in range(start, start + count)]
        mass = sum(weights)
        time_point = sum(w * t for t, w in enumerate(weights, start)) / mass
        coefficients.append(1 - decay ** time_point)
        masses.append(mass)
        start += count
    states = torch.cat([origin + c * direction for c in coefficients], dim=0).to(before.dtype)
    outputs = evaluate(states)
    # Write the quadrature around H0 to avoid a separate identity coefficient.
    result = origin + step * decay ** (horizon - 1) * direction
    for output, mass in zip(outputs.split(1), masses, strict=True):
        result = result + mass * (output.float() - origin)
    return result.to(before.dtype)


def transport_layer(previous_inputs, previous_outputs, next_inputs):
    """Reuse an intermediate activation and apply an adjacent-node secant update."""
    origin = previous_inputs.float()
    previous = previous_outputs.float()
    displacement = next_inputs.float() - origin
    # Use a centered secant in the interior and a one-sided secant at the edges.
    def difference(x):
        return torch.cat((x[1:2] - x[:1], x[2:] - x[:-2], x[-1:] - x[-2:-1]), dim=0)

    dx = difference(origin)
    residual_change = difference(previous - origin)
    coefficient = ((dx * displacement).sum(-1, keepdim=True) /
                   dx.square().sum(-1, keepdim=True).clamp_min(1e-4)).clamp(-1, 1)
    return (previous + displacement + coefficient * residual_change).to(previous_outputs.dtype)


def lift_nodes(first_input, second_input, first_middle, second_middle, nodes):
    """Lift later time nodes through the early layers using two measured states."""
    secant = second_input.float() - first_input.float()
    shift = nodes.float() - second_input.float()
    fraction = ((shift * secant).sum(-1, keepdim=True) /
                secant.square().sum(-1, keepdim=True).clamp_min(1e-4)).clamp(0, 4)
    residual_change = second_middle.float() - first_middle.float() - secant
    return (second_middle.float() + shift + fraction * residual_change).to(nodes.dtype)


def predict(base, ids, lengths, mode, core=None):
    if callable(mode):
        return mode(base, ids, lengths)
    if mode == 'time-nodes':
        return core.predict(base, ids, lengths)
    if mode == 'midpoint':
        before, context = base.prefix(ids)
        state, _ = collocate(base, core, before, context)
        return base.score(state, context, lengths).argmax(-1)
    if mode == 'lift':
        before, context = base.prefix(ids)
        first_mid = base.stack(before, context, 9, 14)
        first = base.stack(first_mid, context, 14, 18)
        h1 = damped_update(before, first, 10, 0.1)
        second_mid = base.stack(h1, context, 9, 14)
        h2 = damped_update(h1, base.stack(second_mid, context, 14, 18), 10, 0.1)

        def lifted_block(nodes):
            return base.stack(lift_nodes(before, h1, first_mid, second_mid, nodes), context, 14, 18)

        state = parallel_finish((before, h1, h2), lifted_block, rounds=8, step=0.1, nodes=4, sweeps=1)
        return base.score(state, context, lengths).argmax(-1)
    before, first, context = base.inputs(ids)

    def run_block(state):
        return base.stack(state, context, 9, 18)

    if mode == 'coarse':
        state = core(before, first)
    elif mode == 'teacher':
        state = base.teacher(before, first, context)
    elif mode.startswith('quad'):
        groups = (4, 5) if mode == 'quad2' else (3, 3, 3)
        state = time_quadrature(run_block, before, first, groups=groups)
    elif mode.startswith('picard') or mode == 'reuse':
        first_step = damped_update(before, first, 10, 0.1)
        second_step = damped_update(first_step, run_block(first_step), 10, 0.1)
        if mode == 'reuse':
            previous = None

            def reuse_block(nodes):
                nonlocal previous
                if previous is None:
                    intermediate = base.stack(nodes, context, 9, 14)
                    previous = nodes, intermediate
                else:
                    intermediate = transport_layer(*previous, nodes)
                return base.stack(intermediate, context, 14, 18)

            block = reuse_block
        else:
            block = run_block
        state = parallel_finish((before, first_step, second_step), block, rounds=8, step=0.1,
                                nodes=2 if mode == 'picard-small' else 4,
                                sweeps=2 if mode in ('picard2', 'reuse') else 1)
    else:
        state = first
    return base.score(state, context, lengths).argmax(-1)


def prepare_prompts(rows, render, pad, device, bucket=16):
    validate_test_rows(rows)
    return pack_prompts(eval_shard(rows), render, pad, device, bucket)


def pack_prompts(rows, render, pad, device, bucket=16):
    if bucket < 1:
        raise ValueError('Bucket width must be positive')
    prompts = []
    for row in rows:
        ids = render(row)
        n = len(ids)
        prompts.append((torch.tensor([ids + [pad] * (-n % bucket)], device=device),
                        torch.tensor([n], device=device), row['gold_index']))
    return prompts


def build_graphs(base, prompts, mode, core=None):
    exemplars = {(tuple(ids.shape), tuple(length.shape)): (ids, length) for ids, length, _ in prompts}
    pool = torch.cuda.graph_pool_handle()
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    graphs = {}
    for shape, (ids, length) in sorted(exemplars.items(), reverse=True):
        static_ids, static_length = ids.clone(), length.clone()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            for _ in range(3):
                predict(base, static_ids, static_length, mode, core)
        torch.cuda.current_stream().wait_stream(warm_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=warm_stream):
            result = predict(base, static_ids, static_length, mode, core)
        graphs[shape] = graph, static_ids, static_length, result
    return graphs


@torch.inference_mode()
def evaluate(base, prompts, mode, *, eager, core=None):
    begin = time.perf_counter()
    graphs = {} if eager else build_graphs(base, prompts, mode, core)
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - begin

    def run(ids, length):
        if eager:
            return predict(base, ids, length, mode, core).item()
        graph, static_ids, static_length, result = graphs[tuple(ids.shape), tuple(length.shape)]
        static_ids.copy_(ids)
        static_length.copy_(length)
        graph.replay()
        return result.item()

    for ids, length, _ in prompts[:20]:
        run(ids, length)
    world = eval_world_size()
    if world > 1:
        dist.barrier()
    begin = time.perf_counter()
    correct, samples = 0, []
    for ids, length, label in prompts:
        start = time.perf_counter()
        chosen = run(ids, length)
        samples.append((time.perf_counter() - start) * 1000)
        correct += chosen == label
    seconds = time.perf_counter() - begin
    graph_count = len(graphs)
    if world > 1:
        with torch.inference_mode(False):
            gathered = [None] * world
            dist.all_gather_object(gathered, (correct, samples, graph_count, capture_seconds, seconds))
        correct = sum(result[0] for result in gathered)
        samples = [sample for result in gathered for sample in result[1]]
        graph_count = sum(result[2] for result in gathered)
        capture_seconds = max(result[3] for result in gathered)
        seconds = max(result[4] for result in gathered)
    return dict(correct=correct, total=len(samples), accuracy=correct / len(samples), graphs=graph_count,
                evaluation_seconds=seconds, examples_per_second=len(samples) / seconds,
                capture_seconds=capture_seconds, wall_ms_p50=statistics.median(samples),
                wall_ms_p95=sorted(samples)[math.ceil(.95 * len(samples)) - 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bucket', type=int, default=16)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--coarse-adapter', type=Path)
    parser.add_argument('--midpoint-adapter', type=Path)
    parser.add_argument('--quant-bits', choices=(4, 8), type=int)
    parser.add_argument('--quant-scope', choices=('all', 'shared'), default='all')
    parser.add_argument('--modes', nargs='+',
                        choices=('native', 'quad2', 'quad3', 'picard', 'picard-small', 'picard2', 'lift', 'reuse', 'teacher',
                                 'coarse', 'midpoint'),
                        default=('native', 'quad2', 'quad3', 'picard2', 'teacher'))
    args = parser.parse_args()
    if args.bucket < 1:
        parser.error('Bucket width must be positive')
    rows = read_rows(args.input)
    device = torch.device('cuda:0')
    model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={'': 0},
                            attn_implementation='sdpa', trust_remote_code=False).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    cfg = model.config
    slots = getattr(cfg, 'latent_slots', 0)
    marker = marker_token_id(tokenizer) if slots else None
    if getattr(cfg, 'loop_count', 1) == 10:
        validate_checkpoint(cfg, marker)
    elif (args.modes != ['native'] or getattr(cfg, 'loop_count', 1) != 1 or cfg.model_type != 'qwen3'
          or cfg.num_hidden_layers < 18 or any(kind != 'full_attention' for kind in cfg.layer_types)):
        parser.error('Only the native mode supports a plain single-pass Qwen3 checkpoint')
    label_tokens = [tokenizer.encode(x, add_special_tokens=False) for x in 'ABCD']
    if any(len(x) != 1 for x in label_tokens):
        raise ValueError('Expected one token per answer label')
    base = Backbone(model, [x[0] for x in label_tokens])
    if args.quant_bits:
        from loop_jev.evaluation.packed_linear import pack_backbone

        pack_backbone(base, args.quant_bits, args.quant_scope)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompts = prepare_prompts(rows, lambda r: render_prompt(tokenizer, r, slots, marker), pad, device, args.bucket)
    results = {}
    adapters = {}
    for mode in args.modes:
        core = None
        if mode in ('coarse', 'midpoint'):
            adapter = getattr(args, f'{mode}_adapter')
            if adapter is None:
                parser.error(f'{mode} requires --{mode}-adapter')
            weights = torch.load(adapter, map_location='cpu', weights_only=True)
            midpoint = mode == 'midpoint'
            if weights['teacher'] != str(Path(args.model).resolve()) or weights.get('midpoint', False) != midpoint:
                parser.error(f'Invalid {mode} adapter for the backbone')
            core = CoarseCorrection(cfg.hidden_size, weights['width'], midpoint=midpoint).to(
                device=device, dtype=torch.bfloat16).eval()
            core.load_state_dict(weights['weights'])
            adapters[mode] = str(adapter)
        results[mode] = evaluate(base, prompts, mode, eager=args.eager, core=core)
        print(mode, results[mode], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(model=args.model, input=str(args.input), batch_size=1,
        device=torch.cuda.get_device_name(), bucket=args.bucket, eager=args.eager, adapters=adapters,
        quant_bits=args.quant_bits, quant_scope=args.quant_scope if args.quant_bits else None,
        results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
