"""Keep causal prefix memory while replaying only the moving workspace tail."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from loop_jev.evaluation.coarse_state import Backbone, ENDPOINT_MASS
from loop_jev.evaluation.graph_short_loop import evaluate, prepare_prompts
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import damped_update
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.trajectory import parallel_finish
from loop_jev.transition import load_loop_model


class RecordKV:
    def __init__(self):
        self.states = {}

    def update(self, key, value, layer_idx):
        self.states[layer_idx] = key, value
        return key, value


class ReplayKV:
    def __init__(self, cache, positions, previous=None, offsets=None):
        self.cache = cache
        self.positions = positions
        self.previous = previous
        self.offsets = offsets

    def update(self, key, value, layer_idx):
        def inject(full, local, old):
            full = full.expand(local.size(0), -1, -1, -1)
            if old is not None:
                full = (full.float() + self.offsets[:, None, None, None] * (full.float() - old.float())).to(full.dtype)
            indices = self.positions[:, None, :, None].expand_as(local)
            return full.scatter(2, indices, local)

        k, v = self.cache.states[layer_idx]
        old_k, old_v = self.previous.states[layer_idx] if self.previous else (None, None)
        return inject(k, key, old_k), inject(v, value, old_v)


def tail_positions(lengths, seq_length, tail):
    tail = min(tail, seq_length)
    return (lengths - tail).clamp_min(0)[:, None] + torch.arange(tail, device=lengths.device)[None]


def select(state, positions):
    return state.gather(1, positions[..., None].expand(-1, -1, state.size(-1)))


class TailReplay:
    def __init__(self, width, *, warm=False, extrapolate=False):
        self.width = width
        self.warm = warm
        self.extrapolate = extrapolate

    @staticmethod
    def block(base, state, rope, mask, cache):
        for layer in base.model.layers[9:18]:
            state = layer(state, attention_mask=mask, position_embeddings=rope, past_key_values=cache, use_cache=False)
        return state

    def predict(self, base, ids, lengths):
        h0, context = base.prefix(ids)
        pos = tail_positions(lengths, ids.size(1), self.width)
        indices = pos[..., None].expand(-1, -1, h0.size(-1))
        full_pos, rope, _ = context
        tail_rope = tuple(select(r, pos) for r in rope)
        mask = (full_pos[:, None, None, :] <= pos[:, None, :, None])
        first = RecordKV()
        f0 = self.block(base, h0, rope, None, first)
        h1 = damped_update(h0, f0, 10, .1)
        h0_tail, h1_tail = select(h0, pos), select(h1, pos)
        if self.warm:
            second = RecordKV()
            f1 = self.block(base, h1, rope, None, second)
            h2 = damped_update(h1, f1, 10, .1)
            h2_tail = select(h2, pos)
            background = h2.float() + (1 - .9**8) * (f1.float() - h2.float())
        else:
            second = first
            f1_tail = self.block(base, h1_tail, tail_rope, mask, ReplayKV(first, pos))
            h2_tail = damped_update(h1_tail, f1_tail, 10, .1)
            background = torch.lerp(h0.float(), f0.float(), ENDPOINT_MASS)
        offsets = 1.5 + 4 * torch.arange(2, device=h0.device, dtype=torch.float32) if self.extrapolate else None

        def run(states):
            context_cache = ReplayKV(second, pos, first if self.extrapolate else None, offsets)
            return self.block(base, states, tail_rope, mask, context_cache)

        tail = parallel_finish((h0_tail, h1_tail, h2_tail), run, rounds=8, step=.1, nodes=2, sweeps=1)
        return base.score(background.to(h0.dtype).scatter(1, indices, tail), context, lengths).argmax(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', nargs='+', choices=('native', 'tail32', 'tail48', 'warm32', 'warm48', 'kv32', 'teacher'),
                   default=('native', 'tail32', 'tail48', 'warm32', 'kv32'))
    args = p.parse_args()
    model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={'': 0},
                            attn_implementation='sdpa', trust_remote_code=False).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    marker = marker_token_id(tokenizer)
    validate_checkpoint(model.config, marker)
    labels = [tokenizer.encode(x, add_special_tokens=False) for x in 'ABCD']
    if any(len(x) != 1 for x in labels):
        raise ValueError('Expected single-token labels')
    base = Backbone(model, [x[0] for x in labels])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompts = prepare_prompts(read_rows(args.test), lambda r: render_prompt(tokenizer, r, model.config.latent_slots,
                              marker), pad, torch.device('cuda:0'))
    results = {}
    for mode in args.modes:
        if mode in ('native', 'teacher'):
            result = evaluate(base, prompts, mode, eager=False)
        else:
            core = TailReplay(int(mode[-2:]), warm=not mode.startswith('tail'), extrapolate=mode.startswith('kv'))
            result = evaluate(base, prompts, 'time-nodes', eager=False, core=core)
        results[mode] = result
        print(mode, result, flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(model=args.model, input=str(args.test), device=torch.cuda.get_device_name(),
            batch_size=1, bucket=16, results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
