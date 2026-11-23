"""Approximate recurrent block calls by reusing an anchored early-layer residual."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from loop_jev.evaluation.coarse_state import Backbone
from loop_jev.evaluation.graph_short_loop import evaluate, prepare_prompts
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import damped_update
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.trajectory import parallel_finish
from loop_jev.transition import load_loop_model


def transport(start, anchor, x):
    return (x.float() + (anchor.float() - start.float())).to(x.dtype)


class ResidualReplay:
    def __init__(self, depth, *, warm=False):
        self.cut = 18 - depth
        self.warm = warm

    def predict(self, base, ids, lengths):
        h0, context = base.prefix(ids)
        anchor = base.stack(h0, context, 9, self.cut)
        first = base.stack(anchor, context, self.cut, 18)

        def block(x):
            return base.stack(transport(h0, anchor, x), context, self.cut, 18)

        h1 = damped_update(h0, first, 10, .1)
        f1 = block(h1) if self.warm else transport(h0, first, h1)
        h2 = damped_update(h1, f1, 10, .1)
        # One parallel sweep at midpoints of the unobserved horizon.
        state = parallel_finish((h0, h1, h2), block, rounds=8, step=.1, nodes=2, sweeps=1)
        return base.score(state, context, lengths).argmax(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--modes', nargs='+', choices=('native', 'teacher', '1', '2', '3', 'warm1', 'warm2', 'warm3'),
                        default=('native', '1', '2', '3', 'warm1', 'warm2', 'warm3', 'teacher'))
    args = parser.parse_args()
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
    rows = read_rows(args.test)
    prompts = prepare_prompts(rows, lambda r: render_prompt(tokenizer, r, model.config.latent_slots, marker),
                              pad, torch.device('cuda:0'))
    results = {}
    for mode in args.modes:
        if mode in ('native', 'teacher'):
            result = evaluate(base, prompts, mode, eager=False)
        else:
            core = ResidualReplay(int(mode[-1]), warm=mode.startswith('warm'))
            result = evaluate(base, prompts, 'time-nodes', eager=False, core=core)
        results[mode] = result
        print(mode, result, flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(model=args.model, input=str(args.test), device=torch.cuda.get_device_name(),
            batch_size=1, bucket=16, results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
