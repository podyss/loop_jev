"""Use anchored feed-forward channel deltas for inexpensive trajectory replay."""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from loop_jev.evaluation.coarse_state import Backbone, pack
from loop_jev.evaluation.graph_short_loop import evaluate, prepare_prompts
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import damped_update
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.trajectory import parallel_finish
from loop_jev.transition import load_loop_model


def attention(layer, state, rope, mask):
    residual, _ = layer.self_attn(layer.input_layernorm(state), position_embeddings=rope, attention_mask=mask)
    return residual


def activation(layer, state):
    mlp = layer.mlp
    state = layer.post_attention_layernorm(state)
    return mlp.act_fn(mlp.gate_proj(state)) * mlp.up_proj(state)


@torch.inference_mode()
def calibrate(base, examples, pad, device):
    layers = base.model.layers[9:18]
    energy = [layer.mlp.down_proj.weight.new_zeros(layer.mlp.down_proj.in_features, dtype=torch.float32)
              for layer in layers]
    for start in range(0, len(examples), 4):
        ids, _, valid = pack(examples[start:start + 4], pad, device)
        origin, (_, rope, mask) = base.prefix(ids, valid)
        state, anchors = origin, []
        for r in range(10):
            hidden = state
            for i, layer in enumerate(layers):
                hidden = hidden + attention(layer, hidden, rope, mask)
                act = activation(layer, hidden)
                if r == 0:
                    anchors.append(act)
                elif r in (1, 3, 6, 9):
                    energy[i].add_(((act.float() - anchors[i].float()).square() * valid[..., None]).sum((0, 1)))
                hidden = hidden + layer.mlp.down_proj(act)
            state = damped_update(state, hidden, 10, .1)
    return [e * layer.mlp.down_proj.weight.float().square().sum(0) for e, layer in zip(energy, layers, strict=True)]


class DeltaLayer:
    def __init__(self, layer, indices):
        self.layer = layer
        self.indices = indices
        mlp = layer.mlp
        self.gate_up = torch.cat((mlp.gate_proj.weight.index_select(0, indices),
                                  mlp.up_proj.weight.index_select(0, indices)))
        self.down = mlp.down_proj.weight.index_select(1, indices).contiguous()

    def anchor(self, state, rope, mask):
        attended = attention(self.layer, state, rope, mask)
        state = state + attended
        act = activation(self.layer, state)
        mlp = self.layer.mlp.down_proj(act)
        fixed = mlp.float() - F.linear(act.index_select(-1, self.indices), self.down).float()
        return state + mlp, (attended, fixed)

    def replay(self, state, rope, mask, anchor, *, freeze_attention):
        attended, fixed = anchor
        if not freeze_attention:
            attended = attention(self.layer, state, rope, mask)
        state = state + attended
        x = self.layer.post_attention_layernorm(state)
        gate, up = F.linear(x, self.gate_up).chunk(2, -1)
        delta = F.linear(self.layer.mlp.act_fn(gate) * up, self.down)
        return state + (fixed + delta.float()).to(state.dtype)


class DeltaReplay:
    def __init__(self, base, energy, width, *, freeze_attention):
        layers = base.model.layers[9:18]
        self.layers = [DeltaLayer(layer, e.topk(width).indices.sort().values)
                       for layer, e in zip(layers, energy, strict=True)]
        self.freeze_attention = freeze_attention

    def predict(self, base, ids, lengths):
        h0, context = base.prefix(ids)
        _, rope, mask = context
        first, anchors = h0, []
        for layer in self.layers:
            first, anchor = layer.anchor(first, rope, mask)
            anchors.append(anchor)

        def block(x):
            for layer, anchor in zip(self.layers, anchors, strict=True):
                x = layer.replay(x, rope, mask, anchor, freeze_attention=self.freeze_attention)
            return x

        h1 = damped_update(h0, first, 10, .1)
        h2 = damped_update(h1, block(h1), 10, .1)
        state = parallel_finish((h0, h1, h2), block, rounds=8, step=.1, nodes=2, sweeps=1)
        return base.score(state, context, lengths).argmax(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--train', type=Path, default=Path('data/gsm-mc/train.jsonl'))
    parser.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', default=('512', '1024', 'fixed512', 'fixed1024'))
    args = parser.parse_args()
    model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={'': 0},
                            attn_implementation='sdpa', trust_remote_code=False).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    marker = marker_token_id(tokenizer)
    validate_checkpoint(model.config, marker)
    labels = [tokenizer.encode(x, add_special_tokens=False) for x in 'ABCD']
    if any(len(x) != 1 for x in labels):
        raise ValueError('Expected single-token labels')
    base, device = Backbone(model, [x[0] for x in labels]), torch.device('cuda:0')
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def render(row):
        return render_prompt(tokenizer, row, model.config.latent_slots, marker)

    pool = read_rows(args.train)
    if len(pool) < 1024 or any(r['split'] != 'train' for r in pool) or len({r['id'] for r in pool}) != len(pool):
        raise ValueError('Expected a unique training pool')
    train_ids = {r['id'] for r in random.Random(42).sample(pool, 1024)}
    calibration = random.Random(210).sample([r for r in pool if r['id'] not in train_ids], 128)
    energy = calibrate(base, sorted((render(r) for r in calibration), key=len), pad, device)
    prompts = prepare_prompts(read_rows(args.test), render, pad, device)
    results = {}
    for mode in args.modes:
        width = int(mode.removeprefix('fixed'))
        if width < 1 or width > model.config.intermediate_size:
            parser.error('Requested channel count exceeds MLP width')
        core = DeltaReplay(base, energy, width, freeze_attention=mode.startswith('fixed'))
        fractions = [(e.topk(width).values.sum() / e.sum()).item() for e in energy]
        print(mode, 'retained change energy:', fractions, flush=True)
        results[mode] = evaluate(base, prompts, 'time-nodes', eager=False, core=core)
        print(mode, results[mode], flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(model=args.model, input=str(args.test), device=torch.cuda.get_device_name(),
            calibration_examples=128, batch_size=1, bucket=16, results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
