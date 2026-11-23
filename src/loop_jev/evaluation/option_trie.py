"""Score numeric answer strings in a single causal, prefix-sharing forward."""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from loop_jev.data.shared_train import digest, shared_rows
from loop_jev.evaluation.coarse_state import Backbone
from loop_jev.evaluation.graph_short_loop import evaluate
from loop_jev.evaluation.short_loop import validate_test_rows
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.transition import load_loop_model


def make_trie(prompt, options, pad):
    token_ids, positions = list(prompt), list(range(len(prompt)))
    branches, parents, children = [15] * len(prompt), [-1] * len(prompt), {}
    for option, tokens in enumerate(options):
        parent = len(prompt) - 1
        for token in tokens:
            pair = (parent, token)
            if pair not in children:
                children[pair] = len(token_ids)
                token_ids.append(token)
                positions.append(positions[parent] + 1)
                branches.append(0)
                parents.append(parent)
            parent = children[pair]
            branches[parent] |= 1 << option
    size = len(token_ids)
    padded = size + (-size % 16)
    # Only trie edges require scores. Pack their parents into a short fixed-size suffix.
    edges = parents[len(prompt):]
    edge_count = len(edges)
    if len(options) != 4 or not all(options):
        raise ValueError('Expected four nonempty candidate strings')
    return (token_ids + [pad] * (padded - size), positions + [0] * (padded - size),
            branches + [0] * (padded - size), edges, edge_count, len(prompt))


def trie_mask(positions, branches):
    earlier = positions[:, None, :, None] >= positions[:, None, None, :]
    shared = (branches[:, None, :, None].bitwise_and(branches[:, None, None, :])) != 0
    return earlier & shared


class NumericReadout:
    def __init__(self, model, *, local=False, average=False, joint=False, mix=0.0):
        self.head = model.lm_head.weight
        self.local = local
        self.average = average
        self.joint = joint
        self.mix = mix

    def encode(self, base, packed):
        token_ids, positions, branches = packed.split(1, dim=0)
        embedding = base.model.embed_tokens(token_ids)
        mask = trie_mask(positions, branches)
        rope = base.model.rotary_emb(embedding, positions)
        return base.stack(embedding, (positions, rope, mask), 0, len(base.model.layers))

    def scores(self, base, packed, meta):
        return self.read(base, packed, meta, self.encode(base, packed))

    def read(self, base, packed, meta, state):
        token_ids, _, branches = packed.split(1, dim=0)
        n = token_ids.size(1)
        parent = meta[3:]
        offsets = torch.arange(parent.numel(), device=packed.device)
        live = offsets < meta[0]
        child = (meta[1] + offsets).clamp(max=n - 1)
        query = base.model.norm(state[0].index_select(0, parent))
        tokens = token_ids[0].index_select(0, child)
        if self.local:
            raw = F.linear(query, self.head.index_select(0, tokens)).float()
            siblings = (parent[:, None] == parent[None, :]) & live[None, :]
            normalizer = raw.masked_fill(~siblings, -1e6).logsumexp(-1)
            log_prob = raw.diagonal() - normalizer
        else:
            raw = F.linear(query, self.head).float()
            log_prob = raw.gather(1, tokens[:, None]).squeeze(1) - raw.logsumexp(-1)
        membership = branches[0].index_select(0, child)[:, None].bitwise_and(
            1 << torch.arange(4, device=packed.device)[None]) != 0
        active = membership & live[:, None]
        scores = (log_prob[:, None] * active).sum(0)
        if self.average:
            scores = scores / active.sum(0).clamp_min(1)
        if self.joint:
            label_state = state[0].index_select(0, meta[2:3])
            labels = F.linear(base.model.norm(label_state), base.readout).float().squeeze(0)
        else:
            labels = torch.zeros_like(scores)
        return scores, labels

    def predict(self, base, packed, meta):
        numeric, labels = self.scores(base, packed, meta)
        return combine(numeric, labels, self.mix).argmax()


def combine(numeric, labels, mix):
    return numeric + mix * labels


def join_canonical(ids, positions, branches, prefix_len, count, canonical, pad):
    shared = 0
    for a, b in zip(ids[:prefix_len], canonical):
        if a != b:
            break
        shared += 1
    stop = prefix_len + count
    canonical = canonical[shared:]
    if not canonical:
        raise ValueError('Expected a distinct canonical continuation')
    ids = ids[:stop] + canonical
    branches = [b | 16 for b in branches[:shared]] + branches[shared:stop] + [16] * len(canonical)
    positions = positions[:stop] + list(range(shared, shared + len(canonical)))
    width = -len(ids) % 16
    return ids + [pad] * width, positions + [0] * width, branches + [0] * width, len(ids) - 1


def make_prompt(tokenizer, row, pad, device, *, config=None, layout='joint'):
    no_slots = layout.endswith('-noslots')
    canonical = None
    if config is not None and layout.startswith('label-compact'):
        # Re-encode the whole user turn so boundary merges match the numeric branch.
        options_text = '\n'.join(f'({label}) {option}' for label, option in zip('ABCD', row['options'], strict=True))
        turn = [{'role': 'user', 'content': row['question'] + '\nOptions:\n' + options_text}]
        canonical = tokenizer.apply_chat_template(turn, tokenize=True, return_dict=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        canonical += [config.latent_marker_token_id] * (0 if no_slots else config.latent_slots)
        canonical += tokenizer.encode('Answer: (', add_special_tokens=False)
    elif config is not None:
        canonical = render_prompt(tokenizer, row, 0 if no_slots else config.latent_slots,
                                  config.latent_marker_token_id)
    if layout == 'shared-options':
        answer_width = len(tokenizer.encode('Answer: (', add_special_tokens=False))
        prompt = canonical[:-answer_width]
    else:
        question = 'Question: ' + row['question'] if layout.startswith('compact') else row['question']
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question +
            '\nGive only the final numerical answer.'}], tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False)
    prompt += tokenizer.encode('Answer:', add_special_tokens=False)
    options = [tokenizer.encode(' ' + str(option) + '\n', add_special_tokens=False) for option in row['options']]
    ids, positions, branches, parents, count, prefix_len = make_trie(prompt, options, pad)
    label_position = 0
    if canonical is not None:
        ids, positions, branches, label_position = join_canonical(ids, positions, branches, prefix_len, count,
                                                                  canonical, pad)
    parents += [0] * (-count % 16)
    return (torch.tensor([ids, positions, branches], device=device),
            torch.tensor([count, prefix_len, label_position, *parents], device=device), row['gold_index'])


@torch.inference_mode()
def calibrate(core, base, prompts):
    scores, gold = [], []
    for packed, lengths, label in prompts:
        scores.append(core.scores(base, packed, lengths))
        gold.append(label)
    numeric, labels = (torch.stack(x) for x in zip(*scores, strict=True))
    gold = torch.tensor(gold, device=numeric.device)
    candidates = (0., .25, .5, .75, 1., 1.5, 2., 3.)
    counts = [(combine(numeric, labels, w).argmax(-1) == gold).sum().item() for w in candidates]
    return max(zip(candidates, counts, strict=True), key=lambda pair: pair[1])[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    p.add_argument('--train', type=Path, default=Path('data/gsm-mc/train.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', nargs='+', choices=('full', 'average', 'local', 'joint', 'shared-options',
                                                'compact', 'compact-noslots', 'label-compact', 'label-compact-noslots'),
                   default=('local', 'full', 'average'))
    p.add_argument('--adapter', type=Path, help='Optional merged time-fold checkpoint')
    p.add_argument('--shared-train', action='store_true', help='Calibrate on the exact 1024 training rows')
    args = p.parse_args()
    model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={'': 0},
                            attn_implementation='sdpa', trust_remote_code=False).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    base = Backbone(model, [tokenizer.encode(x, add_special_tokens=False)[0] for x in 'ABCD'])
    if args.adapter:
        from loop_jev.evaluation.time_fold import merge_projections

        saved = torch.load(args.adapter, map_location='cpu', weights_only=True)
        if saved['teacher'] != str(Path(args.model).resolve()):
            raise ValueError('Adapter does not match the backbone')
        merge_projections(base, saved['rank'], saved['weights'])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows = read_rows(args.test)
    validate_test_rows(rows)
    results = {}
    for mode in args.modes:
        core = NumericReadout(model, local=mode == 'local', average=mode == 'average',
                              joint=mode not in ('full', 'average', 'local'))

        def packed(row):
            return make_prompt(tokenizer, row, pad, torch.device('cuda:0'), config=model.config if core.joint else None,
                               layout=mode)

        if core.joint:
            train = read_rows(args.train)
            if args.shared_train:
                calibration = shared_rows(train)
            else:
                if (len(train) < 1536 or any(r['split'] != 'train' for r in train)
                        or len({r['id'] for r in train}) != len(train)):
                    raise ValueError('Expected a unique training pool with disjoint calibration questions')
                seen = {r['id'] for r in random.Random(42).sample(train, 1024)}
                calibration = random.Random(442).sample([r for r in train if r['id'] not in seen], 512)
            core.mix = calibrate(core, base, [packed(row) for row in calibration])
            print('Calibration mix:', core.mix, flush=True)
        current = [packed(row) for row in rows]
        results[mode] = evaluate(base, current, 'time-nodes', eager=False, core=core)
        if core.joint:
            results[mode]['mix'] = core.mix
        print(mode, results[mode], flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(model=args.model, input=str(args.test), device=torch.cuda.get_device_name(),
            batch_size=1, bucket=16, results=results, adapter=str(args.adapter) if args.adapter else None,
            train_sha256=digest(args.train.read_bytes()) if args.shared_train else None), indent=2) + '\n')


if __name__ == '__main__':
    main()
