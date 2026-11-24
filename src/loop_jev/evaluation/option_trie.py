"""Score numeric answer strings in a single causal, prefix-sharing forward."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.utils.hub import cached_file

from loop_jev.data.shared_train import digest, shared_rows
from loop_jev.dual_readout import DualReadoutMix, MIX_NETWORK_FILENAME
from loop_jev.evaluation.coarse_state import Backbone
from loop_jev.evaluation.distributed_eval import eval_rank, eval_shard, eval_world_size, evaluation_device
from loop_jev.evaluation.graph_short_loop import evaluate
from loop_jev.evaluation.short_loop import validate_test_rows
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.transition import load_loop_model


LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def make_trie(prompt, options, pad):
    if not 1 <= len(options) <= len(LABELS) or not all(options):
        raise ValueError('Expected 1 to 26 nonempty candidate strings')
    token_ids, positions = list(prompt), list(range(len(prompt)))
    branches, parents, children = [(1 << len(options)) - 1] * len(prompt), [-1] * len(prompt), {}
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
    return (token_ids + [pad] * (padded - size), positions + [0] * (padded - size),
            branches + [0] * (padded - size), edges, edge_count, len(prompt))


def trie_mask(positions, branches):
    earlier = positions[:, None, :, None] >= positions[:, None, None, :]
    shared = (branches[:, None, :, None].bitwise_and(branches[:, None, None, :])) != 0
    return earlier & shared


def trie_edges(token_ids, branches, meta, option_count=4):
    offsets = torch.arange(meta.size(1) - 3, device=meta.device)
    live = offsets < meta[:, :1]
    child = (meta[:, 1:2] + offsets).clamp(max=token_ids.size(1) - 1)
    tokens = token_ids.gather(1, child)
    membership = branches.gather(1, child)[..., None].bitwise_and(
        1 << torch.arange(option_count, device=meta.device)) != 0
    return tokens, live, membership & live[..., None]


def numeric_token_logprobs(logits, tokens):
    logits = logits.float()
    return logits.gather(-1, tokens[..., None]).squeeze(-1) - logits.logsumexp(-1)


def mix_head_positions(meta):
    # Last token of the numeric prompt and last token of the letter prompt.
    return torch.stack((meta[..., 1] - 1, meta[..., 2]), dim=-1)


class NumericReadout:
    def __init__(self, model, *, local=False, average=False, joint=False, mix=0.0, mix_network=None,
                 mix_head=None, execute_loops=False):
        self.head = model.lm_head.weight
        self.local = local
        self.average = average
        self.joint = joint
        self.mix = mix
        self.mix_network = mix_network
        self.mix_head = mix_head
        self.execute_loops = execute_loops

    def encode(self, base, packed):
        token_ids, positions, branches = packed.split(1, dim=0)
        embedding = base.model.embed_tokens(token_ids)
        mask = trie_mask(positions, branches)
        rope = base.model.rotary_emb(embedding, positions)
        context = positions, rope, mask
        if not self.execute_loops:
            return base.stack(embedding, context, 0, len(base.model.layers))

        config = base.model.config
        start = config.loop_start_layer
        end = config.loop_end_layer
        rounds = config.loop_count
        if (getattr(config, 'loop_schedule', 'block') != 'block' or rounds < 2
                or not getattr(config, 'loop_damping', False)
                or getattr(config, 'loop_transition_rank', 0)
                or getattr(config, 'loop_carry_workspace', False)
                or getattr(config, 'loop_evidence_reread', False)
                or getattr(config, 'loop_staged_handoff', False)
                or getattr(config, 'loop_multiscale_workspace', None)):
            raise ValueError('Trie loop execution requires a plain damped block-loop checkpoint')
        if not 0 <= start < end <= len(base.model.layers):
            raise ValueError('Invalid loop block in model config')

        state = base.stack(embedding, context, 0, start)
        step = config.loop_damping_step or 1 / rounds
        for _ in range(rounds):
            block_output = base.stack(state, context, start, end)
            state = state + step * (block_output - state)
        return base.stack(state, context, end, len(base.model.layers))

    def scores(self, base, packed, meta):
        return self.read(base, packed, meta, self.encode(base, packed))

    def read(self, base, packed, meta, state):
        token_ids, _, branches = packed.split(1, dim=0)
        parent = meta[3:]
        tokens, live, active = trie_edges(token_ids, branches, meta[None], base.readout.size(0))
        tokens, live, active = tokens[0], live[0], active[0]
        query = base.model.norm(state[0].index_select(0, parent))
        if self.local:
            raw = F.linear(query, self.head.index_select(0, tokens)).float()
            siblings = (parent[:, None] == parent[None, :]) & live[None, :]
            normalizer = raw.masked_fill(~siblings, -1e6).logsumexp(-1)
            log_prob = raw.diagonal() - normalizer
        else:
            log_prob = numeric_token_logprobs(F.linear(query, self.head), tokens)
        scores = (log_prob[:, None] * active).sum(0)
        if self.average:
            scores = scores / active.sum(0).clamp_min(1)
        if self.joint:
            label_state = state[0].index_select(0, meta[2:3])
            labels = F.linear(base.model.norm(label_state), base.readout).float().squeeze(0)
        else:
            labels = torch.zeros_like(scores)
        return scores.masked_fill(~active.any(0), float('-inf')), labels

    def predict(self, base, packed, meta):
        state = self.encode(base, packed)
        numeric, labels = self.read(base, packed, meta, state)
        if self.mix_head is not None:
            mix = self.mix_head(base.model.norm(state[0].index_select(0, mix_head_positions(meta))))
        elif self.mix_network is not None:
            mix = self.mix_network(numeric, labels)
        else:
            mix = self.mix
        return combine(numeric, labels, mix).argmax()


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
    letter_branch = branches[0] + 1
    branches = ([b | letter_branch for b in branches[:shared]] + branches[shared:stop]
                + [letter_branch] * len(canonical))
    positions = positions[:stop] + list(range(shared, shared + len(canonical)))
    width = -len(ids) % 16
    return ids + [pad] * width, positions + [0] * width, branches + [0] * width, len(ids) - 1


def compact_label_prompt(tokenizer, row, config, *, no_slots):
    # Re-encode the whole user turn so boundary merges match the numeric branch.
    options_text = '\n'.join(f'({label}) {option}' for label, option in zip(LABELS, row['options']))
    turn = [{'role': 'user', 'content': row['question'] + '\nOptions:\n' + options_text}]
    prompt = tokenizer.apply_chat_template(turn, tokenize=True, return_dict=False,
                                           add_generation_prompt=True, enable_thinking=False)
    if not no_slots:
        prompt += [config.latent_marker_token_id] * config.latent_slots
    return prompt + tokenizer.encode('Answer: (', add_special_tokens=False)


def make_prompt(tokenizer, row, pad, device, *, config=None, layout='joint'):
    if not 1 <= len(row['options']) <= len(LABELS) or not 0 <= row['gold_index'] < len(row['options']):
        raise ValueError(f"Invalid option count or target for {row['id']}")
    no_slots = layout.endswith('-noslots')
    canonical = None
    if layout == 'joint-noslots':
        canonical = render_prompt(tokenizer, row)
    elif config is not None and layout.startswith('label-compact'):
        canonical = compact_label_prompt(tokenizer, row, config, no_slots=no_slots)
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
    if eval_world_size() > 1:
        with torch.inference_mode(False):
            totals = torch.tensor(counts)
            dist.all_reduce(totals)
            counts = totals.tolist()
    return max(zip(candidates, counts, strict=True), key=lambda pair: pair[1])[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    p.add_argument('--allow-variable-options', action='store_true',
                   help='Accept a custom test JSONL with 1 to 26 candidates per question')
    p.add_argument('--train', type=Path, default=Path('data/gsm-mc/train.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', nargs='+', choices=('full', 'average', 'local', 'joint', 'joint-noslots', 'shared-options',
                                                'compact', 'compact-noslots', 'label-compact', 'label-compact-noslots'),
                   default=('local', 'full', 'average'))
    p.add_argument('--adapter', type=Path, help='Optional merged time-fold checkpoint')
    p.add_argument('--shared-train', action='store_true', help='Calibrate on the exact 1024 training rows')
    p.add_argument('--mix', type=float,
                   help='Use a fixed weight instead of the saved joint-noslots mix network or calibration')
    p.add_argument('--execute-loops', action='store_true',
                   help='Execute all configured damped block-loop rounds before the dual readout')
    args = p.parse_args()
    if args.mix is not None and (not math.isfinite(args.mix) or args.mix < 0):
        p.error('--mix must be a finite nonnegative number')
    with evaluation_device() as device:
        run(args, device)


def run(args, device):
    model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={'': device.index},
                            attn_implementation='sdpa', trust_remote_code=False).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    labels = LABELS if args.allow_variable_options else LABELS[:4]
    base = Backbone(model, [tokenizer.encode(x, add_special_tokens=False)[0] for x in labels])
    if args.adapter:
        from loop_jev.evaluation.time_fold import merge_projections

        saved = torch.load(args.adapter, map_location='cpu', weights_only=True)
        if saved['teacher'] != str(Path(args.model).resolve()):
            raise ValueError('Adapter does not match the backbone')
        merge_projections(base, saved['rank'], saved['weights'])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows = read_rows(args.test)
    if args.allow_variable_options:
        if not rows or any(not 1 <= len(row['options']) <= len(LABELS)
                           or not 0 <= row['gold_index'] < len(row['options']) for row in rows):
            raise ValueError('Expected a nonempty test set with 1 to 26 options and valid targets')
    else:
        validate_test_rows(rows)
    rows = eval_shard(rows)
    mix_network = None
    mix_head = None
    mix_initial = getattr(model.config, 'dual_readout_mix_network_initial', None)
    if args.mix is None and 'joint-noslots' in args.modes:
        mix_head = getattr(model, 'dual_readout_mix_head', None)
        if mix_head is None and mix_initial is not None:
            if not math.isfinite(mix_initial) or mix_initial <= 0:
                raise ValueError('Checkpoint mix network initial weight must be finite and positive')
            weights = cached_file(args.model, MIX_NETWORK_FILENAME)
            mix_network = DualReadoutMix(mix_initial)
            mix_network.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
            mix_network.to(device).eval().requires_grad_(False)

    results = {}
    for mode in args.modes:
        core = NumericReadout(model, local=mode == 'local', average=mode == 'average',
                              joint=mode not in ('full', 'average', 'local'),
                              mix_network=mix_network if mode == 'joint-noslots' else None,
                              mix_head=mix_head if mode == 'joint-noslots' else None,
                              execute_loops=args.execute_loops)

        def packed(row):
            return make_prompt(tokenizer, row, pad, device, config=model.config if core.joint else None,
                               layout=mode)

        if core.joint and args.mix is not None:
            core.mix = args.mix
            if eval_rank() == 0:
                print('Manual mix:', core.mix, flush=True)
        elif core.mix_head is not None:
            if eval_rank() == 0:
                print('Using the hidden-state mix head', flush=True)
        elif core.mix_network is not None:
            if eval_rank() == 0:
                print('Using the per-example mix network', flush=True)
        elif core.joint:
            train = read_rows(args.train)
            if args.shared_train:
                calibration = shared_rows(train)
            else:
                if (len(train) < 1536 or any(r['split'] != 'train' for r in train)
                        or len({r['id'] for r in train}) != len(train)):
                    raise ValueError('Expected a unique training pool with disjoint calibration questions')
                seen = {r['id'] for r in random.Random(42).sample(train, 1024)}
                calibration = random.Random(442).sample([r for r in train if r['id'] not in seen], 512)
            core.mix = calibrate(core, base, [packed(row) for row in eval_shard(calibration)])
            if eval_rank() == 0:
                print('Calibration mix:', core.mix, flush=True)
        current = [packed(row) for row in rows]
        results[mode] = evaluate(base, current, 'time-nodes', eager=False, core=core)
        if args.execute_loops:
            results[mode]['loop_rounds'] = model.config.loop_count
        if core.joint:
            results[mode]['mix'] = 'adaptive' if core.mix_head is not None or core.mix_network is not None else core.mix
            if core.mix_head is not None:
                results[mode]['mix_source'] = 'hidden-state'
        if eval_rank() == 0:
            print(mode, results[mode], flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(model=args.model, input=str(args.test), device=torch.cuda.get_device_name(),
                eval_world_size=eval_world_size(), batch_size=1, bucket=16, results=results,
                adapter=str(args.adapter) if args.adapter else None,
                train_sha256=digest(args.train.read_bytes()) if args.shared_train else None), indent=2) + '\n')


if __name__ == '__main__':
    main()
