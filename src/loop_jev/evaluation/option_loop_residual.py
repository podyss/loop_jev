"""Distill the ten-round change in option log odds into a single-pass readout."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from loop_jev.data.shared_train import digest, shared_rows
from loop_jev.evaluation.coarse_state import Backbone
from loop_jev.evaluation.distributed_eval import eval_rank, eval_shard, eval_world_size, evaluation_device, gather_rows
from loop_jev.evaluation.graph_short_loop import evaluate, pack_prompts
from loop_jev.evaluation.option_loop_cascade import MIX, LoopReadout, gather
from loop_jev.evaluation.option_trie import NumericReadout, compact_label_prompt, make_prompt
from loop_jev.evaluation.short_loop import validate_test_rows
from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.transition import load_loop_model


FEATURE_GROUPS = ('relative', 'candidate', 'label', 'scores')
ABLATIONS = {
    'full': FEATURE_GROUPS,
    'scores-only': ('scores',),
    'scores-plus-candidate': ('candidate', 'scores'),
    'scores-plus-relative': ('relative', 'scores'),
    'scores-plus-label': ('label', 'scores'),
    **{f'without-{part}': tuple(group for group in FEATURE_GROUPS if group != part) for part in FEATURE_GROUPS},
}


def feature_mask(groups, projection):
    width = projection.size(-1)
    return torch.tensor([group in groups for group in FEATURE_GROUPS], device=projection.device).repeat_interleave(
        torch.tensor([width, width, width, 3], device=projection.device), output_size=3 * width + 3)


def center(x, dim=-1):
    return x - x.mean(dim=dim, keepdim=True)


def ending_parents(packed, meta):
    offsets = torch.arange(meta.numel() - 3, device=meta.device)
    child = (meta[1] + offsets).clamp(max=packed.size(-1) - 1)
    branches = packed[2].index_select(0, child)
    members = branches[:, None].bitwise_and(1 << torch.arange(4, device=meta.device)) != 0
    last = torch.where(members & (offsets < meta[0])[:, None], offsets[:, None], 0).amax(0)
    return meta[3:].index_select(0, last)


class CandidateFeatures(NumericReadout):
    def __init__(self, model, projection, head=None, *, mask=None, mix=MIX):
        super().__init__(model, joint=True)
        self.projection = projection
        self.adapter = head
        self.mask = mask
        self.mix = mix

    def features(self, base, packed, meta):
        state = self.encode(base, packed)
        numeric, labels = self.read(base, packed, meta, state)
        indices = torch.cat((ending_parents(packed, meta), meta[2:3]))
        projected = (base.model.norm(state[0].index_select(0, indices)) @ self.projection).float()
        relative = center(projected[:4], dim=0)
        fast = numeric + self.mix * labels
        features = torch.cat((relative, projected[:4], projected[4:].expand(4, -1),
                              torch.stack((center(numeric), center(labels), numeric / 5), -1)), dim=-1)
        if self.mask is not None:
            features = features.masked_fill(~self.mask, 0)
        return fast, features

    def predict(self, base, ids, meta):
        scores, features = self.features(base, ids, meta)
        return self.adapter(scores, features).argmax(-1)


class CandidateCorrection(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def correction(self, features):
        return center(self.net(features).squeeze(-1))

    def forward(self, scores, features):
        return scores + self.correction(features)


class LabelOnlyReadout:
    def __init__(self, projection, adapter=None, *, mix):
        self.projection = projection
        self.adapter = adapter
        self.mix = mix

    def predict(self, base, ids, lengths):
        _, first, context = base.inputs(ids)
        answer = base.answer_state(first, context, lengths)
        labels = F.linear(answer, base.readout).float().squeeze(0)
        if self.adapter is None:
            return labels.argmax()

        projected = (answer @ self.projection).float().expand(4, -1)
        missing = labels.new_zeros(4)
        scores = torch.stack((missing, center(labels), missing), dim=-1)
        # The saved head expects numeric-branch features in its first two groups.
        features = torch.cat((labels.new_zeros(4, 2 * projected.size(-1)), projected, scores), dim=-1)
        return self.adapter(self.mix * labels, features).argmax()


def train_adapter(scores, features, targets, gold, validation, *, distill):
    head = CandidateCorrection(features.size(-1)).to(scores.device)
    opt = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.05)
    generator = torch.Generator(device=scores.device).manual_seed(131)
    fit_size = len(gold) if validation is None else validation.start
    best, best_key, best_epoch = None, None, None
    for epoch in range(13):
        if epoch:
            for ix in torch.randperm(fit_size, generator=generator, device=scores.device).split(32):
                change = head.correction(features[ix])
                loss = F.cross_entropy(scores[ix] + change, gold[ix])
                if distill:
                    loss = loss + .5 * F.smooth_l1_loss(change, targets[ix])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        if validation is None:
            continue
        with torch.no_grad():
            prediction = head(scores[validation], features[validation])
            hit = (prediction.argmax(-1) == gold[validation]).sum().item()
            key = hit, -F.cross_entropy(prediction, gold[validation]).item()
            if best_key is None or key > best_key:
                best = {k: v.detach().clone() for k, v in head.state_dict().items()}
                best_key, best_epoch = key, epoch
    if validation is None:
        best_epoch = epoch
    else:
        head.load_state_dict(best)
    return head.eval().requires_grad_(False), best_epoch


def fit_adapter(scores, features, targets, gold, validation, *, distill):
    if eval_rank() == 0:
        torch.manual_seed(131)
        head, epoch = train_adapter(scores, features, targets, gold, validation, distill=distill)
    if eval_world_size() == 1:
        return head, epoch
    checkpoint = [({k: v.cpu() for k, v in head.state_dict().items()}, epoch) if eval_rank() == 0 else None]
    dist.broadcast_object_list(checkpoint, src=0)
    weights, epoch = checkpoint[0]
    if eval_rank() != 0:
        head = CandidateCorrection(features.size(-1)).to(scores.device)
        head.load_state_dict(weights)
        head.eval().requires_grad_(False)
    return head, epoch


def run_ablation(args, model, base, projection, test, scores, features, gold, validation, mix, train_sha256):
    results = {}
    if eval_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in ('no-head', *ABLATIONS):
        if name == 'no-head':
            core = NumericReadout(model, joint=True, mix=mix)
            groups, epoch, params = (), None, 0
        else:
            groups = ABLATIONS[name]
            mask = None if groups == FEATURE_GROUPS else feature_mask(groups, projection)
            kept = features if mask is None else features.masked_fill(~mask, 0)
            adapter, epoch = fit_adapter(scores, kept, None, gold, validation, distill=False)
            core = CandidateFeatures(model, projection, adapter, mask=mask, mix=mix)
            params = sum(p.numel() for p in adapter.parameters())
        result = evaluate(base, test, 'time-nodes', eager=False, core=core)
        results[name] = {**result, 'features': groups, 'epoch': epoch, 'head_parameters': params}
        if eval_rank() == 0:
            print(name, results[name], flush=True)
            args.output.write_text(json.dumps(dict(model=args.model, input=str(args.test), train=str(args.train),
                train_size=1024, validation_size=0 if validation is None else 512,
                mix=mix, train_sha256=train_sha256, seed=131, batch_size=1, bucket=16,
                device=torch.cuda.get_device_name(), eval_world_size=eval_world_size(), results=results), indent=2) + '\n')


def shared_mix(report, model, train_sha256):
    if Path(report['model']).resolve() != Path(model).resolve() or report.get('train_sha256') != train_sha256:
        raise ValueError('Calibration must use the same backbone and shared training sample')
    mix = report['results']['label-compact-noslots']['mix']
    if not math.isfinite(mix) or mix < 0:
        raise ValueError('Invalid dual-readout calibration weight')
    return mix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--test', type=Path, default=Path('data/gsm-mc/test.jsonl'))
    p.add_argument('--train', type=Path, default=Path('data/gsm-mc/train.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shared-train', action='store_true', help='Fit all 1024 rows for 12 fixed epochs')
    p.add_argument('--calibration', type=Path, help='Same-sample dual-readout report, required with --shared-train')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--checkpoint', type=Path, help='Evaluate a saved candidate head without retraining')
    mode.add_argument('--ablate', action='store_true', help='Retrain answer-only heads with selected feature groups masked')
    p.add_argument('--label-only', action='store_true', help='Evaluate a saved head with only the letter readout')
    args = p.parse_args()
    if args.shared_train != (args.calibration is not None):
        p.error('--shared-train and --calibration must be used together')
    if args.label_only and not args.checkpoint:
        p.error('--label-only requires --checkpoint')
    with evaluation_device() as device:
        run(args, device)


def run(args, device):
    rows, train = read_rows(args.test), read_rows(args.train)
    validate_test_rows(rows)
    rows = eval_shard(rows)
    if any(r['split'] != 'train' for r in train) or len({r['id'] for r in train}) != len(train):
        raise ValueError('Expected unique training examples')
    train_sha256 = None
    validation = slice(1024, None)
    mix = MIX
    if args.shared_train:
        sample = shared_rows(train)
        validation = None
        train_sha256 = digest(args.train.read_bytes())
        mix = shared_mix(json.loads(args.calibration.read_text()), args.model, train_sha256)
    elif not args.checkpoint:
        seen = {r['id'] for r in random.Random(42).sample(train, 1024)}
        legacy = {r['id'] for r in random.Random(442).sample([r for r in train if r['id'] not in seen], 512)}
        sample = random.Random(721).sample([r for r in train if r['id'] not in seen | legacy], 1536)
    model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={'': device.index}, attn_implementation='sdpa',
                            trust_remote_code=False).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    marker = marker_token_id(tokenizer)
    validate_checkpoint(model.config, marker)
    labels = [tokenizer.encode(x, add_special_tokens=False) for x in 'ABCD']
    if any(len(x) != 1 for x in labels):
        raise ValueError('Expected single-token labels')
    base = Backbone(model, [x[0] for x in labels])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    generator = torch.Generator().manual_seed(883)
    width = model.config.hidden_size
    projection = (torch.randn(width, 64, generator=generator) / width**.5).to(device, dtype=torch.bfloat16)
    reader = CandidateFeatures(model, projection, mix=mix)

    def packed(row):
        return make_prompt(tokenizer, row, pad, device, config=model.config, layout='label-compact-noslots')

    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if Path(saved['model']).resolve() != Path(args.model).resolve():
            raise ValueError('Checkpoint does not match the backbone')
        if saved.get('train_sha256') != train_sha256 or saved.get('mix', MIX) != mix:
            raise ValueError('Checkpoint does not match the training sample or dual-readout calibration')
        projection = saved['projection']
        adapter = CandidateCorrection(3 * projection.size(-1) + 3).to(device)
        adapter.load_state_dict(saved['adapter'])
        if args.label_only:
            test = pack_prompts(rows, lambda row: compact_label_prompt(tokenizer, row, model.config, no_slots=True),
                                pad, device)
            cores = [('label-only', LabelOnlyReadout(projection, mix=mix)),
                     (args.checkpoint.stem, LabelOnlyReadout(projection, adapter.eval(), mix=mix))]
        else:
            test = [packed(row) for row in rows]
            cores = [('fast-only', NumericReadout(model, joint=True, mix=mix)),
                     (args.checkpoint.stem, CandidateFeatures(model, projection, adapter.eval(), mix=mix))]
        results = {}
        for name, core in cores:
            results[name] = evaluate(base, test, 'time-nodes', eager=False, core=core)
            if eval_rank() == 0:
                print(name, results[name], flush=True)
        if eval_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(model=args.model, input=str(args.test), batch_size=1, bucket=16,
                device=torch.cuda.get_device_name(), eval_world_size=eval_world_size(), mix=mix,
                train_sha256=train_sha256, label_only=args.label_only, results=results), indent=2) + '\n')
        return

    with torch.no_grad():
        features, scores = [], []
        for row in eval_shard(sample):
            ids, meta, _ = packed(row)
            current, encoded = reader.features(base, ids, meta)
            scores.append(current)
            features.append(encoded)
        scores = gather_rows(torch.stack(scores), len(sample))
        features = gather_rows(torch.stack(features), len(sample))
        if eval_rank() == 0:
            print('Collected frozen single-pass candidate features', flush=True)
        delta = None
        if not args.ablate:
            slow_rows = sample[:1024]
            slow = pack_prompts(eval_shard(slow_rows),
                                lambda r: render_prompt(tokenizer, r, model.config.latent_slots, marker), pad, device)
            delta = center(gather(base, LoopReadout(True), slow) - gather(base, LoopReadout(False), slow))
            delta = gather_rows(delta, len(slow_rows))
            if eval_rank() == 0:
                print('Collected real ten-round logit residual targets', flush=True)
    gold = torch.tensor([r['gold_index'] for r in sample], device=device)
    test = [packed(row) for row in rows]
    if args.ablate:
        run_ablation(args, model, base, projection, test, scores, features, gold, validation, mix, train_sha256)
        return
    adapters = {}
    epochs = {}
    for name, use_delta in [('answer-only', False), ('loop-residual', True)]:
        adapters[name], epochs[name] = fit_adapter(scores, features, delta, gold, validation, distill=use_delta)
    results = {}
    if eval_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, adapter in adapters.items():
        result = evaluate(base, test, 'time-nodes', eager=False,
                          core=CandidateFeatures(model, projection, adapter, mix=mix))
        results[name] = {**result, 'epoch': epochs[name]}
        if eval_rank() == 0:
            torch.save(dict(model=args.model, projection=projection.cpu(), adapter=adapter.cpu().state_dict(),
                            mix=mix, train_sha256=train_sha256),
                       args.output.with_name(f'{args.output.stem}-{name}.pt'))
            print(name, results[name], flush=True)
            args.output.write_text(json.dumps(dict(model=args.model, input=str(args.test),
                batch_size=1, bucket=16, device=torch.cuda.get_device_name(), eval_world_size=eval_world_size(),
                mix=mix, train_sha256=train_sha256, results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
