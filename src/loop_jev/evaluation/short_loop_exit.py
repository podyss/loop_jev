"""Calibrate a two-depth batch-one schedule on held-out GSM-MC rows."""

import argparse
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import enable_configured_looping
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.transition import load_loop_model


@torch.inference_mode()
def collect(model, tokenizer, rows, labels, depths):
    cfg = model.config
    weight = model.lm_head.weight[labels].contiguous()
    captured = {}
    records = []

    def read(round_number, state, suffix):
        if round_number in depths:
            captured[round_number] = F.linear(model.model.norm(suffix(state))[:, -1, :], weight).float()

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        prompts = [render_prompt(tokenizer, row, cfg.latent_slots, cfg.latent_marker_token_id) for row in batch]
        width = max(map(len, prompts))
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device="cuda")
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device="cuda")
        final = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1,
                      block_state_observer=read).logits
        if depths:
            early = torch.stack([captured[d] for d in depths], dim=1).topk(2, dim=-1)
            predictions = early.indices[..., 0].tolist()
            margins = (early.values[..., 0] - early.values[..., 1]).tolist()
        else:
            predictions = margins = [()] * len(batch)
        records.extend(zip((row["gold_index"] for row in batch), final[:, -1, labels].argmax(-1).tolist(),
                           predictions, margins, strict=True))
    return records


def calibrate(records, depths, max_drop):
    reference = sum(final == gold for gold, final, _, _ in records)
    baseline = (10, -1, math.inf, reference)
    best = baseline
    for index, depth in enumerate(depths):
        # Compare the gain/loss at each attainable margin threshold on validation only.
        ranked = sorted(((margins[index], int(early[index] == gold) - int(final == gold))
                         for gold, final, early, margins in records), reverse=True)
        correct = reference
        for i, (margin, delta) in enumerate(ranked):
            correct += delta
            if i + 1 < len(ranked) and ranked[i + 1][0] == margin:
                continue
            average = 10 - (10 - depth) * (i + 1) / len(ranked)
            if correct >= reference - max_drop and (average, -correct) < (best[0], -best[3]):
                best = (average, depth, margin, correct)
    return reference, best


@torch.inference_mode()
def evaluate(model, tokenizer, rows, labels, depth, threshold):
    cfg = model.config
    weight = model.lm_head.weight[labels].contiguous()
    elapsed, results, rounds = [], [], Counter()
    took_exit = False

    def should_exit(hidden):
        nonlocal took_exit
        scores = F.linear(hidden[:, -1, :], weight).float().topk(2, dim=-1).values
        took_exit = (scores[0, 0] - scores[0, 1]).item() >= threshold
        return took_exit

    warmup = min(12, len(rows))
    for i, row in enumerate(rows[:warmup] + rows):
        ids = torch.tensor([render_prompt(tokenizer, row, cfg.latent_slots, cfg.latent_marker_token_id)], device="cuda")
        took_exit = False
        torch.cuda.synchronize()
        start = time.perf_counter()
        options = {"block_early_exit": (depth, should_exit)} if depth != -1 else {}
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=1, **options).logits
        pred = logits[0, -1, labels].argmax().item()
        duration = 1000 * (time.perf_counter() - start)
        if i >= warmup:
            elapsed.append(duration)
            results.append(pred == row["gold_index"])
            rounds[depth if took_exit else cfg.loop_count] += 1
    return dict(correct=sum(results), total=len(results), round_counts=rounds,
                mean_rounds=sum(d * n for d, n in rounds.items()) / len(results),
                wall_ms_mean=statistics.mean(elapsed), wall_ms_p50=statistics.median(elapsed),
                wall_ms_p95=sorted(elapsed)[math.ceil(0.95 * len(elapsed)) - 1]), results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-calibration-drop", type=int, default=0)
    parser.add_argument("--latency-examples", type=int, default=128)
    args = parser.parse_args()
    calibration, test = read_rows(args.calibration), read_rows(args.test)
    if (not calibration or not test or args.max_calibration_drop < 0 or args.latency_examples < 1
            or {r["id"] for r in calibration} & {r["id"] for r in test}
            or any(len(r["options"]) != 4 for r in calibration + test)):
        parser.error("Expected disjoint, nonempty calibration and test examples with four options")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={"": 0},
                            attn_implementation="sdpa", trust_remote_code=False).eval()
    validate_checkpoint(model.config, marker_token_id(tokenizer))
    enable_configured_looping(model)
    label_tokens = [tokenizer.encode(label, add_special_tokens=False) for label in "ABCD"]
    if any(len(t) != 1 for t in label_tokens):
        parser.error("The tokenizer must encode each answer label to one token")
    labels = [t[0] for t in label_tokens]
    depths = (3, 4, 5, 6)
    validation = collect(model, tokenizer, calibration, labels, depths)
    ref, (avg, depth, threshold, correct) = calibrate(validation, depths, args.max_calibration_drop)
    selected = dict(depth=depth, threshold=threshold if depth != -1 else None, correct=correct,
                    reference_correct=ref, total=len(validation), mean_rounds=avg)
    print(f"calibration: {selected}", flush=True)
    rows = collect(model, tokenizer, test, labels, (depth,) if depth != -1 else ())
    exited = [depth != -1 and margin[0] >= threshold for _, _, _, margin in rows]
    base_rows = [gold == final for gold, final, _, _ in rows]
    new_rows = [gold == (early[0] if use_early else final)
                for (gold, final, early, _), use_early in zip(rows, exited, strict=True)]
    sample = random.Random(42).sample(test, min(args.latency_examples, len(test)))
    baseline_time, _ = evaluate(model, tokenizer, sample, labels, -1, None)
    adaptive_time, _ = evaluate(model, tokenizer, sample, labels, depth, threshold)
    def timing(result):
        return {k: v for k, v in result.items() if k.startswith("wall_ms_")}
    baseline = dict(correct=sum(base_rows), total=len(rows), **timing(baseline_time))
    rounds = Counter(depth if exit_early else 10 for exit_early in exited)
    adaptive = dict(correct=sum(new_rows), total=len(rows), round_counts=rounds,
                    mean_rounds=sum(d * n for d, n in rounds.items()) / len(rows), **timing(adaptive_time))
    adaptive.update(wins=sum(a and not b for a, b in zip(new_rows, base_rows, strict=True)),
                    losses=sum(b and not a for a, b in zip(new_rows, base_rows, strict=True)))
    print(f"adaptive: {adaptive}", flush=True)
    print(f"baseline: {baseline}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(calibration=selected, original=baseline, adaptive=adaptive,
                   accuracy_mode="batch-4 replay of the calibrated policy", latency_examples=len(sample),
                   latency_mode="batch-1 live exit", device=torch.cuda.get_device_name())
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
