"""Report accuracy or latency across all 1,319 GSM-MC test questions."""

import argparse
import json
import math
import statistics
import time
from functools import partial
from pathlib import Path

import torch
from transformers import AutoTokenizer

from loop_jev.evaluation.zero_shot import read_rows, render_prompt
from loop_jev.looping import enable_looping
from loop_jev.training.latent_kv import marker_token_id
from loop_jev.training.loop_removal import validate_checkpoint
from loop_jev.trajectory import extrapolate, parallel_finish, segmented_extrapolate
from loop_jev.transition import load_loop_model

TEST_SIZE = 1319


def validate_test_rows(rows):
    expected_ids = {f"gsm-mc:test:{index}" for index in range(TEST_SIZE)}
    if (len(rows) != TEST_SIZE or {row.get("id") for row in rows} != expected_ids
            or any(row.get("split") != "test" or len(row.get("options", ())) != 4
                   or row.get("gold_index") not in range(4) for row in rows)):
        raise ValueError(f"Expected the complete GSM-MC test split ({TEST_SIZE} unique four-option questions)")


def parse_mode(value):
    if value == "native":
        return 1, None
    try:
        rounds, step = value.split(":")
        rounds, step = int(rounds), float(step)
        if 1 <= rounds <= 10 and 0 < step <= 1:
            return rounds, step
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Expected native or ROUNDS:STEP (rounds 1-10, step in (0, 1])")


def parse_experiment(value):
    parts = value.split(":")
    if len(parts) in (1, 2):
        return parse_mode(value), None, None
    if len(parts) == 4:
        mode = parse_mode(":".join(parts[:2]))
        if mode[0] == 3 and parts[2] in ("linear", "dmd", "split-dmd") and parts[3] in tuple("1234567"):
            return mode, parts[2], int(parts[3])
        if mode == (2, 0.1) and parts[2] in ("picard", "picard2") and parts[3] in ("2", "4", "8"):
            return mode, parts[2], int(parts[3])
    raise argparse.ArgumentTypeError("Expected native, ROUNDS:STEP, 3:STEP:{linear,dmd,split-dmd}:HORIZON, "
                                     "or 2:0.1:{picard,picard2}:NODES")


def _finish_path(path, mask, _evaluate, *, transform):
    return transform(path, mask)


def _finish_parallel(path, _mask, evaluate, *, step, nodes, sweeps):
    return parallel_finish(path, evaluate, rounds=8, step=step, nodes=nodes, sweeps=sweeps)


@torch.inference_mode()
def evaluate(model, mode, batches, label_ids, *, transform=None, latency_only=False):
    rounds, step = mode
    if step is None:
        model.model.forward = getattr(model.model, "_unlooped_forward", model.model.forward)
    else:
        enable_looping(model, 9, 18, rounds, damping=True, damping_step=step, collect_all_readouts=False)
    extra = {} if transform is None else {"block_final_transform": transform}

    def predict(ids):
        logits = model(input_ids=ids, logits_to_keep=1, use_cache=False, **extra).logits[0, -1, label_ids]
        return logits.argmax().item()

    if latency_only:
        warmup = min(20, len(batches))
        for _, ids in batches[:warmup]:
            predict(ids)
        torch.cuda.synchronize()
        latencies = []
        for _, ids in batches:
            start = time.perf_counter()
            predict(ids)  # Reading the GPU prediction synchronizes the measured forward.
            latencies.append((time.perf_counter() - start) * 1000)
        return dict(total=len(latencies), warmup=warmup, wall_ms_p50=statistics.median(latencies),
                    wall_ms_p95=sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1],
                    wall_ms_mean=statistics.mean(latencies))

    correct = 0
    for row, ids in batches:
        correct += predict(ids) == row["gold_index"]
    return dict(correct=correct, total=len(batches), accuracy=correct / len(batches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, default=Path("data/gsm-mc/test.jsonl"))
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path, help="Write full-test accuracy only")
    target.add_argument("--latency-output", type=Path, help="Write full-test latency only")
    parser.add_argument("--modes", nargs="+", default=["10:0.1", "native", "2:0.5", "3:0.3333333333333333"])
    args = parser.parse_args()
    try:
        modes = {name: parse_experiment(name) for name in args.modes}
        rows = read_rows(args.input)
        validate_test_rows(rows)
    except (argparse.ArgumentTypeError, ValueError) as error:
        parser.error(str(error))
    model = load_loop_model(args.model, dtype=torch.bfloat16, device_map={"": 0},
                            attn_implementation="sdpa", trust_remote_code=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    cfg = model.config
    saved_rounds = getattr(cfg, "loop_count", 1)
    slots = getattr(cfg, "latent_slots", 0)
    marker = marker_token_id(tokenizer) if slots or saved_rounds == 10 else None
    if saved_rounds == 10:
        validate_checkpoint(cfg, marker)
    elif saved_rounds not in (1, 2, 3) or len(modes) != 1:
        parser.error("A shortened checkpoint can only be measured with its saved configuration")
    if saved_rounds != 10:
        expected_step = cfg.loop_damping_step if saved_rounds > 1 else None
        if next(iter(modes.values())) != ((saved_rounds, expected_step), None, None):
            parser.error("Requested mode does not match the checkpoint")
    label_tokens = [tokenizer.encode(label, add_special_tokens=False) for label in "ABCD"]
    if any(len(ids) != 1 for ids in label_tokens):
        parser.error("Expected one token per answer label")
    label_ids = [ids[0] for ids in label_tokens]
    dev = torch.device("cuda:0")
    batches = [(row, torch.tensor([render_prompt(tokenizer, row, slots, marker)], device=dev))
               for row in rows]
    results = {}
    for name, (mode, method, horizon) in modes.items():
        if method in ("picard", "picard2"):
            sweeps = 2 if method == "picard2" else 1
            transform = partial(_finish_parallel, step=mode[1], nodes=horizon, sweeps=sweeps)
            budget = dict(horizon=8, nodes=horizon, sweeps=sweeps, sequential_block_calls=mode[0] + sweeps,
                          block_inputs=mode[0] + sweeps * horizon)
        elif method == "split-dmd":
            width = slots + len(tokenizer.encode("Answer: (", add_special_tokens=False))
            transform = partial(_finish_path, transform=partial(segmented_extrapolate, steps=horizon,
                                                                 workspace_width=width))
        elif method is not None:
            transform = partial(_finish_path, transform=partial(extrapolate, steps=horizon, method=method))
        else:
            transform = None
        if method not in ("picard", "picard2"):
            budget = dict(horizon=horizon, sequential_block_calls=mode[0], block_inputs=mode[0])
        stats = evaluate(model, mode, batches, label_ids, transform=transform,
                         latency_only=args.latency_output is not None)
        results[name] = dict(rounds=mode[0], step=mode[1], transform=method, **budget, **stats)
        print(name, json.dumps(results[name]), flush=True)
    output = args.latency_output or args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(model=args.model, input=str(args.input), batch_size=1,
                                     device=torch.cuda.get_device_name(), results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
