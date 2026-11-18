"""Train and compare ordered and shuffled recurrent CoT supervision."""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file
from loop_jev.training.answer_slot_distill import read_rows


TRAIN = Path("data/gsm-mc/train.jsonl")
SEED = 42
TRAIN_COUNT = 1024
VALIDATION_COUNT = 1024
MODES = {"ordered3": (3, "ordered"), "shuffled3": (3, "shuffled"),
         "ce3": (3, None), "ce1": (1, None)}


def validation_rows(rows):
    trained = set(random.Random(SEED).sample(range(len(rows)), TRAIN_COUNT))
    remaining = [index for index in range(len(rows)) if index not in trained]
    return [dict(rows[index], split="validation")
            for index in random.Random(SEED + 1).sample(remaining, VALIDATION_COUNT)]


def check_metadata(path, loops, order, train_digest):
    metadata = json.loads((path / "experiment.json").read_text(encoding="utf-8"))
    expected = dict(status="complete", model="Qwen/Qwen3-1.7B", train_sha256=train_digest,
                    examples=TRAIN_COUNT, max_train_examples=TRAIN_COUNT, max_length=16384,
                    seed=SEED, completed_epochs=3, learning_rate=2e-6, temperature=2.0,
                    loop_start_layer=9, loop_end_layer=18, loop_count=loops, loop_damping=True,
                    latent_slots=24, answer_supervision="final-loop", logit_loss_type="kl",
                    logit_loss_weight=1.0, hidden_loss_weight=0.0, kv_loss_weight=0.0,
                    process_loss_weight=0.0, batch_size_per_gpu=4, gradient_accumulation_steps=2,
                    data_parallel_size=2, tensor_parallel_size=1)
    if order:
        expected.update(loop_cot_weight=0.1, loop_cot_order=order, loop_cot_tokens=8)
    elif metadata.get("loop_cot_weight", 0) != 0:
        raise ValueError(f"Unexpected CoT supervision in baseline: {path}")
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(f"Mismatched checkpoint {path}: {', '.join(mismatches)}")


def run_module(module, *args):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], check=True)


def train(path, loops, order):
    cot = ("--loop-cot-weight", "0.1", "--loop-cot-order", order) if order else ()
    run_module(
        "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        "--module", "loop_jev.training.answer_slot_distill",
        "--model", "Qwen/Qwen3-1.7B", "--train", TRAIN, "--output", path,
        "--epochs", 3, "--learning-rate", "2e-6", "--batch-size", 4,
        "--gradient-accumulation-steps", 2, "--max-length", 16384,
        "--max-train-examples", TRAIN_COUNT, "--seed", SEED, "--temperature", "2.0",
        "--answer-supervision", "final-loop", "--logit-loss-weight", "1.0",
        "--hidden-loss-weight", "0.0", "--loop-start-layer", 9, "--loop-end-layer", 18,
        "--loop-count", loops, "--loop-damping", "--latent-slots", 24,
        "--tensor-parallel-size", 1, "--data-parallel-size", 2, *cot,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path(f"/tmp/loop-jev-{os.getuid()}/loop-cot"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,2")

    run_dir = Path("outputs/jev-loop-cot-comparison")
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_input = Path("data/gsm-mc/test.jsonl")
    if args.split == "validation":
        eval_input = run_dir / "validation.jsonl"
        rows = validation_rows(read_rows(TRAIN))
        eval_input.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                              encoding="utf-8")
    expected_ids = {row["id"] for row in read_rows(eval_input)}
    digest = hashlib.sha256(TRAIN.read_bytes()).hexdigest()
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for mode, (loops, order) in MODES.items():
        prior = Path(f"outputs/jev-latent24-{mode}-final-ce-kl")
        path = prior if not order and (prior / "experiment.json").is_file() else args.checkpoint_root / mode
        if (path / "experiment.json").is_file():
            check_metadata(path, loops, order, digest)
        elif path.exists() and any(path.iterdir()):
            raise ValueError(f"Incomplete checkpoint: {path}. Use a new --checkpoint-root to retrain.")
        paths[mode] = path
    missing = sum(not (path / "experiment.json").is_file() for path in paths.values())
    if shutil.disk_usage(args.checkpoint_root).free < missing * 8 * 2**30:
        raise OSError(f"Need about {missing * 8} GiB free at {args.checkpoint_root} for new checkpoints")

    results = {}
    for mode, (loops, order) in MODES.items():
        path = paths[mode]
        if not (path / "experiment.json").is_file():
            train(path, loops, order)
            check_metadata(path, loops, order, digest)
        else:
            print(f"Using completed checkpoint: {path}", flush=True)
        scores = {}
        eval_modes = [("native", ())]
        if loops > 1:
            eval_modes.append(("no_loop", ("--disable-looping",)))
        for label, extra in eval_modes:
            out = run_dir / args.split / f"{mode}-{label}.jsonl"
            run_module("loop_jev.evaluation.zero_shot", "--model", path, "--input", eval_input,
                       "--output", out, "--backend", "transformers", "--data-parallel-size", 2,
                       "--batch-size", 4, *extra)
            scores[label], ids = score_file(out, disable_looping=bool(extra))
            if ids != expected_ids:
                raise ValueError(f"Evaluation does not match {eval_input}: {out}")
        results[mode] = {"checkpoint": str(path), **scores}

    def delta(left, right):
        return 100 * (results[left]["native"]["accuracy"] - results[right]["native"]["accuracy"])

    deltas = {"ordered_vs_shuffled_pp": delta("ordered3", "shuffled3"),
              "ordered_vs_ce3_pp": delta("ordered3", "ce3"),
              "ordered_vs_ce1_pp": delta("ordered3", "ce1"),
              "ordered_same_checkpoint_loop_gain_pp": 100 * (
                  results["ordered3"]["native"]["accuracy"]
                  - results["ordered3"]["no_loop"]["accuracy"])}
    summary = {"split": args.split, "examples": len(expected_ids), "results": results, "deltas": deltas}
    out = run_dir / args.split / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"GSM-MC {args.split} ({len(expected_ids)} examples)")
    for mode, scores in results.items():
        fields = [f"{label}: {score['accuracy']:.5f} ({score['correct']}/{score['total']})"
                  for label, score in scores.items() if label != "checkpoint"]
        print(f"{mode:10} {'  '.join(fields)}")
    for name, value in deltas.items():
        print(f"{name}: {value:+.2f} pp")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
