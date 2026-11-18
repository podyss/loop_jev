"""Summarize matched latent-workspace teacher experiments."""

import argparse
import json
from pathlib import Path


def score_file(path, *, disable_looping):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"Empty evaluation: {path}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate example IDs: {path}")
    if any(row["disable_looping"] != disable_looping for row in rows):
        raise ValueError(f"Evaluation mode differs from filename: {path}")
    correct = sum(row["correct_prediction"] for row in rows)
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}, set(ids)


def summarize(root):
    results = {}
    expected_ids = None
    for mode in ("kv3", "ce3", "kv1", "ce1"):
        loops = 3 if mode.endswith("3") else 1
        directory = root / f"jev-latent24-{mode}-final-ce-kl"
        metadata = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
        if (metadata["status"] != "complete" or metadata["loop_count"] != loops
                or metadata["latent_slots"] != 24
                or metadata["kv_loss_weight"] != (0.1 if mode.startswith("kv") else 0.0)):
            raise ValueError(f"Unexpected checkpoint configuration: {directory}")
        looped, ids = score_file(
            directory / "eval" / f"gsm-mc-test-loop{loops}.jsonl", disable_looping=False,
        )
        if expected_ids is None:
            expected_ids = ids
        if ids != expected_ids:
            raise ValueError(f"Evaluations use different example IDs: {directory}")
        direct = looped
        if loops > 1:
            direct, ids = score_file(
                directory / "eval" / "gsm-mc-test-no-loop.jsonl", disable_looping=True,
            )
            if ids != expected_ids:
                raise ValueError(f"Looped and direct evaluations use different examples: {directory}")
        results[mode] = {"looped": looped, "no_loop": direct,
                         "checkpoint": str(directory)}

    accuracy = lambda mode: results[mode]["looped"]["accuracy"]
    deltas = {
        "kv_gain_at_3_loops_pp": 100 * (accuracy("kv3") - accuracy("ce3")),
        "kv_gain_at_1_loop_pp": 100 * (accuracy("kv1") - accuracy("ce1")),
        "loop_gain_with_kv_pp": 100 * (accuracy("kv3") - accuracy("kv1")),
        "loop_gain_without_kv_pp": 100 * (accuracy("ce3") - accuracy("ce1")),
        "kv3_same_checkpoint_loop_gain_pp": 100 * (
            accuracy("kv3") - results["kv3"]["no_loop"]["accuracy"]),
    }
    return {"examples": len(expected_ids), "results": results, "deltas": deltas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"GSM-MC test ({summary['examples']} examples)")
    print(f"{'experiment':<12} {'looped':>18} {'no loop':>18}")
    for mode, result in summary["results"].items():
        def fmt(score):
            return f"{score['accuracy']:.5f} ({score['correct']}/{score['total']})"
        print(f"{mode:<12} {fmt(result['looped']):>18} {fmt(result['no_loop']):>18}")
    for name, value in summary["deltas"].items():
        print(f"{name}: {value:+.2f} pp")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
