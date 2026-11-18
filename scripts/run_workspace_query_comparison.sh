#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"

prior=outputs/jev-loop-cot-comparison
validation="$prior/validation.jsonl"
eval_dir=outputs/jev-workspace-query-comparison/validation
checkpoint_root="${1:-/dev/shm/loop-jev-$(id -u)/workspace-query}"
for file in "$validation" "$prior"/validation/{ce1-native,ce3-native,ce3-no_loop}.jsonl; do
  [[ -s "$file" ]] || { echo "Missing validation input or baseline: $file" >&2; exit 1; }
done

for loops in 1 3; do
  output="$checkpoint_root/query-$loops"
  if [[ ! -f "$output/experiment.json" ]]; then
    uv run torchrun --standalone --nproc_per_node=2 \
      --module loop_jev.training.answer_slot_distill \
      --model Qwen/Qwen3-1.7B --train data/gsm-mc/train.jsonl --output "$output" \
      --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
      --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
      --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
      --loop-start-layer 9 --loop-end-layer 18 --loop-count "$loops" --loop-damping \
      --latent-slots 24 --workspace-query-weight 0.1 --tensor-parallel-size 1 --data-parallel-size 2
  fi
  modes=(native)
  if [[ "$loops" == 3 ]]; then modes+=(no-loop); fi
  for mode in "${modes[@]}"; do
    extra=()
    if [[ "$mode" == no-loop ]]; then extra+=(--disable-looping); fi
    uv run python -m loop_jev.evaluation.zero_shot \
      --model "$output" --input "$validation" --output "$eval_dir/query-$loops-$mode.jsonl" \
      --backend transformers --data-parallel-size 2 --batch-size 4 "${extra[@]}"
  done
done

uv run python - "$eval_dir" "$validation" "$checkpoint_root" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file
from loop_jev.training.answer_slot_distill import read_rows

output, validation, checkpoints = map(Path, sys.argv[1:])
expected_ids = {row["id"] for row in read_rows(validation)}
if len(expected_ids) != 1024:
    raise ValueError(f"Expected the original 1024-example validation split: {validation}")
prior = validation.parent / "validation"
files = {
    "ce_kl_1": (prior / "ce1-native.jsonl", False),
    "ce_kl_3": (prior / "ce3-native.jsonl", False),
    "ce_kl_3_no_loop": (prior / "ce3-no_loop.jsonl", True),
    "query_1": (output / "query-1-native.jsonl", False),
    "query_3": (output / "query-3-native.jsonl", False),
    "query_3_no_loop": (output / "query-3-no-loop.jsonl", True),
}
results = {}
print(f"GSM-MC validation ({len(expected_ids)} examples)")
for label, (path, off) in files.items():
    score, ids = score_file(path, disable_looping=off)
    if ids != expected_ids:
        raise ValueError(f"Wrong evaluation set: {path}")
    results[label] = score
    print(f"{label}: {score['accuracy']:.5f} ({score['correct']}/{score['total']})")

def delta(a, b):
    return 100 * (results[a]["accuracy"] - results[b]["accuracy"])

deltas = {
    "query_gain_at_1_loop_pp": delta("query_1", "ce_kl_1"),
    "query_gain_at_3_loops_pp": delta("query_3", "ce_kl_3"),
    "query_3_vs_query_1_pp": delta("query_3", "query_1"),
    "query_3_same_checkpoint_loop_gain_pp": delta("query_3", "query_3_no_loop"),
    "extra_loop_gain_from_query_pp": delta("query_3", "query_1") - delta("ce_kl_3", "ce_kl_1"),
}
for label, value in deltas.items():
    print(f"{label}: {value:+.2f} pp")
training = {}
for loops in (1, 3):
    metadata = json.loads((checkpoints / f"query-{loops}" / "experiment.json").read_text())
    training[f"query_{loops}"] = {
        "covered_examples": metadata["workspace_query_coverage"],
        "epochs": [epoch["workspace_query"] for epoch in metadata["training_history"]],
        "peak_reserved_gib": max(epoch["peak_reserved_gib"] for epoch in metadata["training_history"]),
    }
summary = output / "summary.json"
summary.write_text(json.dumps({"results": results, "deltas": deltas, "training": training}, indent=2) + "\n")
print(f"Saved: {summary}")
PY
