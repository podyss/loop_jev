#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"

validation=outputs/jev-loop-cot-comparison/validation.jsonl
eval_dir=outputs/jev-ce-only-comparison/validation
checkpoint_root="${1:-/tmp/loop-jev-$(id -u)/ce-only}"
[[ -s "$validation" ]] || { echo "Missing validation set: $validation" >&2; exit 1; }

for loops in 1 3; do
  output="$checkpoint_root/ce-only-$loops"
  if [[ ! -f "$output/experiment.json" ]]; then
    uv run torchrun --standalone --nproc_per_node=2 \
      --module loop_jev.training.answer_slot_distill \
      --model Qwen/Qwen3-1.7B --train data/gsm-mc/train.jsonl --output "$output" \
      --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
      --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
      --answer-supervision final-loop --logit-loss-weight 0 --hidden-loss-weight 0 \
      --loop-start-layer 9 --loop-end-layer 18 --loop-count "$loops" --loop-damping \
      --latent-slots 24 --tensor-parallel-size 1 --data-parallel-size 2
  fi
  modes=(native)
  if [[ "$loops" == 3 ]]; then modes+=(no-loop); fi
  for mode in "${modes[@]}"; do
    extra=()
    if [[ "$mode" == no-loop ]]; then extra+=(--disable-looping); fi
    uv run python -m loop_jev.evaluation.zero_shot \
      --model "$output" --input "$validation" --output "$eval_dir/ce-only-$loops-$mode.jsonl" \
      --backend transformers --data-parallel-size 2 --batch-size 4 "${extra[@]}"
  done
done

uv run python - "$eval_dir" "$validation" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file
from loop_jev.training.answer_slot_distill import read_rows

output, validation = map(Path, sys.argv[1:])
expected_ids = {row["id"] for row in read_rows(validation)}
prior_dir = validation.parent / "validation"
files = {
    "ce_only_1": output / "ce-only-1-native.jsonl",
    "ce_only_3": output / "ce-only-3-native.jsonl",
    "ce_only_3_no_loop": output / "ce-only-3-no-loop.jsonl",
    "ce_kl_1": prior_dir / "ce1-native.jsonl",
    "ce_kl_3": prior_dir / "ce3-native.jsonl",
}
results = {}
for label, path in files.items():
    score, ids = score_file(path, disable_looping=label == "ce_only_3_no_loop")
    if ids != expected_ids:
        raise ValueError(f"Wrong evaluation set: {path}")
    results[label] = score
    print(f"{label}: {score['accuracy']:.5f} ({score['correct']}/{score['total']})")

def delta(a, b):
    return 100 * (results[a]["accuracy"] - results[b]["accuracy"])

deltas = {
    "ce_only_loop_gain_pp": delta("ce_only_3", "ce_only_1"),
    "ce_kl_loop_gain_pp": delta("ce_kl_3", "ce_kl_1"),
    "ce_only_vs_ce_kl_at_3_loops_pp": delta("ce_only_3", "ce_kl_3"),
    "ce_only_same_checkpoint_loop_gain_pp": delta("ce_only_3", "ce_only_3_no_loop"),
}
for label, value in deltas.items():
    print(f"{label}: {value:+.2f} pp")
summary = output / "summary.json"
summary.write_text(json.dumps({"results": results, "deltas": deltas}, indent=2) + "\n")
print(f"Saved: {summary}")
PY
