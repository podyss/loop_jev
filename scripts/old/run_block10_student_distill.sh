#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
scratch="/var/tmp/loop-jev-$(id -u)"
teacher="${LOOP_TEACHER_MODEL:-$scratch/ce-kl-1024-block10-seed43-step0.1}"
checkpoints="${STUDENT_CHECKPOINT_ROOT:-outputs/jev-block10-student-distill/checkpoints}"
test_input=data/gsm-mc/test.jsonl
run="outputs/jev-block10-student-distill/test"
test -f "$teacher/config.json"
test -f "$test_input"
mkdir -p "$checkpoints" "$run"

for arm in ce bypass-kd loop10-kd; do
  checkpoint="$checkpoints/$arm"
  previous="$scratch/block10-student-distill/$arm"
  if [[ ! -e "$checkpoint" && -s "$previous/experiment.json" ]]; then
    checkpoint="$previous"
  fi
  flags=(--loop-count 1 --no-loop-damping --logit-loss-weight 0)
  if [[ "$arm" != ce ]]; then
    flags=(--loop-teacher-model "$teacher" --loop-start-layer 9 --loop-end-layer 18
      --loop-count 10 --loop-damping --loop-damping-step 0.1 --logit-loss-weight 1.0)
    if [[ "$arm" == bypass-kd ]]; then
      flags+=(--disable-teacher-looping)
    fi
  fi

  if [[ ! -s "$checkpoint/experiment.json" ]]; then
    available=$(df --output=avail -k "$checkpoints" | tail -n 1)
    if (( available < 8 * 1024 * 1024 )); then
      echo "Need at least 8 GiB free at $checkpoints to save $arm" >&2
      exit 1
    fi
    uv run torchrun --standalone --nproc_per_node=2 \
      --module loop_jev.training.answer_slot_distill \
      --model Qwen/Qwen3-1.7B --train data/gsm-mc/train.jsonl --output "$checkpoint" \
      --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
      --max-length 16384 --max-train-examples 1024 --seed 43 --train-sample-seed 42 \
      --temperature 2.0 --answer-supervision final-loop --latent-slots 0 \
      --hidden-loss-weight 0 --process-loss-weight 0 \
      --tensor-parallel-size 1 --data-parallel-size 2 "${flags[@]}"
  fi

  uv run python -m loop_jev.evaluation.zero_shot --model "$checkpoint" \
    --input "$test_input" --output "$run/$arm.jsonl" --backend transformers \
    --data-parallel-size 2 --batch-size 4 --disable-looping
done

uv run python - "$run" "$test_input" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file

run, test_input = map(Path, sys.argv[1:])
expected = {json.loads(line)["id"] for line in test_input.read_text().splitlines() if line}
if len(expected) != 1319:
    raise ValueError("Expected 1319 unique test IDs")
results = {}
for arm in ("ce", "bypass-kd", "loop10-kd"):
    results[arm], ids = score_file(run / f"{arm}.jsonl", disable_looping=True)
    if ids != expected:
        raise ValueError(f"Unexpected test IDs for {arm}")
    result = results[arm]
    print(f"{arm:12} {result['accuracy']:.5f} ({result['correct']}/{result['total']})")
deltas = {
    f"loop10_kd_vs_{control.replace('-', '_')}_pp": (
        100 * (results["loop10-kd"]["accuracy"] - results[control]["accuracy"])
    )
    for control in ("ce", "bypass-kd")
}
for name, value in deltas.items():
    print(f"{name}: {value:+.2f} pp")
summary = run / "summary.json"
summary.write_text(json.dumps({"results": results, "deltas": deltas}, indent=2) + "\n")
print(f"Saved: {summary}")
PY
