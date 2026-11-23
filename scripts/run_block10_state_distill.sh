#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
teacher="${LOOP_TEACHER_MODEL:-/var/tmp/loop-jev-$(id -u)/ce-kl-1024-block10-seed43-step0.1}"
fixed_root=outputs/jev-block10-state-distill
root="$fixed_root"
arm=state-delta-kd
baseline_arm=loop10-kd
baseline=outputs/jev-block10-student-distill/test/loop10-kd.jsonl
delta_name=state_delta_vs_loop10_kd_pp
capture_flags=()
training_flags=()
logit_weight=1.0
if [[ $# -eq 0 ]]; then
  checkpoint="${STATE_DISTILL_CHECKPOINT:-$root/checkpoint}"
elif [[ $# -eq 1 && $1 == --curriculum ]]; then
  baseline_arm=state-delta-kd
  baseline="$root/test/$baseline_arm.jsonl"
  root=outputs/jev-block10-state-curriculum
  checkpoint="${STATE_CURRICULUM_CHECKPOINT:-$root/checkpoint}"
  arm=state-curriculum-kd
  delta_name=state_curriculum_vs_state_delta_pp
  capture_flags=(--target-rounds 4 7 10)
  training_flags=(--block-state-rounds 4 7 10)
elif [[ $# -eq 1 && $1 == --shuffle-within-label ]]; then
  baseline_arm=state-delta-kd
  baseline="$fixed_root/test/$baseline_arm.jsonl"
  root=outputs/jev-block10-state-shuffled
  checkpoint="${STATE_SHUFFLED_CHECKPOINT:-$root/checkpoint}"
  arm=state-shuffled-kd
  delta_name=state_shuffled_vs_state_delta_pp
  training_flags=(--block-state-shuffle-seed 42)
elif [[ $# -eq 1 && $1 == --no-logit-kd ]]; then
  baseline_arm=state-delta-kd
  baseline="$fixed_root/test/$baseline_arm.jsonl"
  root=outputs/jev-block10-state-no-logit-kd
  checkpoint="${STATE_NO_LOGIT_KD_CHECKPOINT:-$root/checkpoint}"
  arm=state-no-logit-kd
  delta_name=state_no_logit_vs_state_delta_pp
  logit_weight=0
elif [[ $# -eq 1 && $1 == --absolute-state ]]; then
  baseline_arm=state-delta-kd
  baseline="$fixed_root/test/$baseline_arm.jsonl"
  root=outputs/jev-block10-state-absolute
  checkpoint="${STATE_ABSOLUTE_CHECKPOINT:-$root/checkpoint}"
  arm=state-absolute-kd
  delta_name=state_absolute_vs_state_delta_pp
  capture_flags=(--target-mode absolute)
  training_flags=(--block-state-target-mode absolute)
else
  echo "Usage: $0 [--curriculum | --shuffle-within-label | --no-logit-kd | --absolute-state]" >&2
  exit 2
fi
cache="$root/answer-deltas.pt"
if [[ $arm == state-shuffled-kd ]]; then
  cache="$fixed_root/answer-deltas.pt"
  test -s "$cache" || { echo "Missing fixed-delta cache: $cache" >&2; exit 1; }
fi
if [[ $arm == state-no-logit-kd ]]; then
  cache="$fixed_root/answer-deltas.pt"
fi
if [[ $arm == state-absolute-kd ]]; then
  cache="$root/answer-states.pt"
fi
base=Qwen/Qwen3-1.7B
train=data/gsm-mc/train.jsonl
test_input=data/gsm-mc/test.jsonl
test -s "$teacher/config.json"
test -s "$baseline"
test -s "$test_input"
mkdir -p "$root/test" "$(dirname "$checkpoint")"

if [[ ! -s "$checkpoint/experiment.json" ]]; then
  available=$(df --output=avail -k "$(dirname "$checkpoint")" | tail -n 1)
  if (( available < 8 * 1024 * 1024 )); then
    echo "Need at least 8 GiB free to save $checkpoint" >&2
    exit 1
  fi
  if [[ ! -s "$cache" ]]; then
    uv run torchrun --standalone --nproc_per_node=2 --module loop_jev.training.block_state_cache \
      --model "$base" --teacher "$teacher" --train "$train" --output "$cache" \
      --max-train-examples 1024 --train-sample-seed 42 --batch-size 4 "${capture_flags[@]}"
  fi
  uv run torchrun --standalone --nproc_per_node=2 --module loop_jev.training.answer_slot_distill \
    --model "$base" --loop-teacher-model "$teacher" --train "$train" --output "$checkpoint" \
    --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
    --max-length 16384 --max-train-examples 1024 --seed 43 --train-sample-seed 42 \
    --temperature 2.0 --answer-supervision final-loop --latent-slots 0 \
    --hidden-loss-weight 0 --process-loss-weight 0 --logit-loss-weight "$logit_weight" \
    --loop-start-layer 9 --loop-end-layer 18 --loop-count 10 --loop-damping --loop-damping-step 0.1 \
    --tensor-parallel-size 1 --data-parallel-size 2 --block-state-cache "$cache" --block-state-weight 0.5 \
    "${training_flags[@]}"
fi

uv run python -m loop_jev.evaluation.zero_shot --model "$checkpoint" \
  --input "$test_input" --output "$root/test/$arm.jsonl" --backend transformers \
  --data-parallel-size 2 --batch-size 4 --disable-looping

uv run python - "$root/test" "$test_input" "$baseline" "$baseline_arm" "$arm" "$delta_name" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file

run, test_input, baseline = map(Path, sys.argv[1:4])
baseline_arm, arm, delta_name = sys.argv[4:]
expected = {json.loads(line)["id"] for line in test_input.read_text().splitlines() if line}
if len(expected) != 1319:
    raise ValueError("Expected 1319 unique test IDs")
results = {}
for name, path in ((baseline_arm, baseline), (arm, run / f"{arm}.jsonl")):
    stats, ids = score_file(path, disable_looping=True)
    if ids != expected:
        raise ValueError(f"Unexpected test IDs for {name}")
    results[name] = stats
    print(f"{name:15} {stats['accuracy']:.5f} ({stats['correct']}/{stats['total']})")
delta = 100 * (results[arm]["accuracy"] - results[baseline_arm]["accuracy"])
print(f"{delta_name}: {delta:+.2f} pp")
summary = run / "summary.json"
summary.write_text(json.dumps({"results": results, delta_name: delta}, indent=2) + "\n")
print(f"Saved: {summary}")
PY
