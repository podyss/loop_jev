#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
teacher="${LOOP_TEACHER_MODEL:-/var/tmp/loop-jev-$(id -u)/ce-kl-1024-block10-seed43-step0.1}"
root=outputs/jev-block10-layer-effect-distill
checkpoints="${LAYER_EFFECT_CHECKPOINT_ROOT:-$root/checkpoints}"
base=Qwen/Qwen3-1.7B
train=data/gsm-mc/train.jsonl
test_input=data/gsm-mc/test.jsonl
cache="$root/layer-effects.pt"
mkdir -p "$root/test" "$checkpoints"
test -s "$teacher/config.json"
test -s "$test_input"

needed=0
for mode in local shared; do
  if [[ ! -s "$checkpoints/$mode/experiment.json" ]]; then
    needed=$((needed + 8 * 1024 * 1024))
  fi
done
available=$(df --output=avail -k "$checkpoints" | tail -n 1)
if (( available < needed )); then
  echo "Need $((needed / 1024 / 1024)) GiB free for missing checkpoints at $checkpoints" >&2
  exit 1
fi

if (( needed > 0 )) && [[ ! -s "$cache" ]]; then
  uv run torchrun --standalone --nproc_per_node=2 --module loop_jev.training.block_layer_effect \
    --model "$base" --teacher "$teacher" --train "$train" --output "$cache" \
    --layers 12 15 18 --max-train-examples 1024 --train-sample-seed 42 --batch-size 4
fi

for mode in local shared; do
  checkpoint="$checkpoints/$mode"
  if [[ ! -s "$checkpoint/experiment.json" ]]; then
    uv run torchrun --standalone --nproc_per_node=2 --module loop_jev.training.answer_slot_distill \
      --model "$base" --loop-teacher-model "$teacher" --train "$train" --output "$checkpoint" \
      --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
      --max-length 16384 --max-train-examples 1024 --seed 43 --train-sample-seed 42 \
      --temperature 2.0 --answer-supervision final-loop --latent-slots 0 \
      --hidden-loss-weight 0 --process-loss-weight 0 --logit-loss-weight 1.0 \
      --loop-start-layer 9 --loop-end-layer 18 --loop-count 10 --loop-damping --loop-damping-step 0.1 \
      --tensor-parallel-size 1 --data-parallel-size 2 --layer-effect-cache "$cache" \
      --layer-effect-layers 12 15 18 --layer-effect-mode "$mode" --layer-effect-weight 0.5
  fi
  uv run python -m loop_jev.evaluation.zero_shot --model "$checkpoint" \
    --input "$test_input" --output "$root/test/$mode.jsonl" --backend transformers \
    --data-parallel-size 2 --batch-size 4 --disable-looping
done

uv run python - "$root/test" "$test_input" "$checkpoints" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file

run, test_input, checkpoints = map(Path, sys.argv[1:])
expected = {json.loads(line)["id"] for line in test_input.read_text().splitlines() if line}
if len(expected) != 1319:
    raise ValueError("Expected 1319 unique test IDs")
paths = {
    "loop10-kd": Path("outputs/jev-block10-student-distill/test/loop10-kd.jsonl"),
    "state-delta-kd": Path("outputs/jev-block10-state-distill/test/state-delta-kd.jsonl"),
    "local": run / "local.jsonl",
    "shared": run / "shared.jsonl",
}
results = {}
for mode, path in paths.items():
    if mode not in ("local", "shared") and not path.is_file():
        continue
    stats, ids = score_file(path, disable_looping=True)
    if ids != expected:
        raise ValueError(f"Unexpected test IDs for {mode}")
    if mode in ("local", "shared"):
        metadata = json.loads((checkpoints / mode / "experiment.json").read_text())
        stats["peak_reserved_gib"] = max(e["peak_reserved_gib"] for e in metadata["training_history"])
    results[mode] = stats
    print(f"{mode:15} {stats['accuracy']:.5f} ({stats['correct']}/{stats['total']})")
deltas = {f"local_vs_{mode.replace('-', '_')}_pp": 100 * (results["local"]["accuracy"] - stats["accuracy"])
          for mode, stats in results.items() if mode != "local"}
for name, delta in deltas.items():
    print(f"{name}: {delta:+.2f} pp")
summary = run / "summary.json"
summary.write_text(json.dumps({"results": results, **deltas}, indent=2) + "\n")
print(f"Saved: {summary}")
PY
