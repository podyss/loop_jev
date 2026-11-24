#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"

test_input=data/gsm-mc/test.jsonl
train=data/gsm-mc/train.jsonl
original=outputs/jev-wide-loop-comparison/test
step="${LOOP_DAMPING_STEP:-}"
flags=()
suffix=
if [[ -n "$step" ]]; then
  [[ "$step" == 0.5 ]] || { echo "Expected LOOP_DAMPING_STEP=0.5 for the fixed-step comparison" >&2; exit 1; }
  suffix=-half
  flags=(--loop-damping-step "$step")
  if [[ ! -s "$original/native.jsonl" || ! -s "$original/no-loop.jsonl" ]]; then
    LOOP_DAMPING_STEP= bash scripts/run_wide_loop_comparison.sh
  fi
fi
run="outputs/jev-wide-loop${suffix}-comparison/test"
checkpoint="${1:-/var/tmp/loop-jev-$(id -u)/ce-kl-1024-loop3-wide${suffix}}"
for file in "$train" "$test_input" outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl; do
  [[ -s "$file" ]] || { echo "Missing experiment input: $file" >&2; exit 1; }
done

if [[ ! -f "$checkpoint/experiment.json" ]]; then
  uv run torchrun --standalone --nproc_per_node=2 \
    --module loop_jev.training.answer_slot_distill \
    --model Qwen/Qwen3-1.7B --train "$train" --output "$checkpoint" \
    --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
    --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
    --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
    --loop-start-layer 6 --loop-end-layer 21 --loop-count 3 --loop-damping --latent-slots 24 \
    --tensor-parallel-size 1 --data-parallel-size 2 "${flags[@]}"
fi

uv run python - "$checkpoint" "$train" "$step" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint, train = map(Path, sys.argv[1:3])
step = float(sys.argv[3]) if sys.argv[3] else None
metadata = json.loads((checkpoint / "experiment.json").read_text())
config = json.loads((checkpoint / "config.json").read_text())
expected = dict(status="complete", model="Qwen/Qwen3-1.7B", examples=1024, seed=42,
                completed_epochs=3, batch_size_per_gpu=4, gradient_accumulation_steps=2,
                loop_count=3, latent_slots=24, loop_damping=True, loop_schedule="block",
                loop_start_layer=6, loop_end_layer=21, answer_supervision="final-loop",
                logit_loss_weight=1.0, hidden_loss_weight=0.0,
                train_sha256=hashlib.sha256(train.read_bytes()).hexdigest())
index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
if any(not (checkpoint / shard).is_file() for shard in set(index["weight_map"].values())):
    raise ValueError(f"Checkpoint is missing weights: {checkpoint}")
if (any(metadata.get(key) != value for key, value in expected.items())
        or metadata.get("loop_damping_step") != step or config.get("loop_damping_step") != step):
    raise ValueError(f"Unexpected checkpoint configuration: {checkpoint}")
PY

for mode in native no-loop; do
  flags=()
  if [[ "$mode" == no-loop ]]; then flags+=(--disable-looping); fi
  uv run python -m loop_jev.evaluation.zero_shot --model "$checkpoint" --input "$test_input" \
    --output "$run/$mode.jsonl" --backend transformers --data-parallel-size 2 --batch-size 4 "${flags[@]}"
done
cp "$checkpoint/experiment.json" "$run/experiment.json"

uv run python - "$run" "$test_input" "$original" "$step" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file

run, test_input, original = map(Path, sys.argv[1:5])
fixed_step = bool(sys.argv[5])
expected_ids = {json.loads(line)["id"] for line in test_input.read_text().splitlines() if line}
if len(expected_ids) != 1319:
    raise ValueError("Expected 1319 unique test IDs")
results = {}
print("GSM-MC test (1319 examples)")
files = {
    "block3_native": Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl"),
    "block3_no_loop": Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl"),
    "wide3_native": run / "native.jsonl",
    "wide3_no_loop": run / "no-loop.jsonl",
}
if fixed_step:
    files.update(wide_third_native=original / "native.jsonl", wide_third_no_loop=original / "no-loop.jsonl")
for name, path in files.items():
    score, ids = score_file(path, disable_looping=name.endswith("_no_loop"))
    if ids != expected_ids:
        raise ValueError(f"Unexpected evaluation IDs: {path}")
    results[name] = score
    print(f"{name}: {score['accuracy']:.5f} ({score['correct']}/{score['total']})")

def gain(a, b):
    return 100 * (results[a]["accuracy"] - results[b]["accuracy"])

deltas = dict(native_vs_block_pp=gain("wide3_native", "block3_native"),
              no_loop_vs_block_pp=gain("wide3_no_loop", "block3_no_loop"),
              same_checkpoint_loop_gain_pp=gain("wide3_native", "wide3_no_loop"))
if fixed_step:
    deltas.update(native_vs_third_pp=gain("wide3_native", "wide_third_native"),
                  no_loop_vs_third_pp=gain("wide3_no_loop", "wide_third_no_loop"))
for name, value in deltas.items():
    print(f"{name}: {value:+.2f} pp")
metadata = json.loads((run / "experiment.json").read_text())
peak = max(epoch["peak_reserved_gib"] for epoch in metadata["training_history"])
print(f"peak_reserved_gib: {peak:.3f}")
summary = run / "summary.json"
summary.write_text(json.dumps(dict(results=results, deltas=deltas, peak_reserved_gib=peak), indent=2) + "\n")
print(f"Saved: {summary}")
PY
