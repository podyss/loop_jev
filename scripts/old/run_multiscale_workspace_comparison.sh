#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"

train=data/gsm-mc/train.jsonl
test_input=data/gsm-mc/test.jsonl
eval_dir=outputs/jev-multiscale-workspace-comparison/test
checkpoints="${1:-/dev/shm/loop-jev-$(id -u)/multiscale-workspace}"
for file in "$train" "$test_input" outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl; do
  [[ -s "$file" ]] || { echo "Missing input or baseline: $file" >&2; exit 1; }
done

for mode in up up-down; do
  output="$checkpoints/$mode"
  if [[ ! -f "$output/experiment.json" ]]; then
    uv run torchrun --standalone --nproc_per_node=2 \
      --module loop_jev.training.answer_slot_distill \
      --model Qwen/Qwen3-1.7B --train "$train" --output "$output" \
      --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
      --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
      --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
      --loop-start-layer 9 --loop-end-layer 18 --loop-count 3 --loop-damping --latent-slots 24 \
      --loop-multiscale-workspace "$mode" --tensor-parallel-size 1 --data-parallel-size 2
  fi

  uv run python - "$output" "$train" "$mode" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint, train = map(Path, sys.argv[1:3])
mode = sys.argv[3]
metadata = json.loads((checkpoint / "experiment.json").read_text())
config = json.loads((checkpoint / "config.json").read_text())
expected = dict(status="complete", model="Qwen/Qwen3-1.7B", examples=1024, seed=42,
                completed_epochs=3, batch_size_per_gpu=4, gradient_accumulation_steps=2,
                loop_count=3, latent_slots=24, loop_damping=True, loop_schedule="block",
                loop_start_layer=9, loop_end_layer=18, answer_supervision="final-loop",
                logit_loss_weight=1.0, loop_multiscale_workspace=mode,
                train_sha256=hashlib.sha256(train.read_bytes()).hexdigest())
if (any(metadata.get(key) != value for key, value in expected.items())
        or config.get("loop_multiscale_workspace") != mode):
    raise ValueError(f"Unexpected checkpoint configuration: {checkpoint}")
PY

  for readout in native no-loop; do
    flags=()
    if [[ "$readout" == no-loop ]]; then flags+=(--disable-looping); fi
    uv run python -m loop_jev.evaluation.zero_shot \
      --model "$output" --input "$test_input" --output "$eval_dir/$mode-$readout.jsonl" \
      --backend transformers --data-parallel-size 2 --batch-size 4 "${flags[@]}"
  done
done

uv run python - "$eval_dir" "$test_input" "$checkpoints" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file

output, test_input, checkpoints = map(Path, sys.argv[1:])
expected_ids = {json.loads(line)["id"] for line in test_input.read_text().splitlines() if line.strip()}
if len(expected_ids) != 1319:
    raise ValueError("Expected the original 1319-example test split")
files = {
    "block3_damped": (Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl"), False),
    "block3_damped_no_loop": (Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl"), True),
    "up3": (output / "up-native.jsonl", False),
    "up3_no_loop": (output / "up-no-loop.jsonl", True),
    "up_down3": (output / "up-down-native.jsonl", False),
    "up_down3_no_loop": (output / "up-down-no-loop.jsonl", True),
}
results = {}
print(f"GSM-MC test ({len(expected_ids)} examples)")
for label, (path, off) in files.items():
    score, ids = score_file(path, disable_looping=off)
    if ids != expected_ids:
        raise ValueError(f"Wrong evaluation set: {path}")
    results[label] = score
    print(f"{label}: {score['accuracy']:.5f} ({score['correct']}/{score['total']})")

def delta(a, b):
    return 100 * (results[a]["accuracy"] - results[b]["accuracy"])

deltas = {
    "feedback_gain_pp": delta("up_down3", "up3"),
    "aggregation_vs_damped_block_pp": delta("up3", "block3_damped"),
    "multiscale_vs_damped_block_pp": delta("up_down3", "block3_damped"),
    "multiscale_same_checkpoint_loop_gain_pp": delta("up_down3", "up_down3_no_loop"),
    "multiscale_no_loop_vs_block_no_loop_pp": delta("up_down3_no_loop", "block3_damped_no_loop"),
}
for label, value in deltas.items():
    print(f"{label}: {value:+.2f} pp")

memory = {}
for mode in ("up", "up-down"):
    metadata = json.loads((checkpoints / mode / "experiment.json").read_text())
    memory[mode] = max(epoch["peak_reserved_gib"] for epoch in metadata["training_history"])
    print(f"{mode} peak_reserved_gib: {memory[mode]:.3f}")
summary = output / "summary.json"
summary.write_text(json.dumps(dict(results=results, deltas=deltas, peak_reserved_gib=memory), indent=2) + "\n")
print(f"Saved: {summary}")
PY
