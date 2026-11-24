#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"

test_input=data/gsm-mc/test.jsonl
train=data/gsm-mc/train.jsonl
run=outputs/jev-answer-refinement-comparison/test
checkpoint="${1:-/dev/shm/loop-jev-$(id -u)/answer-refinement}"
for file in "$train" "$test_input" outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl; do
  [[ -s "$file" ]] || { echo "Missing training input, test input, or baseline: $file" >&2; exit 1; }
done

if [[ ! -f "$checkpoint/experiment.json" ]]; then
  uv run torchrun --standalone --nproc_per_node=2 \
    --module loop_jev.training.answer_slot_distill \
    --model Qwen/Qwen3-1.7B --train "$train" --output "$checkpoint" \
    --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
    --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
    --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
    --loop-start-layer 9 --loop-end-layer 18 --loop-count 3 --loop-damping --latent-slots 24 \
    --loop-answer-refinement --tensor-parallel-size 1 --data-parallel-size 2
fi

uv run python - "$checkpoint" "$train" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint, train = map(Path, sys.argv[1:])
metadata = json.loads((checkpoint / "experiment.json").read_text())
expected = dict(status="complete", model="Qwen/Qwen3-1.7B", examples=1024, seed=42,
                completed_epochs=3, batch_size_per_gpu=4, gradient_accumulation_steps=2,
                loop_count=3, latent_slots=24, loop_damping=True, loop_schedule="block",
                loop_start_layer=9, loop_end_layer=18, answer_supervision="final-loop",
                logit_loss_weight=1.0, loop_answer_refinement=True,
                train_sha256=hashlib.sha256(train.read_bytes()).hexdigest())
if any(metadata.get(key) != value for key, value in expected.items()):
    raise ValueError(f"Unexpected checkpoint configuration: {checkpoint}")
PY

evaluate() {
  local mode="$1"
  shift
  uv run python -m loop_jev.evaluation.zero_shot --model "$checkpoint" --input "$test_input" \
    --output "$run/$mode.jsonl" --backend transformers --data-parallel-size 2 --batch-size 4 "$@"
}
evaluate native
evaluate no-loop --disable-looping

uv run python - "$run" "$test_input" "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file
from loop_jev.evaluation.zero_shot import read_rows

run, test_input, checkpoint = map(Path, sys.argv[1:])
expected_ids = {row["id"] for row in read_rows(test_input)}
if len(expected_ids) != 1319:
    raise ValueError("Expected 1319 unique test IDs")
results = {}
print(f"GSM-MC test ({len(expected_ids)} examples)")
for name, path, off in (
    ("block3_native", Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl"), False),
    ("block3_no_loop", Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl"), True),
    ("refine3_native", run / "native.jsonl", False),
    ("refine3_no_loop", run / "no-loop.jsonl", True),
):
    score, ids = score_file(path, disable_looping=off)
    rows = read_rows(path)
    if ids != expected_ids or any(row.get("loop_readout_round") is not None for row in rows):
        raise ValueError(f"Unexpected evaluation IDs or readout depth: {path}")
    results[name] = score
    print(f"{name}: {score['accuracy']:.5f} ({score['correct']}/{score['total']})")

def gain(a, b):
    return 100 * (results[a]["accuracy"] - results[b]["accuracy"])

deltas = dict(native_vs_block_pp=gain("refine3_native", "block3_native"),
              no_loop_vs_block_pp=gain("refine3_no_loop", "block3_no_loop"),
              same_checkpoint_loop_gain_pp=gain("refine3_native", "refine3_no_loop"))
for name, value in deltas.items():
    print(f"{name}: {value:+.2f} pp")
metadata = json.loads((checkpoint / "experiment.json").read_text())
peak = max(epoch["peak_reserved_gib"] for epoch in metadata["training_history"])
print(f"peak_reserved_gib: {peak:.3f}")
summary = run / "summary.json"
summary.write_text(json.dumps(dict(results=results, deltas=deltas, peak_reserved_gib=peak), indent=2) + "\n")
print(f"Saved: {summary}")
PY
