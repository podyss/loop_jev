#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"

train=data/gsm-mc/train.jsonl
test_input=data/gsm-mc/test.jsonl
eval_dir=outputs/jev-bidirectional-arithmetic-comparison/test
checkpoint="${1:-/dev/shm/loop-jev-$(id -u)/bidirectional-arithmetic}"
for file in "$train" "$test_input" outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl; do
  [[ -s "$file" ]] || { echo "Missing input or baseline: $file" >&2; exit 1; }
done

if [[ ! -f "$checkpoint/experiment.json" ]]; then
  uv run torchrun --standalone --nproc_per_node=2 \
    --module loop_jev.training.answer_slot_distill \
    --model Qwen/Qwen3-1.7B --train "$train" --output "$checkpoint" \
    --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
    --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
    --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
    --loop-start-layer 9 --loop-end-layer 18 --loop-count 3 --loop-damping --latent-slots 24 \
    --program-objective bidirectional --program-composition-weight 0.25 \
    --tensor-parallel-size 1 --data-parallel-size 2
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
                logit_loss_weight=1.0, program_objective="bidirectional", program_composition_weight=0.25,
                train_sha256=hashlib.sha256(train.read_bytes()).hexdigest())
if any(metadata.get(key) != value for key, value in expected.items()):
    raise ValueError(f"Unexpected checkpoint configuration: {checkpoint}")
PY

for mode in native no-loop; do
  flags=()
  if [[ "$mode" == no-loop ]]; then flags+=(--disable-looping); fi
  uv run python -m loop_jev.evaluation.zero_shot \
    --model "$checkpoint" --input "$test_input" --output "$eval_dir/$mode.jsonl" \
    --backend transformers --data-parallel-size 2 --batch-size 4 "${flags[@]}"
done

uv run python - "$eval_dir" "$test_input" "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file
from loop_jev.training.answer_slot_distill import read_rows
from loop_jev.training.bidirectional_arithmetic import compile_targets

output, test_input, checkpoint = map(Path, sys.argv[1:])
rows = read_rows(test_input)
expected_ids = {row["id"] for row in rows}
if len(rows) != 1319 or len(expected_ids) != 1319:
    raise ValueError("Expected the original 1319-example test split")
eligible = {row["id"] for row in rows if compile_targets(row) is not None}
files = {
    "block3_damped": (Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl"), False),
    "block3_damped_no_loop": (Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl"), True),
    "bidirectional3": (output / "native.jsonl", False),
    "bidirectional3_no_loop": (output / "no-loop.jsonl", True),
}
results, subsets = {}, {}
print(f"GSM-MC test ({len(rows)} examples; {len(eligible)} with verified bidirectional targets)")
for label, (path, off) in files.items():
    score, ids = score_file(path, disable_looping=off)
    if ids != expected_ids:
        raise ValueError(f"Wrong evaluation set: {path}")
    results[label] = score
    count = sum(row["correct_prediction"] for row in read_rows(path) if row["id"] in eligible)
    subsets[label] = dict(correct=count, total=len(eligible))
    print(f"{label}: {score['accuracy']:.5f} ({score['correct']}/{score['total']}); verified subset: {count}/{len(eligible)}")

def delta(a, b):
    return 100 * (results[a]["accuracy"] - results[b]["accuracy"])

deltas = dict(bidirectional_vs_block_pp=delta("bidirectional3", "block3_damped"),
              bidirectional_no_loop_vs_block_no_loop_pp=delta("bidirectional3_no_loop", "block3_damped_no_loop"),
              bidirectional_same_checkpoint_loop_gain_pp=delta("bidirectional3", "bidirectional3_no_loop"))
for label, value in deltas.items():
    print(f"{label}: {value:+.2f} pp")
metadata = json.loads((checkpoint / "experiment.json").read_text())
peak = max(epoch["peak_reserved_gib"] for epoch in metadata["training_history"])
print(f"train target coverage: {metadata['program_composition_coverage']}/1024; peak_reserved_gib: {peak:.3f}")
summary = output / "summary.json"
summary.write_text(json.dumps(dict(results=results, verified_subset=subsets, deltas=deltas,
                                   train_coverage=metadata["program_composition_coverage"],
                                   peak_reserved_gib=peak), indent=2) + "\n")
print(f"Saved: {summary}")
PY
