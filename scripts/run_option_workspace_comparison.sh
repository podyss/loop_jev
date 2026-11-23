#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

test_input=data/gsm-mc/test.jsonl
eval_dir=outputs/jev-option-workspace-comparison/test
output="${1:-/dev/shm/loop-jev-$(id -u)/ce-kl-1024-option-workspace3}"
for file in "$test_input" outputs/jev-latent24-ce1-final-ce-kl/eval/gsm-mc-test-loop1.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl; do
  [[ -s "$file" ]] || { echo "Missing test input or baseline: $file" >&2; exit 1; }
done

if [[ ! -f "$output/experiment.json" ]]; then
  uv run torchrun --standalone --nproc_per_node=2 \
    --module loop_jev.training.answer_slot_distill \
    --model Qwen/Qwen3-1.7B --train data/gsm-mc/train.jsonl --output "$output" \
    --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
    --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
    --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
    --loop-start-layer 9 --loop-end-layer 18 --loop-count 3 --loop-damping \
    --latent-slots 24 --option-workspace --tensor-parallel-size 1 --data-parallel-size 2
fi

uv run python - "$output/experiment.json" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())
if (metadata.get("status") != "complete" or not metadata.get("option_workspace")
        or metadata.get("examples") != 1024 or metadata.get("loop_count") != 3):
    raise ValueError(f"Expected a completed 1024-example three-round option workspace run: {sys.argv[1]}")
PY

for mode in native no-loop; do
  flags=()
  if [[ "$mode" == no-loop ]]; then flags+=(--disable-looping); fi
  uv run python -m loop_jev.evaluation.zero_shot \
    --model "$output" --input "$test_input" --output "$eval_dir/option3-$mode.jsonl" \
    --backend transformers --data-parallel-size 2 --batch-size 4 "${flags[@]}"
done

uv run python - "$eval_dir" "$test_input" "$output" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file
from loop_jev.training.answer_slot_distill import read_rows

output, test_input, checkpoint = map(Path, sys.argv[1:])
expected_ids = {row["id"] for row in read_rows(test_input)}
if len(expected_ids) != 1319:
    raise ValueError(f"Expected the original 1319-example test split: {test_input}")
files = {
    "ce_kl_1": (Path("outputs/jev-latent24-ce1-final-ce-kl/eval/gsm-mc-test-loop1.jsonl"), False),
    "block3_damped": (Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl"), False),
    "block3_damped_no_loop": (Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl"), True),
    "option3": (output / "option3-native.jsonl", False),
    "option3_no_loop": (output / "option3-no-loop.jsonl", True),
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
    "option_vs_damped_block_pp": delta("option3", "block3_damped"),
    "option_vs_single_pass_pp": delta("option3", "ce_kl_1"),
    "option_same_checkpoint_loop_gain_pp": delta("option3", "option3_no_loop"),
}
for label, value in deltas.items():
    print(f"{label}: {value:+.2f} pp")
metadata = json.loads((checkpoint / "experiment.json").read_text())
peak = max(epoch["peak_reserved_gib"] for epoch in metadata["training_history"])
print(f"peak_reserved_gib: {peak:.3f}")
summary = output / "summary.json"
summary.write_text(json.dumps({"results": results, "deltas": deltas, "peak_reserved_gib": peak}, indent=2) + "\n")
print(f"Saved: {summary}")
PY
