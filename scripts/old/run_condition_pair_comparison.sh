#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"

test_input=data/gsm-mc/test.jsonl
run=outputs/jev-condition-pair-comparison
checkpoint="${1:-/dev/shm/loop-jev-$(id -u)/condition-pairs}"
baseline=outputs/jev-latent24-ce3-final-ce-kl
for file in "$test_input" outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl "$baseline/config.json"; do
  [[ -s "$file" ]] || { echo "Missing test input or baseline: $file" >&2; exit 1; }
done

uv run python -m loop_jev.training.condition_pairs --train data/gsm-mc/train.jsonl \
  --evaluation-input "$test_input" --output "$run/data"

if [[ ! -f "$checkpoint/experiment.json" ]]; then
  uv run torchrun --standalone --nproc_per_node=2 \
    --module loop_jev.training.answer_slot_distill \
    --model Qwen/Qwen3-1.7B --train "$run/data/train.jsonl" --output "$checkpoint" \
    --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
    --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
    --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
    --loop-start-layer 9 --loop-end-layer 18 --loop-count 3 --loop-damping --latent-slots 24 \
    --condition-pair-weight 0.25 --tensor-parallel-size 1 --data-parallel-size 2
fi

uv run python - "$checkpoint" "$run/data/train.jsonl" <<'PY'
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
                logit_loss_weight=1.0, condition_pair_weight=0.25,
                train_sha256=hashlib.sha256(train.read_bytes()).hexdigest())
if any(metadata.get(key) != value for key, value in expected.items()):
    raise ValueError(f"Unexpected checkpoint configuration: {checkpoint}")
PY

evaluate() {
  local model="$1" data="$2" output="$3" mode="$4"
  local -a flags=()
  if [[ "$mode" == no-loop ]]; then flags+=(--disable-looping); fi
  uv run python -m loop_jev.evaluation.zero_shot --model "$model" --input "$data" --output "$output" \
    --backend transformers --data-parallel-size 2 --batch-size 4 "${flags[@]}"
}

for mode in native no-loop; do
  evaluate "$checkpoint" "$test_input" "$run/test/$mode.jsonl" "$mode"
  evaluate "$checkpoint" "$run/data/pairs.jsonl" "$run/pairs/condition3-$mode.jsonl" "$mode"
  evaluate "$baseline" "$run/data/pairs.jsonl" "$run/pairs/block3-$mode.jsonl" "$mode"
done

uv run python - "$run" "$test_input" "$checkpoint" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file
from loop_jev.training.answer_slot_distill import read_rows

run, test_input, checkpoint = map(Path, sys.argv[1:])
expected_ids = {row["id"] for row in read_rows(test_input)}
pair_rows = read_rows(run / "data/pairs.jsonl")
pair_ids = {row["id"] for row in pair_rows}
if len(expected_ids) != 1319 or len(pair_ids) != len(pair_rows) or len(pair_rows) % 2:
    raise ValueError("Unexpected test or matched condition split")

results, pair_results = {}, {}
print(f"GSM-MC test ({len(expected_ids)} examples); held-out condition pairs ({len(pair_rows) // 2})")
for model in ("block3", "condition3"):
    for mode in ("native", "no-loop"):
        off = mode == "no-loop"
        label = f"{model}_{mode.replace('-', '_')}"
        original = (run / "test" / f"{mode}.jsonl" if model == "condition3" else
                    Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl" if off else
                         "outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl"))
        score, ids = score_file(original, disable_looping=off)
        path = run / "pairs" / f"{model}-{mode}.jsonl"
        _, evaluated_pair_ids = score_file(path, disable_looping=off)
        if ids != expected_ids or pair_ids != evaluated_pair_ids:
            raise ValueError(f"Wrong evaluation split: {label}")
        results[label] = score
        paired = defaultdict(dict)
        for row in read_rows(path):
            paired[row["pair_id"]][row["variant"]] = row["correct_prediction"]
        if any(set(pair) != {"original", "changed"} for pair in paired.values()):
            raise ValueError(f"Unmatched condition pair: {path}")
        both = sum(all(pair.values()) for pair in paired.values())
        pair_results[label] = dict(correct=both, total=len(paired), accuracy=both / len(paired),
                                   original_correct=sum(p["original"] for p in paired.values()),
                                   changed_correct=sum(p["changed"] for p in paired.values()))
        print(f"{label}: {score['accuracy']:.5f} ({score['correct']}/{score['total']}); "
              f"both pair members correct: {both}/{len(paired)}")

def gain(data, a, b):
    return 100 * (data[a]["accuracy"] - data[b]["accuracy"])

deltas = dict(test_vs_block_pp=gain(results, "condition3_native", "block3_native"),
              test_loop_gain_pp=gain(results, "condition3_native", "condition3_no_loop"),
              test_no_loop_vs_block_pp=gain(results, "condition3_no_loop", "block3_no_loop"),
              pair_vs_block_pp=gain(pair_results, "condition3_native", "block3_native"),
              pair_loop_gain_pp=gain(pair_results, "condition3_native", "condition3_no_loop"))
for key, value in deltas.items():
    print(f"{key}: {value:+.2f} pp")
metadata = json.loads((checkpoint / "experiment.json").read_text())
peak = max(e["peak_reserved_gib"] for e in metadata["training_history"])
print(f"peak_reserved_gib: {peak:.3f}")
summary = run / "test/summary.json"
summary.write_text(json.dumps(dict(results=results, condition_pairs=pair_results,
                                   deltas=deltas, peak_reserved_gib=peak), indent=2) + "\n")
print(f"Saved: {summary}")
PY
