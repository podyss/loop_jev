#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

train=data/gsm-mc/train.jsonl
test_input=data/gsm-mc/test.jsonl
eval_dir=outputs/jev-program-composition-comparison/test
checkpoints="${1:-/dev/shm/loop-jev-$(id -u)/program-composition}"
for file in "$train" "$test_input" outputs/jev-latent24-ce1-final-ce-kl/eval/gsm-mc-test-loop1.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl; do
  [[ -s "$file" ]] || { echo "Missing input or baseline: $file" >&2; exit 1; }
done

# This control and program-1 use the same standard prompt, without latent slots.
for experiment in direct-ce-kl program-1 program-3; do
  output="$checkpoints/$experiment"
  loops=1
  slots=0
  weight=0.25
  if [[ "$experiment" == direct-ce-kl ]]; then weight=0; fi
  if [[ "$experiment" == program-3 ]]; then
    loops=3
    slots=24
  fi
  if [[ ! -f "$output/experiment.json" ]]; then
    uv run torchrun --standalone --nproc_per_node=2 \
      --module loop_jev.training.answer_slot_distill \
      --model Qwen/Qwen3-1.7B --train "$train" --output "$output" \
      --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
      --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
      --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
      --loop-start-layer 9 --loop-end-layer 18 --loop-count "$loops" --loop-damping \
      --latent-slots "$slots" --program-composition-weight "$weight" \
      --tensor-parallel-size 1 --data-parallel-size 2
  fi

  uv run python - "$output/experiment.json" "$train" "$loops" "$slots" "$weight" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

meta_path, train, loops, slots, weight = sys.argv[1:]
metadata = json.loads(Path(meta_path).read_text())
expected = dict(status="complete", model="Qwen/Qwen3-1.7B", examples=1024, seed=42,
                completed_epochs=3, batch_size_per_gpu=4, gradient_accumulation_steps=2,
                loop_count=int(loops), latent_slots=int(slots), loop_damping=True,
                loop_start_layer=9, loop_end_layer=18, answer_supervision="final-loop",
                logit_loss_weight=1.0, program_composition_weight=float(weight),
                train_sha256=hashlib.sha256(Path(train).read_bytes()).hexdigest())
if any(metadata.get(key) != value for key, value in expected.items()):
    raise ValueError(f"Unexpected checkpoint configuration: {meta_path}")
PY

  modes=(native)
  if [[ "$loops" == 3 ]]; then modes+=(no-loop); fi
  for mode in "${modes[@]}"; do
    flags=()
    if [[ "$mode" == no-loop ]]; then flags+=(--disable-looping); fi
    uv run python -m loop_jev.evaluation.zero_shot \
      --model "$output" --input "$test_input" --output "$eval_dir/$experiment-$mode.jsonl" \
      --backend transformers --data-parallel-size 2 --batch-size 4 "${flags[@]}"
  done
done

uv run python - "$eval_dir" "$test_input" "$train" "$checkpoints" <<'PY'
import json
import random
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file
from loop_jev.training.answer_slot_distill import read_rows
from loop_jev.training.program_composition import compile_program

output, test_input, train, checkpoints = map(Path, sys.argv[1:])
test_rows = read_rows(test_input)
expected_ids = {row["id"] for row in test_rows}
train_rows = read_rows(train)
selected = random.Random(42).sample(train_rows, 1024)
if len(expected_ids) != 1319 or expected_ids.intersection(row["id"] for row in selected):
    raise ValueError("Expected the original disjoint 1319-example test split")
seen = {program.signature for row in selected if (program := compile_program(row))}
unseen_ids = {row["id"] for row in test_rows
              if (program := compile_program(row)) and program.signature not in seen}

files = {
    "ce_kl_1_with_slots": (Path("outputs/jev-latent24-ce1-final-ce-kl/eval/gsm-mc-test-loop1.jsonl"), False),
    "ce_kl_3": (Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl"), False),
    "ce_kl_3_no_loop": (Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl"), True),
    "direct_ce_kl": (output / "direct-ce-kl-native.jsonl", False),
    "program_1": (output / "program-1-native.jsonl", False),
    "program_3": (output / "program-3-native.jsonl", False),
    "program_3_no_loop": (output / "program-3-no-loop.jsonl", True),
}
results, unseen = {}, {}
print(f"GSM-MC test ({len(expected_ids)} examples; {len(unseen_ids)} with unseen operator trees)")
for label, (path, off) in files.items():
    score, ids = score_file(path, disable_looping=off)
    if ids != expected_ids:
        raise ValueError(f"Wrong evaluation set: {path}")
    results[label] = score
    correct = sum(row["correct_prediction"] for row in read_rows(path) if row["id"] in unseen_ids)
    unseen[label] = dict(correct=correct, total=len(unseen_ids),
                         accuracy=correct / len(unseen_ids) if unseen_ids else None)
    print(f"{label}: {score['accuracy']:.5f} ({score['correct']}/{score['total']}); "
          f"unseen trees: {correct}/{len(unseen_ids)}")

def delta(a, b):
    return 100 * (results[a]["accuracy"] - results[b]["accuracy"])

deltas = {
    "program_gain_at_1_loop_pp": delta("program_1", "direct_ce_kl"),
    "program_gain_at_3_loops_pp": delta("program_3", "ce_kl_3"),
    "program_3_vs_program_1_pp": delta("program_3", "program_1"),
    "program_3_same_checkpoint_loop_gain_pp": delta("program_3", "program_3_no_loop"),
    "program_3_no_loop_vs_ce_kl_3_no_loop_pp": delta("program_3_no_loop", "ce_kl_3_no_loop"),
}
for label, value in deltas.items():
    print(f"{label}: {value:+.2f} pp")

training = {}
for experiment in ("direct-ce-kl", "program-1", "program-3"):
    metadata = json.loads((checkpoints / experiment / "experiment.json").read_text())
    training[experiment] = dict(covered_examples=metadata["program_composition_coverage"],
                                epochs=[e.get("program_composition") for e in metadata["training_history"]],
                                peak_reserved_gib=max(e["peak_reserved_gib"] for e in metadata["training_history"]))
    print(f"{experiment} peak_reserved_gib: {training[experiment]['peak_reserved_gib']:.3f}")
summary = output / "summary.json"
summary.write_text(json.dumps(dict(results=results, unseen_operator_tree=unseen,
                                   deltas=deltas, training=training), indent=2) + "\n")
print(f"Saved: {summary}")
PY
