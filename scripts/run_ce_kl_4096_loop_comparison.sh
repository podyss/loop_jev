#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"

root=outputs/jev-ce-kl-4096-comparison
prior=outputs/jev-loop-cot-comparison
train="$root/train.jsonl"
validation="$prior/validation.jsonl"
checkpoint_root="${1:-/tmp/loop-jev-$(id -u)/ce-kl-4096}"
extra_root="${2:-/dev/shm/loop-jev-$(id -u)/ce-kl-4096}"
experiments=(loop1 loop3 loop2 loop3-late)
declare -A rounds=([loop1]=1 [loop3]=3 [loop2]=2 [loop3-late]=3)
declare -A starts=([loop1]=9 [loop3]=9 [loop2]=9 [loop3-late]=14)
declare -A roots=([loop1]="$checkpoint_root" [loop3]="$checkpoint_root"
                  [loop2]="$extra_root" [loop3-late]="$extra_root")

uv run python - "$train" "$validation" <<'PY'
import json
import random
import sys
from pathlib import Path

from loop_jev.training.answer_slot_distill import read_rows

train, validation = map(Path, sys.argv[1:])
rows = read_rows(Path("data/gsm-mc/train.jsonl"))
held_out = {row["id"] for row in read_rows(validation)}
original = random.Random(42).sample(range(len(rows)), 1024)
assert len(held_out) == 1024 and not held_out.intersection(rows[i]["id"] for i in original)
original_set = set(original)
available = [i for i, row in enumerate(rows) if i not in original_set and row["id"] not in held_out]
indices = original + random.Random(43).sample(available, 3072)
assert len({rows[i]["id"] for i in indices}) == 4096
train.parent.mkdir(parents=True, exist_ok=True)
train.write_text("".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in indices), encoding="utf-8")
print(f"Prepared {len(indices)} training examples, excluding {len(held_out)} validation examples")
PY

mkdir -p "$checkpoint_root" "$extra_root"
declare -A missing_by_device=()
for experiment in "${experiments[@]}"; do
  base="${roots[$experiment]}"
  if [[ -f "$base/$experiment/experiment.json" ]]; then continue; fi
  device=$(stat -c %d -- "$base")
  missing_by_device[$device]=$(( ${missing_by_device[$device]:-0} + 1 ))
done
for base in "$checkpoint_root" "$extra_root"; do
  device=$(stat -c %d -- "$base")
  missing="${missing_by_device[$device]:-0}"
  available=$(df --output=avail -k "$base" | tail -n 1)
  if (( available < missing * 8 * 1024 * 1024 )); then
    echo "Need about $((missing * 8)) GiB free on the filesystem at $base" >&2
    exit 1
  fi
done

for experiment in "${experiments[@]}"; do
  loops="${rounds[$experiment]}"
  start="${starts[$experiment]}"
  end=$((start + 9))
  output="${roots[$experiment]}/$experiment"
  echo "=== $experiment: $loops round(s), layers [$start,$end), checkpoint $output ==="
  if [[ ! -f "$output/experiment.json" ]]; then
    uv run torchrun --standalone --nproc_per_node=2 \
      --module loop_jev.training.answer_slot_distill \
      --model Qwen/Qwen3-1.7B --train "$train" --output "$output" \
      --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
      --max-length 16384 --max-train-examples 4096 --seed 42 --temperature 2.0 \
      --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
      --loop-start-layer "$start" --loop-end-layer "$end" --loop-count "$loops" --loop-damping \
      --latent-slots 24 --tensor-parallel-size 1 --data-parallel-size 2
  fi
  uv run python - "$output/experiment.json" "$train" "$loops" "$start" "$end" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

metadata_path, train = map(Path, sys.argv[1:3])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
expected = {"status": "complete", "examples": 4096, "loop_count": int(sys.argv[3]),
            "loop_start_layer": int(sys.argv[4]), "loop_end_layer": int(sys.argv[5]),
            "latent_slots": 24, "loop_damping": True, "logit_loss_weight": 1.0,
            "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest()}
if any(metadata.get(k) != v for k, v in expected.items()):
    raise ValueError(f"Checkpoint does not match this experiment: {metadata_path}")
PY
  modes=(native)
  if (( loops > 1 )); then modes+=(no-loop); fi
  for mode in "${modes[@]}"; do
    extra=()
    if [[ "$mode" == no-loop ]]; then extra+=(--disable-looping); fi
    uv run python -m loop_jev.evaluation.zero_shot \
      --model "$output" --input "$validation" --output "$root/validation/$experiment-$mode.jsonl" \
      --backend transformers --data-parallel-size 2 --batch-size 4 "${extra[@]}"
  done
done

uv run python - "$root/validation" "$prior/validation" "$validation" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file
from loop_jev.training.answer_slot_distill import read_rows

output, prior, validation = map(Path, sys.argv[1:])
expected_ids = {row["id"] for row in read_rows(validation)}
files = {
    "1024_loop1": prior / "ce1-native.jsonl",
    "1024_loop3": prior / "ce3-native.jsonl",
    "4096_loop1": output / "loop1-native.jsonl",
    "4096_loop3": output / "loop3-native.jsonl",
    "4096_loop3_no_loop": output / "loop3-no-loop.jsonl",
    "4096_loop2": output / "loop2-native.jsonl",
    "4096_loop2_no_loop": output / "loop2-no-loop.jsonl",
    "4096_loop3_late": output / "loop3-late-native.jsonl",
    "4096_loop3_late_no_loop": output / "loop3-late-no-loop.jsonl",
}
results = {}
for label, path in files.items():
    score, ids = score_file(path, disable_looping=label.endswith("no_loop"))
    if ids != expected_ids:
        raise ValueError(f"Wrong evaluation set: {path}")
    results[label] = score
    print(f"{label}: {score['accuracy']:.5f} ({score['correct']}/{score['total']})")

def delta(a, b):
    return 100 * (results[a]["accuracy"] - results[b]["accuracy"])

deltas = {
    "loop_gain_at_1024_pp": delta("1024_loop3", "1024_loop1"),
    "loop_gain_at_4096_pp": delta("4096_loop3", "4096_loop1"),
    "more_data_gain_loop1_pp": delta("4096_loop1", "1024_loop1"),
    "more_data_gain_loop3_pp": delta("4096_loop3", "1024_loop3"),
    "same_checkpoint_loop_gain_pp": delta("4096_loop3", "4096_loop3_no_loop"),
    "two_rounds_vs_one_pp": delta("4096_loop2", "4096_loop1"),
    "three_rounds_vs_two_pp": delta("4096_loop3", "4096_loop2"),
    "later_block_vs_middle_at_three_pp": delta("4096_loop3_late", "4096_loop3"),
    "later_block_vs_one_pp": delta("4096_loop3_late", "4096_loop1"),
}
for label, value in deltas.items():
    print(f"{label}: {value:+.2f} pp")
summary = output / "summary.json"
summary.write_text(json.dumps({"results": results, "deltas": deltas}, indent=2) + "\n", encoding="utf-8")
print(f"Saved: {summary}")
PY
