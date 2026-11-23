#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
initial="${LOOP_TEACHER_MODEL:-/var/tmp/loop-jev-$(id -u)/ce-kl-1024-block10-seed43-step0.1}"
root=outputs/jev-block10-loop-removal
checkpoints="${LOOP_REMOVAL_CHECKPOINT_ROOT:-$root/checkpoints}"
train=data/gsm-mc/train.jsonl
test_input=data/gsm-mc/test.jsonl

test -s "$initial/config.json" || { echo "Missing loop checkpoint: $initial" >&2; exit 1; }
test -s "$train"
test -s "$test_input"
mkdir -p "$root/test" "$checkpoints"

needed=0
for mode in abrupt gradual; do
  if [[ ! -s "$checkpoints/$mode/experiment.json" ]]; then
    needed=$((needed + 8 * 1024 * 1024))
  fi
done
available=$(df --output=avail -k "$checkpoints" | tail -n 1)
if (( available < needed )); then
  echo "Need $((needed / 1024 / 1024)) GiB free at $checkpoints" >&2
  exit 1
fi

evaluate() {
  local model="$1" arm="$2"
  shift 2
  uv run python -m loop_jev.evaluation.zero_shot --model "$model" \
    --input "$test_input" --output "$root/test/$arm.jsonl" --backend transformers \
    --data-parallel-size 2 --batch-size 4 "$@"
}

evaluate "$initial" direct --disable-looping --disable-latent-slots
evaluate "$initial" original-loop10
for mode in abrupt gradual; do
  checkpoint="$checkpoints/$mode"
  if [[ ! -s "$checkpoint/experiment.json" ]]; then
    uv run torchrun --standalone --nproc_per_node=2 --module loop_jev.training.answer_slot_distill \
      --model "$initial" --train "$train" --output "$checkpoint" --loop-removal "$mode" \
      --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
      --max-length 16384 --max-train-examples 1024 --seed 43 --train-sample-seed 42 \
      --answer-supervision final-loop --loop-start-layer 9 --loop-end-layer 18 \
      --loop-count 10 --loop-damping --loop-damping-step 0.1 --latent-slots 24 \
      --hidden-loss-weight 0 --process-loss-weight 0 --logit-loss-weight 0 \
      --tensor-parallel-size 1 --data-parallel-size 2
  fi
  uv run python - "$checkpoint" "$initial" "$mode" <<'PY'
import json
import sys
from pathlib import Path

checkpoint, initial, mode = sys.argv[1:]
root = Path(checkpoint)
config = json.loads((root / "config.json").read_text())
run = json.loads((root / "experiment.json").read_text())
if (config.get("loop_count") != 1 or config.get("latent_slots") != 0 or config.get("loop_damping")
        or run["status"] != "complete" or run["model"] != initial or run["loop_removal"] != mode
        or (run["examples"], run["optimizer_steps"]) != (1024, 192)):
    raise ValueError(f"Unexpected no-loop checkpoint: {checkpoint}")
PY
  evaluate "$checkpoint" "$mode"
done

uv run python - "$root/test" "$test_input" <<'PY'
import json
import math
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file

root, test_input = map(Path, sys.argv[1:])
expected = {json.loads(line)["id"] for line in test_input.read_text().splitlines() if line}
if len(expected) != 1319:
    raise ValueError("Expected 1319 GSM-MC test IDs")
paths = {arm: (root / f"{arm}.jsonl", arm == "direct")
         for arm in ("direct", "abrupt", "gradual", "original-loop10")}
paths.update({
    "base-ce": (Path("outputs/jev-block10-student-distill/test/ce.jsonl"), True),
    "state-delta-kd": (Path("outputs/jev-block10-state-distill/test/state-delta-kd.jsonl"), True),
})
results, correctness = {}, {}
for arm, (path, disable_looping) in paths.items():
    result, ids = score_file(path, disable_looping=disable_looping)
    if ids != expected:
        raise ValueError(f"Unexpected test IDs for {arm}")
    results[arm] = result
    correctness[arm] = {r["id"]: r["correct_prediction"] for r in map(json.loads, path.read_text().splitlines())}
    print(f"{arm:16} {result['accuracy']:.5f} ({result['correct']}/{result['total']})")

comparisons = {}
for arm, control in (("gradual", "abrupt"), ("abrupt", "direct"), ("gradual", "direct"),
                     ("abrupt", "base-ce"), ("gradual", "base-ce"),
                     ("abrupt", "original-loop10"), ("gradual", "original-loop10"),
                     ("abrupt", "state-delta-kd"), ("gradual", "state-delta-kd")):
    a, b = correctness[arm], correctness[control]
    wins = sum(a[i] and not b[i] for i in expected)
    losses = sum(b[i] and not a[i] for i in expected)
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / (1 << n))
    delta = 100 * (wins - losses) / len(expected)
    comparisons[f"{arm}_vs_{control}"] = dict(delta_pp=delta, wins=wins, losses=losses, exact_mcnemar_p=p)
    print(f"{arm} vs {control}: {delta:+.2f} pp ({wins} wins, {losses} losses)")
summary = root / "summary.json"
summary.write_text(json.dumps({"results": results, "paired": comparisons}, indent=2) + "\n")
print(f"Saved: {summary}")
PY
