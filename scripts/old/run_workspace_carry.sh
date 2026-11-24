#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

test_input=data/gsm-mc/test.jsonl
eval_dir=outputs/jev-workspace-carry/test
output="${1:-/dev/shm/loop-jev-$(id -u)/workspace-carry}"
for file in "$test_input" outputs/jev-latent24-ce1-final-ce-kl/eval/gsm-mc-test-loop1.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl; do
  [[ -s "$file" ]] || { echo "Missing comparison data: $file" >&2; exit 1; }
done

if [[ ! -f "$output/experiment.json" ]]; then
  uv run torchrun --standalone --nproc_per_node=2 \
    --module loop_jev.training.answer_slot_distill \
    --model Qwen/Qwen3-1.7B --train data/gsm-mc/train.jsonl --output "$output" \
    --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
    --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
    --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
    --loop-start-layer 9 --loop-end-layer 18 --loop-count 3 --loop-damping \
    --latent-slots 24 --loop-carry-workspace --tensor-parallel-size 1 --data-parallel-size 2
fi
uv run python - "$output" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
metadata = json.loads((output / "experiment.json").read_text())
expected = {"status": "complete", "loop_carry_workspace": True, "loop_count": 3,
            "loop_damping": True, "loop_start_layer": 9, "loop_end_layer": 18,
            "latent_slots": 24, "examples": 1024, "answer_supervision": "final-loop",
            "logit_loss_weight": 1.0}
if any(metadata.get(key) != value for key, value in expected.items()):
    raise ValueError(f"Unexpected checkpoint configuration: {output}")
PY

for round in 1 3; do
  uv run python -m loop_jev.evaluation.zero_shot \
    --model "$output" --input "$test_input" --output "$eval_dir/round-$round.jsonl" \
    --backend transformers --data-parallel-size 2 --batch-size 4 --loop-readout-round "$round"
done

uv run python - "$eval_dir" "$test_input" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file
from loop_jev.training.answer_slot_distill import read_rows

out, test_input = map(Path, sys.argv[1:])
expected_ids = {row["id"] for row in read_rows(test_input)}
files = {"ce_kl_1": (Path("outputs/jev-latent24-ce1-final-ce-kl/eval/gsm-mc-test-loop1.jsonl"), None),
         "ce_kl_3": (Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl"), None),
         "workspace_round_1": (out / "round-1.jsonl", 1),
         "workspace_round_3": (out / "round-3.jsonl", 3)}
results = {}
for label, (path, round) in files.items():
    score, ids = score_file(path, disable_looping=False)
    if ids != expected_ids or (round is not None and any(
        row.get("loop_readout_round") != round for row in read_rows(path)
    )):
        raise ValueError(f"Wrong evaluation set or readout round: {path}")
    results[label] = score
    print(f"{label}: {score['accuracy']:.5f} ({score['correct']}/{score['total']})")

def delta(a, b):
    return 100 * (results[a]["accuracy"] - results[b]["accuracy"])

deltas = {
    "workspace_3_vs_1_pp": delta("workspace_round_3", "workspace_round_1"),
    "workspace_3_vs_ce_kl_3_pp": delta("workspace_round_3", "ce_kl_3"),
    "workspace_3_vs_ce_kl_1_pp": delta("workspace_round_3", "ce_kl_1"),
}
for label, value in deltas.items():
    print(f"{label}: {value:+.2f} pp")
summary = out / "summary.json"
summary.write_text(json.dumps({"results": results, "deltas": deltas}, indent=2) + "\n")
print(f"Saved: {summary}")
PY
