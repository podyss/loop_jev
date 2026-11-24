#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"

train=data/gsm-mc/train.jsonl
up=outputs/jev-multiscale-workspace-comparison/test
test_input=data/gsm-mc/test.jsonl
run=outputs/jev-evidence-reread-comparison/test
checkpoints="${1:-/var/tmp/loop-jev-$(id -u)/evidence-reread}"
if [[ ! -s "$up/up-native.jsonl" || ! -s "$up/up-no-loop.jsonl" ]]; then
  bash scripts/run_multiscale_workspace_comparison.sh
fi
for file in "$train" "$test_input" outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl "$up"/up-{native,no-loop}.jsonl; do
  [[ -s "$file" ]] || { echo "Missing input or baseline: $file" >&2; exit 1; }
done

for mode in reread reread-up; do
  checkpoint="$checkpoints/$mode"
  extra=()
  if [[ "$mode" == reread-up ]]; then extra+=(--loop-multiscale-workspace up); fi
  if [[ ! -f "$checkpoint/experiment.json" ]]; then
    uv run torchrun --standalone --nproc_per_node=2 \
      --module loop_jev.training.answer_slot_distill \
      --model Qwen/Qwen3-1.7B --train "$train" --output "$checkpoint" \
      --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
      --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
      --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
      --loop-start-layer 9 --loop-end-layer 18 --loop-count 3 --loop-damping --latent-slots 24 \
      --loop-evidence-reread --tensor-parallel-size 1 --data-parallel-size 2 "${extra[@]}"
  fi

  uv run python - "$checkpoint" "$train" "$mode" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint, train = map(Path, sys.argv[1:3])
scale = "up" if sys.argv[3] == "reread-up" else None
metadata = json.loads((checkpoint / "experiment.json").read_text())
config = json.loads((checkpoint / "config.json").read_text())
expected = dict(status="complete", model="Qwen/Qwen3-1.7B", examples=1024, seed=42,
                completed_epochs=3, batch_size_per_gpu=4, gradient_accumulation_steps=2,
                loop_count=3, latent_slots=24, loop_damping=True, loop_schedule="block",
                loop_start_layer=9, loop_end_layer=18, answer_supervision="final-loop",
                logit_loss_weight=1.0, loop_evidence_reread=True, loop_multiscale_workspace=scale,
                train_sha256=hashlib.sha256(train.read_bytes()).hexdigest())
if (any(metadata.get(key) != value for key, value in expected.items())
        or config.get("loop_evidence_reread") is not True
        or config.get("loop_multiscale_workspace") != scale):
    raise ValueError(f"Unexpected checkpoint configuration: {checkpoint}")
PY

  for readout in native no-reread no-loop; do
    flags=()
    case "$readout" in
      no-reread) flags+=(--disable-evidence-reread) ;;
      no-loop) flags+=(--disable-looping) ;;
    esac
    uv run python -m loop_jev.evaluation.zero_shot --model "$checkpoint" --input "$test_input" \
      --output "$run/$mode-$readout.jsonl" --backend transformers --data-parallel-size 2 --batch-size 4 "${flags[@]}"
  done
  cp "$checkpoint/experiment.json" "$run/$mode-experiment.json"
done

uv run python - "$run" "$test_input" "$up" "$checkpoints" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file
from loop_jev.evaluation.zero_shot import read_rows

run, test_input, up, checkpoints = map(Path, sys.argv[1:])
expected_ids = {row["id"] for row in read_rows(test_input)}
if len(expected_ids) != 1319:
    raise ValueError("Expected 1319 unique test IDs")
files = {
    "block3_native": Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl"),
    "block3_no_loop": Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl"),
    "up3_native": up / "up-native.jsonl",
    "up3_no_loop": up / "up-no-loop.jsonl",
}
for mode in ("reread", "reread-up"):
    for readout in ("native", "no-reread", "no-loop"):
        files[f"{mode}_{readout.replace('-', '_')}"] = run / f"{mode}-{readout}.jsonl"
results = {}
print(f"GSM-MC test ({len(expected_ids)} examples)")
for name, path in files.items():
    score, ids = score_file(path, disable_looping=name.endswith("_no_loop"))
    rows = read_rows(path)
    if ids != expected_ids or any(row.get("loop_readout_round") is not None or
                                  row.get("disable_evidence_reread", False) != name.endswith("_no_reread")
                                  for row in rows):
        raise ValueError(f"Unexpected evaluation IDs or mode: {path}")
    results[name] = score
    print(f"{name}: {score['accuracy']:.5f} ({score['correct']}/{score['total']})")

def gain(a, b):
    return 100 * (results[a]["accuracy"] - results[b]["accuracy"])

deltas, peaks = {}, {}
for mode, matched in (("reread", "block3"), ("reread-up", "up3")):
    deltas[f"{mode}_vs_matched_pp"] = gain(f"{mode}_native", f"{matched}_native")
    deltas[f"{mode}_vs_block_pp"] = gain(f"{mode}_native", "block3_native")
    deltas[f"{mode}_reread_gain_pp"] = gain(f"{mode}_native", f"{mode}_no_reread")
    deltas[f"{mode}_loop_gain_pp"] = gain(f"{mode}_native", f"{mode}_no_loop")
    deltas[f"{mode}_no_loop_vs_block_no_loop_pp"] = gain(f"{mode}_no_loop", "block3_no_loop")
    metadata = json.loads((run / f"{mode}-experiment.json").read_text())
    peaks[mode] = max(epoch["peak_reserved_gib"] for epoch in metadata["training_history"])
for name, value in deltas.items():
    print(f"{name}: {value:+.2f} pp")
for mode, peak in peaks.items():
    print(f"{mode} peak_reserved_gib: {peak:.3f}")
summary = run / "summary.json"
summary.write_text(json.dumps(dict(results=results, deltas=deltas, peak_reserved_gib=peaks,
                                   checkpoint_root=str(checkpoints.resolve())), indent=2) + "\n")
print(f"Saved: {summary}")
PY
