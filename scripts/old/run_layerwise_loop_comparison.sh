#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

test_input=data/gsm-mc/test.jsonl
eval_dir=outputs/jev-layerwise-loop-comparison/test
schedule=layerwise
depth_supervision=false
train_extra=()
if [[ "${1:-}" == --gated || "${1:-}" == --gated-depth ]]; then
  schedule=layerwise-gated
  eval_dir=outputs/jev-layerwise-gated-comparison/test
  if [[ "$1" == --gated-depth ]]; then
    depth_supervision=true
    train_extra+=(--layerwise-depth-supervision)
    eval_dir=outputs/jev-layerwise-gated-depth-comparison/test
    gated_eval=outputs/jev-layerwise-gated-comparison/test
    if [[ ! -s "$gated_eval/layerwise3-native.jsonl" || ! -s "$gated_eval/layerwise3-no-loop.jsonl" ]]; then
      bash scripts/run_layerwise_loop_comparison.sh --gated
    fi
  fi
  shift
  ungated_eval=outputs/jev-layerwise-loop-comparison/test
  if [[ ! -s "$ungated_eval/layerwise3-native.jsonl" || ! -s "$ungated_eval/layerwise3-no-loop.jsonl" ]]; then
    bash scripts/run_layerwise_loop_comparison.sh
  fi
fi
checkpoint_default="/dev/shm/loop-jev-$(id -u)/ce-kl-1024-${schedule}3"
if [[ "$depth_supervision" == true ]]; then checkpoint_default+='-depth'; fi
checkpoint="${1:-$checkpoint_default}"
for file in "$test_input" outputs/jev-latent24-ce1-final-ce-kl/eval/gsm-mc-test-loop1.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl; do
  [[ -s "$file" ]] || { echo "Missing test input or baseline: $file" >&2; exit 1; }
done

undamped_root=outputs/jev-block3-no-damping-comparison
undamped="${UNDAMPED_BLOCK_CHECKPOINT:-$undamped_root/checkpoint}"
if [[ ! -s "$undamped_root/test/native.jsonl" || ! -s "$undamped_root/test/no-loop.jsonl" ]]; then
  if [[ ! -s "$undamped/config.json" ]]; then
    uv run torchrun --standalone --nproc_per_node=2 \
      --module loop_jev.training.answer_slot_distill \
      --model Qwen/Qwen3-1.7B --train data/gsm-mc/train.jsonl --output "$undamped" \
      --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
      --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
      --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
      --loop-start-layer 9 --loop-end-layer 18 --loop-count 3 --no-loop-damping \
      --latent-slots 24 --tensor-parallel-size 1 --data-parallel-size 2
  fi
  for mode in native no-loop; do
    extra=()
    if [[ "$mode" == no-loop ]]; then extra+=(--disable-looping); fi
    uv run python -m loop_jev.evaluation.zero_shot --model "$undamped" --input "$test_input" \
      --output "$undamped_root/test/$mode.jsonl" --backend transformers \
      --data-parallel-size 2 --batch-size 4 "${extra[@]}"
  done
fi

if [[ ! -f "$checkpoint/experiment.json" ]]; then
  uv run torchrun --standalone --nproc_per_node=2 \
    --module loop_jev.training.answer_slot_distill \
    --model Qwen/Qwen3-1.7B --train data/gsm-mc/train.jsonl --output "$checkpoint" \
    --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
    --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
    --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
    --loop-start-layer 9 --loop-end-layer 18 --loop-count 3 --loop-schedule "$schedule" --no-loop-damping \
    --latent-slots 24 --tensor-parallel-size 1 --data-parallel-size 2 "${train_extra[@]}"
fi

uv run python - "$checkpoint" "$schedule" "$depth_supervision" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
metadata = json.loads((checkpoint / "experiment.json").read_text())
config = json.loads((checkpoint / "config.json").read_text())
if metadata.get("layerwise_depth_supervision", False) != (sys.argv[3] == "true"):
    raise ValueError(f"Wrong depth supervision mode for {checkpoint}")
expected = dict(loop_schedule=sys.argv[2], loop_start_layer=9, loop_end_layer=18,
                loop_count=3, loop_damping=False, latent_slots=24)
for path, data, required in (
    ("config.json", config, expected),
    ("experiment.json", metadata, {**expected, "status": "complete", "examples": 1024,
                                  "answer_supervision": "final-loop", "logit_loss_weight": 1.0}),
):
    if differing := [key for key, val in required.items() if data.get(key) != val]:
        raise ValueError(f"Mismatched checkpoint {checkpoint / path}: {', '.join(differing)}")
PY

modes=(native no-loop)
if [[ "$depth_supervision" == true ]]; then modes+=(depth-2); fi
for mode in "${modes[@]}"; do
  extra=()
  if [[ "$mode" == no-loop ]]; then extra+=(--disable-looping); fi
  if [[ "$mode" == depth-2 ]]; then extra+=(--loop-readout-round 2); fi
  uv run python -m loop_jev.evaluation.zero_shot \
    --model "$checkpoint" --input "$test_input" --output "$eval_dir/layerwise3-$mode.jsonl" \
    --backend transformers --data-parallel-size 2 --batch-size 4 "${extra[@]}"
done

uv run python - "$eval_dir" "$test_input" "$checkpoint" "$schedule" "$depth_supervision" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file
from loop_jev.training.answer_slot_distill import read_rows

output, test_input, checkpoint = map(Path, sys.argv[1:4])
gated = sys.argv[4] == "layerwise-gated"
depth_supervision = sys.argv[5] == "true"
expected_ids = {row["id"] for row in read_rows(test_input)}
if len(expected_ids) != 1319:
    raise ValueError(f"Expected the original 1319-example test split: {test_input}")
name = "layerwise3_gated" if gated else "layerwise3"
if depth_supervision:
    name = "layerwise3_gated_depth"
files = {
    "ce_kl_1": (Path("outputs/jev-latent24-ce1-final-ce-kl/eval/gsm-mc-test-loop1.jsonl"), False),
    "block3_damped": (Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl"), False),
    "block3": (Path("outputs/jev-block3-no-damping-comparison/test/native.jsonl"), False),
    "block3_no_loop": (Path("outputs/jev-block3-no-damping-comparison/test/no-loop.jsonl"), True),
    name: (output / "layerwise3-native.jsonl", False),
    name + "_no_loop": (output / "layerwise3-no-loop.jsonl", True),
}
if gated:
    ungated = Path("outputs/jev-layerwise-loop-comparison/test")
    files.update(layerwise3=(ungated / "layerwise3-native.jsonl", False),
                 layerwise3_no_loop=(ungated / "layerwise3-no-loop.jsonl", True))
if depth_supervision:
    prior_gated = Path("outputs/jev-layerwise-gated-comparison/test")
    files.update(layerwise3_gated=(prior_gated / "layerwise3-native.jsonl", False),
                 layerwise3_gated_no_loop=(prior_gated / "layerwise3-no-loop.jsonl", True),
                 layerwise3_gated_depth_2=(output / "layerwise3-depth-2.jsonl", False))
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
    "layerwise_vs_matched_block_pp": delta(name, "block3"),
    "layerwise_vs_damped_block_pp": delta(name, "block3_damped"),
    "layerwise_vs_single_pass_pp": delta(name, "ce_kl_1"),
    "layerwise_same_checkpoint_loop_gain_pp": delta(name, name + "_no_loop"),
    "block_same_checkpoint_loop_gain_pp": delta("block3", "block3_no_loop"),
}
if depth_supervision:
    deltas.update(
        depth_supervision_gain_pp=delta(name, "layerwise3_gated"),
        depth_supervision_no_loop_gain_pp=delta(name + "_no_loop", "layerwise3_gated_no_loop"),
        depth_2_vs_1_pp=delta("layerwise3_gated_depth_2", name + "_no_loop"),
        depth_3_vs_2_pp=delta(name, "layerwise3_gated_depth_2"),
    )
elif gated:
    deltas["gate_gain_pp"] = delta(name, "layerwise3")
    deltas["gate_gain_no_loop_pp"] = delta(name + "_no_loop", "layerwise3_no_loop")
for label, val in deltas.items():
    print(f"{label}: {val:+.2f} pp")
metadata = json.loads((checkpoint / "experiment.json").read_text())
memory = max(epoch["peak_reserved_gib"] for epoch in metadata["training_history"])
summary = output / "summary.json"
summary.write_text(json.dumps({"results": results, "deltas": deltas, "peak_reserved_gib": memory}, indent=2) + "\n")
print(f"Saved: {summary}")
PY
