#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"

test_input=data/gsm-mc/test.jsonl
train=data/gsm-mc/train.jsonl
rounds="${CORE_ROUNDS:-3}"
case "$rounds" in
  3) suffix=; depths=(0 1 2 3) ;;
  10) suffix=-10; depths=(0 1 2 3 5 10) ;;
  *) echo "CORE_ROUNDS must be 3 or 10" >&2; exit 1 ;;
esac
run="outputs/jev-latent-core${suffix}-comparison/test"
checkpoint="${1:-/var/tmp/loop-jev-$(id -u)/ce-kl-1024-latent-core${rounds}}"
for file in "$train" "$test_input" outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl; do
  [[ -s "$file" ]] || { echo "Missing experiment input: $file" >&2; exit 1; }
done

if [[ ! -f "$checkpoint/experiment.json" ]]; then
  uv run torchrun --standalone --nproc_per_node=2 \
    --module loop_jev.training.answer_slot_distill \
    --model Qwen/Qwen3-1.7B --train "$train" --output "$checkpoint" \
    --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
    --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
    --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
    --loop-count "$rounds" --no-loop-damping --latent-slots 24 \
    --latent-core-width 256 --latent-core-learning-rate 1e-4 \
    --tensor-parallel-size 1 --data-parallel-size 2
fi

uv run python - "$checkpoint" "$train" "$rounds" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from safetensors import safe_open

checkpoint, train = map(Path, sys.argv[1:3])
rounds = int(sys.argv[3])
metadata = json.loads((checkpoint / "experiment.json").read_text())
config = json.loads((checkpoint / "config.json").read_text())
expected = dict(status="complete", model="Qwen/Qwen3-1.7B", examples=1024, seed=42,
                completed_epochs=3, batch_size_per_gpu=4, gradient_accumulation_steps=2,
                loop_count=rounds, latent_slots=24, loop_damping=False, latent_core_width=256,
                latent_core_learning_rate=1e-4, answer_supervision="final-loop",
                logit_loss_weight=1.0, hidden_loss_weight=0.0,
                train_sha256=hashlib.sha256(train.read_bytes()).hexdigest())
index_path = checkpoint / "model.safetensors.index.json"
if index_path.is_file():
    weights = json.loads(index_path.read_text())["weight_map"]
else:
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as handle:
        weights = dict.fromkeys(handle.keys(), "model.safetensors")
if ("latent_core.output.weight" not in weights
        or any(not (checkpoint / shard).is_file() for shard in set(weights.values()))):
    raise ValueError(f"Checkpoint is missing weights: {checkpoint}")
if (any(metadata.get(key) != value for key, value in expected.items())
        or any(config.get(key) != expected[key] for key in ("latent_core_width", "loop_count", "latent_slots"))):
    raise ValueError(f"Unexpected checkpoint configuration: {checkpoint}")
PY

for depth in "${depths[@]}" reset; do
  flags=()
  mode="core${depth}"
  case "$depth" in
    0) flags+=(--disable-looping) ;;
    "$rounds") ;;
    reset) mode="core${rounds}_reset"; flags+=(--reset-latent-core) ;;
    *) flags+=(--loop-readout-round "$depth") ;;
  esac
  uv run python -m loop_jev.evaluation.zero_shot --model "$checkpoint" --input "$test_input" \
    --output "$run/$mode.jsonl" --backend transformers --data-parallel-size 2 --batch-size 4 "${flags[@]}"
done
cp "$checkpoint/experiment.json" "$run/experiment.json"

if [[ "$rounds" != 3 && ! -s outputs/jev-latent-core-comparison/test/core3.jsonl ]]; then
  CORE_ROUNDS=3 bash scripts/run_latent_core_comparison.sh
fi

uv run python - "$run" "$test_input" "$rounds" "${depths[@]}" <<'PY'
import json
import sys
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import score_file

run, test_input = map(Path, sys.argv[1:3])
final = f"core{sys.argv[3]}"
reset = f"{final}_reset"
expected_ids = {json.loads(line)["id"] for line in test_input.read_text().splitlines() if line}
if len(expected_ids) != 1319:
    raise ValueError("Expected 1319 unique test IDs")
files = {"block3_native": Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-loop3.jsonl"),
         "block3_no_loop": Path("outputs/jev-latent24-ce3-final-ce-kl/eval/gsm-mc-test-no-loop.jsonl")}
files.update({mode: run / f"{mode}.jsonl" for mode in [*(f"core{depth}" for depth in sys.argv[4:]), reset]})
if final != "core3":
    files["trained3_native"] = run.parents[1] / "jev-latent-core-comparison/test/core3.jsonl"
results = {}
print("GSM-MC test (1319 examples)")
for name, path in files.items():
    score, ids = score_file(path, disable_looping=name in ("block3_no_loop", "core0"))
    if ids != expected_ids:
        raise ValueError(f"Unexpected evaluation IDs: {path}")
    results[name] = score
    print(f"{name}: {score['accuracy']:.5f} ({score['correct']}/{score['total']})")

def gain(a, b):
    return 100 * (results[a]["accuracy"] - results[b]["accuracy"])

deltas = {f"{final}_vs_block_pp": gain(final, "block3_native"),
          "core0_vs_block_no_loop_pp": gain("core0", "block3_no_loop"),
          f"{final}_vs_core0_pp": gain(final, "core0"), f"{final}_vs_core1_pp": gain(final, "core1"),
          f"{final}_vs_reset_pp": gain(final, reset)}
if "trained3_native" in results:
    deltas[f"{final}_vs_trained3_pp"] = gain(final, "trained3_native")
for name, value in deltas.items():
    print(f"{name}: {value:+.2f} pp")
metadata = json.loads((run / "experiment.json").read_text())
peak = max(epoch["peak_reserved_gib"] for epoch in metadata["training_history"])
print(f"peak_reserved_gib: {peak:.3f}")
summary = run / "summary.json"
summary.write_text(json.dumps(dict(results=results, deltas=deltas, peak_reserved_gib=peak), indent=2) + "\n")
print(f"Saved: {summary}")
PY
