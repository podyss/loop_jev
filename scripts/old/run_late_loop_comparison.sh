#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

test_input=data/gsm-mc/test.jsonl
eval_dir=outputs/jev-late-loop-comparison/test
output="${1:-/dev/shm/loop-jev-$(id -u)/ce-kl-1024-loop3-late}"
[[ -s "$test_input" ]] || { echo "Missing test set: $test_input" >&2; exit 1; }

if [[ ! -f "$output/experiment.json" ]]; then
  uv run torchrun --standalone --nproc_per_node=2 \
    --module loop_jev.training.answer_slot_distill \
    --model Qwen/Qwen3-1.7B --train data/gsm-mc/train.jsonl --output "$output" \
    --epochs 3 --learning-rate 2e-6 --batch-size 4 --gradient-accumulation-steps 2 \
    --max-length 16384 --max-train-examples 1024 --seed 42 --temperature 2.0 \
    --answer-supervision final-loop --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
    --loop-start-layer 14 --loop-end-layer 23 --loop-count 3 --loop-damping \
    --latent-slots 24 --tensor-parallel-size 1 --data-parallel-size 2
fi

for mode in round-3 no-loop; do
  if [[ "$mode" == round-3 ]]; then
    flags=(--loop-readout-round 3)
  else
    flags=(--disable-looping)
  fi
  uv run python -m loop_jev.evaluation.zero_shot \
    --model "$output" --input "$test_input" --output "$eval_dir/$mode.jsonl" \
    --backend transformers --data-parallel-size 2 --batch-size 4 "${flags[@]}"
done
