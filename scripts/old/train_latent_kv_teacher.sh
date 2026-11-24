#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"

case "${1:-kv3}" in
  kv3) loop_count=3; kv_weight=0.1 ;;
  kv1) loop_count=1; kv_weight=0.1 ;;
  ce3) loop_count=3; kv_weight=0.0 ;;
  ce1) loop_count=1; kv_weight=0.0 ;;
  *) echo "Usage: $0 {kv3|kv1|ce3|ce1}" >&2; exit 2 ;;
esac

experiment="${1:-kv3}"
output="outputs/jev-latent24-${experiment}-final-ce-kl"
cache="outputs/latent-kv-cache/gsm-mc-train1024-layer17-slots24.pt"

if [[ "$kv_weight" != 0.0 && ! -f "$cache" ]]; then
  uv run python -m loop_jev.training.latent_kv \
    --model Qwen/Qwen3-1.7B \
    --train data/gsm-mc/train.jsonl --output "$cache" \
    --max-train-examples 1024 --max-length 16384 \
    --latent-slots 24 --kv-layer 17 --seed 42
fi

kv_args=()
if [[ "$kv_weight" != 0.0 ]]; then
  kv_args=(--kv-loss-weight "$kv_weight" --kv-cache "$cache" --kv-layer 17)
fi

if [[ ! -f "$output/experiment.json" ]]; then
  uv run torchrun --standalone --nproc_per_node=2 \
    --module loop_jev.training.answer_slot_distill \
    --model Qwen/Qwen3-1.7B \
    --train data/gsm-mc/train.jsonl --output "$output" \
    --epochs 3 --learning-rate 2e-6 \
    --batch-size 4 --gradient-accumulation-steps 2 \
    --max-length 16384 --max-train-examples 1024 \
    --temperature 2.0 --answer-supervision final-loop \
    --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
    --loop-start-layer 9 --loop-end-layer 18 \
    --loop-count "$loop_count" --loop-damping \
    --latent-slots 24 "${kv_args[@]}" \
    --tensor-parallel-size 1 --data-parallel-size 2
else
  echo "Using completed checkpoint: $output"
fi

uv run python -m loop_jev.evaluation.zero_shot \
  --model "$output" --input data/gsm-mc/test.jsonl \
  --output "$output/eval/gsm-mc-test-loop${loop_count}.jsonl" \
  --backend transformers --data-parallel-size 2 --batch-size 4

if [[ "$loop_count" -gt 1 ]]; then
  uv run python -m loop_jev.evaluation.zero_shot \
    --model "$output" --input data/gsm-mc/test.jsonl \
    --output "$output/eval/gsm-mc-test-no-loop.jsonl" \
    --backend transformers --disable-looping \
    --data-parallel-size 2 --batch-size 4
fi
