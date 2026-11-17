#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
exec uv run torchrun --standalone --nproc_per_node=2 \
  --module loop_jev.training.answer_slot_distill \
  --model Qwen/Qwen3-1.7B \
  --train data/gsm-mc/train.jsonl \
  --output outputs/jev-single-pass-depth-suffix0.1 \
  --epochs 3 --learning-rate 2e-6 \
  --batch-size 4 --gradient-accumulation-steps 2 \
  --max-length 16384 --max-train-examples 1024 \
  --temperature 2.0 --answer-supervision final-loop \
  --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
  --loop-start-layer 9 --loop-end-layer 18 --loop-count 1 --loop-damping \
  --suffix-loss-weight 0.1 --suffix-max-tokens 128 \
  --tensor-parallel-size 1 --data-parallel-size 2 "$@"
