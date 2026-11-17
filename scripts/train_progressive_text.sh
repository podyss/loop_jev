#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
progressive_output="${PROGRESSIVE_OUTPUT:-outputs/jev-progressive-text-ce-kl}"
# Keep output under one variable so evaluation always uses the trained checkpoint.
uv run torchrun --standalone --nproc_per_node=2 \
  --module loop_jev.training.progressive_text \
  --model Qwen/Qwen3-1.7B \
  --train data/gsm-mc/train.jsonl \
  --output "$progressive_output" \
  --epochs 3 --learning-rate 2e-6 \
  --batch-size 4 --gradient-accumulation-steps 2 \
  --max-length 16384 --max-train-examples 1024 \
  --temperature 2.0 --answer-supervision final-loop \
  --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
  --loop-start-layer 9 --loop-end-layer 18 --loop-count 1 --no-loop-damping \
  --progressive-text-weight 1.0 --text-max-targets 128 \
  --tensor-parallel-size 1 --data-parallel-size 2

uv run python -m loop_jev.evaluation.zero_shot \
  --model "$progressive_output" \
  --input data/gsm-mc/test.jsonl \
  --output "$progressive_output/eval/gsm-mc-test-no-loop.jsonl" \
  --backend transformers --disable-looping \
  --data-parallel-size 2 --batch-size 4 --dtype bfloat16
