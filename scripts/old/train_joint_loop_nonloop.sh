#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
joint_output="${JOINT_OUTPUT:-outputs/jev-loop3-damped-final-ce-kl-joint-nonloop0.5}"

uv run torchrun --standalone --nproc_per_node=2 \
  --module loop_jev.training.answer_slot_distill \
  --model Qwen/Qwen3-1.7B \
  --train data/gsm-mc/train.jsonl \
  --output "$joint_output" \
  --epochs 3 --learning-rate 2e-6 \
  --batch-size 4 --gradient-accumulation-steps 2 \
  --max-length 16384 --max-train-examples 1024 \
  --temperature 2.0 --answer-supervision final-loop \
  --logit-loss-weight 1.0 --hidden-loss-weight 0.0 \
  --loop-start-layer 9 --loop-end-layer 18 \
  --loop-count 3 --loop-damping \
  --nonloop-loss-weight 0.5 \
  --tensor-parallel-size 1 --data-parallel-size 2

uv run python -m loop_jev.evaluation.zero_shot \
  --model "$joint_output" --input data/gsm-mc/test.jsonl \
  --output "$joint_output/eval/gsm-mc-test-loop3.jsonl" \
  --backend transformers --data-parallel-size 2 --batch-size 4

uv run python -m loop_jev.evaluation.zero_shot \
  --model "$joint_output" --input data/gsm-mc/test.jsonl \
  --output "$joint_output/eval/gsm-mc-test-no-loop.jsonl" \
  --backend transformers --disable-looping \
  --data-parallel-size 2 --batch-size 4
