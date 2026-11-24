#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
checkpoint="${1:-outputs/jev-loop3-damped-final-ce-kl-transition128}"
for mode in loop3 no-loop; do
  extra=()
  if [[ "$mode" == no-loop ]]; then
    extra+=(--disable-looping)
  fi
  uv run python -m loop_jev.evaluation.zero_shot \
    --model "$checkpoint" --input data/gsm-mc/test.jsonl \
    --output "$checkpoint/eval/gsm-mc-test-$mode.jsonl" \
    --backend transformers --data-parallel-size 2 --batch-size 4 "${extra[@]}"
done
