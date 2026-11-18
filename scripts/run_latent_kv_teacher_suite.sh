#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"

for mode in kv3 ce3 kv1 ce1; do
  echo "Running latent teacher experiment: $mode"
  bash scripts/train_latent_kv_teacher.sh "$mode"
done

uv run python -m loop_jev.evaluation.latent_kv_summary \
  --output outputs/jev-latent24-comparison/summary.json
