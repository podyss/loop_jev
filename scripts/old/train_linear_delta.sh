#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/train_transition_head.sh \
  --transition-type linear-delta --transition-scope answer-slot \
  --delta-loss-weight 0.1 \
  --output outputs/jev-loop3-damped-final-ce-kl-linear-delta0.1 "$@"
