#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Same settings as the all-token head, changing only its application positions.
exec bash scripts/train_transition_head.sh \
  --transition-scope answer-slot \
  --output outputs/jev-loop3-damped-final-ce-kl-transition128-answer-slot "$@"
