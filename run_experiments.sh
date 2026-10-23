#!/usr/bin/env bash
set -euo pipefail

STEPS="${STEPS:-1000000}"
SEEDS="${SEEDS:-0 1}"
VARIANTS="baseline time_penalty hover_hack hover_fixed"

for seed in $SEEDS; do
  for variant in $VARIANTS; do
    python train.py "$variant" --steps "$STEPS" --seed "$seed"
    python evaluate.py "$variant" --seed "$seed"
    python record.py "$variant" --seed "$seed"
  done
done

python summarize.py
