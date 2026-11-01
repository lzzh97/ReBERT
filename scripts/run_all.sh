#!/usr/bin/env bash
# Full ReBERT pipeline with the paper's settings. Run from the repository root.
# Usage: bash scripts/run_all.sh [variant]   (variant: full | no_tree | tree_only)
set -euo pipefail

VARIANT="${1:-full}"
LEVEL=6
TEST_BENCHES="b03 b04 b05 b07 b08 b11 b12 b13 b14 b15"

python scripts/prepare_data.py --level "$LEVEL"
python scripts/train.py --variant "$VARIANT" --level "$LEVEL"
python scripts/predict.py --variant "$VARIANT" --level "$LEVEL" \
    --ckpt "work/ckpts/${VARIANT}_lv${LEVEL}/final" --benches $TEST_BENCHES
python scripts/evaluate.py --tag "$VARIANT" --level "$LEVEL" --benches $TEST_BENCHES
