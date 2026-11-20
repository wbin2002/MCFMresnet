#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-outputs}"

for SEED in 2021 2022 2023 2024 2025; do
  python -m mcfm_repro.bootstrap \
    --reference "$OUTPUT_DIR/pretrained_direct_control/resnet18/seed_${SEED}/test_predictions.csv" \
    --comparison "$OUTPUT_DIR/pretrained_direct_control/mcfm_resnet18/seed_${SEED}/test_predictions.csv" \
    --iterations 10000 \
    --seed "$SEED" \
    --output "$OUTPUT_DIR/pretrained_direct_control/bootstrap_seed_${SEED}.json"
done

for FOLD in 1 2 3 4 5; do
  python -m mcfm_repro.bootstrap \
    --reference "$OUTPUT_DIR/scratch_resnet18_fivefold/resnet18/fold_${FOLD}_seed_2021/test_predictions.csv" \
    --comparison "$OUTPUT_DIR/scratch_mcfm_resnet18_fivefold/mcfm_resnet18/fold_${FOLD}_seed_2021/test_predictions.csv" \
    --iterations 10000 \
    --seed "$((2021 + FOLD))" \
    --output "$OUTPUT_DIR/fivefold_bootstrap/fold_${FOLD}.json"
done

