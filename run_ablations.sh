#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${1:-dataset}"
MANIFEST="${2:-mcfm_repro/manifests/internal_manifest_verified.csv}"
OUTPUT_DIR="${3:-outputs}"

COMMON=(
  --manifest "$MANIFEST"
  --dataset "$DATASET_DIR"
  --output "$OUTPUT_DIR"
  --mode five-seed
  --model mcfm_resnet18
  --epochs 20
  --batch-size 64
  --learning-rate 0.001
  --num-workers 8
  --amp
)

# Number of feature maps per contrast branch.
for MAPS in 1 2 3 4; do
  python -m mcfm_repro.run "${COMMON[@]}" \
    --experiment-name "feature_maps_${MAPS}" \
    --maps-per-contrast "$MAPS"
done

# Coefficient-wise leave-one-out ablation.
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name contrast_full --contrasts 1.0,1.3,1.6,2.0
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name contrast_without_1_0 --contrasts 1.3,1.6,2.0
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name contrast_without_1_3 --contrasts 1.0,1.6,2.0
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name contrast_without_1_6 --contrasts 1.0,1.3,2.0
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name contrast_without_2_0 --contrasts 1.0,1.3,1.6

# Existing architecture, activation, and pooling ablations. These remain descriptive.
for FUSION in concat se cbam gccp gap; do
  python -m mcfm_repro.run "${COMMON[@]}" \
    --mode single --seed 2021 \
    --experiment-name "fusion_${FUSION}" \
    --fusion "$FUSION"
done
for ACTIVATION in mish relu; do
  python -m mcfm_repro.run "${COMMON[@]}" \
    --mode single --seed 2021 \
    --experiment-name "activation_${ACTIVATION}" \
    --branch-activation "$ACTIVATION"
done

python -m mcfm_repro.summarize --outputs "$OUTPUT_DIR" --output "$OUTPUT_DIR/all_results.csv"

# Characterize how many pixels are clipped by the manuscript-defined transform.
python -m mcfm_repro.contrast_stats \
  --manifest "$MANIFEST" \
  --dataset "$DATASET_DIR" \
  --split-manifest "$OUTPUT_DIR/contrast_full/mcfm_resnet18/seed_2021/split_manifest.csv" \
  --contrasts 1.0,1.3,1.6,2.0 \
  --output "$OUTPUT_DIR/contrast_full/clipping_fractions.csv"

