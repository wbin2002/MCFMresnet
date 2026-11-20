#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${1:-dataset}"
MANIFEST="${2:-mcfm_repro/manifests/internal_manifest_verified.csv}"
OUTPUT_DIR="${3:-outputs}"

COMMON=(
  --manifest "$MANIFEST"
  --dataset "$DATASET_DIR"
  --output "$OUTPUT_DIR"
  --epochs 20
  --batch-size 64
  --learning-rate 0.001
  --num-workers 8
  --amp
)

# Controlled training-from-scratch comparison and paired five-fold results.
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name scratch_resnet18_fivefold --mode five-fold --model resnet18
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name scratch_mcfm_resnet18_fivefold --mode five-fold --model mcfm_resnet18
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name scratch_resnet18_single --mode single --seed 2021 --model resnet18
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name scratch_mcfm_resnet18_single --mode single --seed 2021 --model mcfm_resnet18
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name scratch_resnet34_single --mode single --seed 2021 --model resnet34
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name scratch_mcfm_resnet34_single --mode single --seed 2021 --model mcfm_resnet34

# Direct five-seed pretrained ResNet18 control requested by the editor/reviewers.
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name pretrained_direct_control --mode five-seed --model resnet18 --pretrained
python -m mcfm_repro.run "${COMMON[@]}" --experiment-name pretrained_direct_control --mode five-seed --model mcfm_resnet18 --pretrained

python -m mcfm_repro.summarize --outputs "$OUTPUT_DIR" --output "$OUTPUT_DIR/all_results.csv"

