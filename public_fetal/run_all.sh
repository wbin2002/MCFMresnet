#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/mnt/dataset/FETAL_PLANES_ZENODO}"
OUTPUT_ROOT="${2:-./public_benchmark_outputs}"
MEDMAMBA_REPO="${MEDMAMBA_REPO:-./third_party/MedMamba}"
SEEDS="${SEEDS:-2021}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_MEDMAMBA="${RUN_MEDMAMBA:-1}"

mkdir -p "${OUTPUT_ROOT}/logs"

python -m public_fetal_benchmark.prepare_manifest \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/manifests" \
  --validation-seed 2021 \
  --validation-fraction 0.10

IFS=',' read -ra SEED_ARRAY <<< "${SEEDS}"
MODELS=(mcfm_resnet18 resnet18 convnextv2_base)
if [[ "${RUN_MEDMAMBA}" == "1" ]]; then
  MODELS+=(medmamba_t)
fi

for seed in "${SEED_ARRAY[@]}"; do
  for model in "${MODELS[@]}"; do
    METRICS_PATH="${OUTPUT_ROOT}/${model}/seed_${seed}/metrics.json"
    if [[ -f "${METRICS_PATH}" ]]; then
      echo "[skip] completed run: ${model} seed ${seed}"
      continue
    fi
    EXTRA_ARGS=()
    if [[ "${model}" == "medmamba_t" ]]; then
      EXTRA_ARGS+=(--medmamba-repo "${MEDMAMBA_REPO}" --no-amp)
    fi
    python -m public_fetal_benchmark.run \
      --dataset-root "${DATASET_ROOT}" \
      --split-manifest "${OUTPUT_ROOT}/manifests/split_manifest.csv" \
      --output-root "${OUTPUT_ROOT}" \
      --model "${model}" \
      --seed "${seed}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --learning-rate 0.001 \
      --optimizer adam \
      --scheduler none \
      --augmentation none \
      --no-pretrained \
      "${EXTRA_ARGS[@]}" \
      2>&1 | tee "${OUTPUT_ROOT}/logs/${model}_seed_${seed}.log"
  done
done

python -m public_fetal_benchmark.summarize --output-root "${OUTPUT_ROOT}"

