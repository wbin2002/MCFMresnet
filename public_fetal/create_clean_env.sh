#!/usr/bin/env bash
set -euo pipefail

# Create a fresh environment instead of trying to repair a partially removed
# PyTorch installation. Override these values when needed, for example:
#   ENV_NAME=fetalbench PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
#     bash public_fetal_benchmark/create_clean_env.sh
ENV_NAME="${ENV_NAME:-fetalbench}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH" >&2
  exit 1
fi

echo "[1/5] Creating clean conda environment: ${ENV_NAME}"
conda create --name "${ENV_NAME}" --yes python=3.11 pip

echo "[2/5] Updating packaging tools from ${PYPI_INDEX_URL}"
conda run --name "${ENV_NAME}" python -m pip install \
  --index-url "${PYPI_INDEX_URL}" --upgrade pip setuptools wheel

echo "[3/5] Installing the matched PyTorch pair from ${PYTORCH_INDEX_URL}"
conda run --name "${ENV_NAME}" python -m pip install --no-cache-dir \
  --index-url "${PYTORCH_INDEX_URL}" \
  torch==2.5.1 torchvision==0.20.1

echo "[4/5] Installing benchmark dependencies from ${PYPI_INDEX_URL}"
conda run --name "${ENV_NAME}" python -m pip install --no-cache-dir \
  --index-url "${PYPI_INDEX_URL}" \
  -r public_fetal_benchmark/requirements.txt

echo "[5/5] Verifying imports and the torchvision C++ operators"
conda run --name "${ENV_NAME}" python -c '
import numpy
import scipy
import torch
import torchvision
from torchvision.ops import nms
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("torchvision nms: OK", nms)
'

echo
echo "Environment is ready. Activate it with: conda activate ${ENV_NAME}"
