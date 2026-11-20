#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-./third_party/MedMamba}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"

python -m pip install --index-url "${PYPI_INDEX_URL}" --upgrade packaging ninja einops timm
python -m pip install --index-url "${PYPI_INDEX_URL}" --no-build-isolation causal-conv1d
python -m pip install --index-url "${PYPI_INDEX_URL}" --no-build-isolation mamba-ssm

if [[ ! -f "${REPO_DIR}/MedMamba.py" ]]; then
  mkdir -p "$(dirname "${REPO_DIR}")"
  git clone https://github.com/YubiaoYue/MedMamba.git "${REPO_DIR}"
fi

echo "MedMamba source: ${REPO_DIR}/MedMamba.py"
echo "Record the exact commit with: git -C ${REPO_DIR} rev-parse HEAD"

