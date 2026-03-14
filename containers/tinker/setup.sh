#!/bin/bash
#
# Set up the Tinker mode Python environment for PostTrainBench.
#
# Usage:
#   bash containers/tinker/setup.sh
#
# This clones the required repos, creates a venv at containers/tinker/.venv,
# and installs all dependencies. Run from the repo root.
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/)
#   - git

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
VENDOR_DIR="${SCRIPT_DIR}/vendor"

TINKER_REPO="https://github.com/thinking-machines-lab/tinker.git"
TINKER_COOKBOOK_REPO="https://github.com/thinking-machines-lab/tinker-cookbook.git"

echo "=== PostTrainBench Tinker Environment Setup ==="
echo "Venv:    ${VENV_DIR}"
echo "Vendor:  ${VENDOR_DIR}"
echo "================================================"

# ---------------------------------------------------------------------------
# Step 1: Clone tinker SDK and tinker-cookbook into vendor/
# ---------------------------------------------------------------------------
mkdir -p "${VENDOR_DIR}"

if [ -d "${VENDOR_DIR}/tinker" ]; then
    echo "[1/5] tinker SDK already cloned, pulling latest..."
    git -C "${VENDOR_DIR}/tinker" pull --ff-only || true
else
    echo "[1/5] Cloning tinker SDK..."
    git clone "${TINKER_REPO}" "${VENDOR_DIR}/tinker"
fi

if [ -d "${VENDOR_DIR}/tinker-cookbook" ]; then
    echo "[2/5] tinker-cookbook already cloned, pulling latest..."
    git -C "${VENDOR_DIR}/tinker-cookbook" pull --ff-only || true
else
    echo "[2/5] Cloning tinker-cookbook..."
    git clone "${TINKER_COOKBOOK_REPO}" "${VENDOR_DIR}/tinker-cookbook"
fi

# ---------------------------------------------------------------------------
# Step 3: Create venv and install dependencies
# ---------------------------------------------------------------------------
echo "[3/5] Creating virtual environment..."
uv venv "${VENV_DIR}"

echo "[4/5] Installing dependencies..."
uv pip install --python "${VENV_DIR}/bin/python" \
    -r "${SCRIPT_DIR}/requirements-tinker.txt"

# Install inspect_evals (same commit as vllm_debug container for consistency)
uv pip install --python "${VENV_DIR}/bin/python" \
    "inspect_evals @ git+https://github.com/UKGovernmentBEIS/inspect_evals.git@06001a83e6d7c709c2ede0570dce7f1031a0bad8"

# Install tinker SDK and cookbook from vendor/
echo "[5/5] Installing tinker SDK and tinker-cookbook (editable)..."
uv pip install --python "${VENV_DIR}/bin/python" -e "${VENDOR_DIR}/tinker"
uv pip install --python "${VENV_DIR}/bin/python" -e "${VENDOR_DIR}/tinker-cookbook"

echo ""
echo "=== Setup complete ==="
echo "Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "tinker-cookbook is at: ${VENDOR_DIR}/tinker-cookbook"
