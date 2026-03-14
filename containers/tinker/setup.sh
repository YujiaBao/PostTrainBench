#!/bin/bash
#
# Set up the Tinker mode Python environment for PostTrainBench.
#
# Usage:
#   bash containers/tinker/setup.sh
#
# This creates a venv at containers/tinker/.venv and installs all
# dependencies. Run from the repo root.
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/)
#   - TINKER_COOKBOOK_PATH set to the tinker-cookbook repo location
#     (default: ~/Repos/tinker-cookbook)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

TINKER_COOKBOOK_PATH="${TINKER_COOKBOOK_PATH:-$HOME/Repos/tinker-cookbook}"

echo "=== PostTrainBench Tinker Environment Setup ==="
echo "Venv:            ${VENV_DIR}"
echo "Tinker cookbook:  ${TINKER_COOKBOOK_PATH}"
echo "================================================"

# Check tinker-cookbook exists
if [ ! -d "$TINKER_COOKBOOK_PATH" ]; then
    echo "ERROR: tinker-cookbook not found at $TINKER_COOKBOOK_PATH"
    echo "Clone it or set TINKER_COOKBOOK_PATH."
    exit 1
fi

# Create venv
echo "[1/4] Creating virtual environment..."
uv venv "${VENV_DIR}"

# Install pinned dependencies
echo "[2/4] Installing dependencies from requirements-tinker.txt..."
uv pip install --python "${VENV_DIR}/bin/python" -r "${SCRIPT_DIR}/requirements-tinker.txt"

# Install inspect_evals (same commit as vllm_debug container)
echo "[3/4] Installing inspect_evals..."
uv pip install --python "${VENV_DIR}/bin/python" \
    "inspect_evals @ git+https://github.com/UKGovernmentBEIS/inspect_evals.git@06001a83e6d7c709c2ede0570dce7f1031a0bad8"

# Install tinker-cookbook in editable mode
echo "[4/4] Installing tinker-cookbook (editable)..."
uv pip install --python "${VENV_DIR}/bin/python" -e "${TINKER_COOKBOOK_PATH}"

echo ""
echo "Done! Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
