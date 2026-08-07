#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This setup targets native Apple Silicon (arm64)." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python - <<'PY'
import torch

if not torch.backends.mps.is_available():
    raise SystemExit(
        "PyTorch cannot access MPS. Confirm this is a native arm64 Python environment."
    )
print(f"PyTorch {torch.__version__}; MPS is available.")
PY

if [[ "${1:-}" != "--skip-download" ]]; then
  python scripts/download-model.py
else
  echo "Skipping the approximately 34 GB model download."
fi

echo "Setup complete. Start the endpoint with ./scripts/serve.sh"
