#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo "Virtual environment not found. Run ./install.sh first."
  exit 1
fi

source .venv/bin/activate

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

DEFAULT_MODEL_PATH="$ROOT_DIR/model/checkpoint_best_ema.pth"
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model checkpoint not found at $MODEL_PATH"
  echo "Set MODEL_PATH or place checkpoint_best_ema.pth under model/."
  exit 1
fi

python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
