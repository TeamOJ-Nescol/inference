#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=""
for candidate in python3.12 python3.13 python3.14 python3.11 python3.10; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.10-3.12 is required. Install Python 3.12 and rerun this script."
  exit 1
fi

"$PYTHON_BIN" -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

mkdir -p model

echo
echo "Environment created in .venv using $PYTHON_BIN."
echo "Place the model checkpoint at model/checkpoint_best_ema.pth"
echo "or set MODEL_PATH before running the server."
