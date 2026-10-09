#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first: https://docs.astral.sh/uv/"
    exit 1
fi

uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/prepare_example.py
echo "Setup complete. Configure CHAT_MODEL in .env, then run ./run.sh"
