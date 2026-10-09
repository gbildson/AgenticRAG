#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
if [ ! -x .venv/bin/python ]; then
    echo "Run ./setup.sh first"
    exit 1
fi
if [ -z "${CHAT_MODEL:-}" ]; then
    echo "Set CHAT_MODEL to a tool-capable model in .env or the environment"
    exit 1
fi

.venv/bin/python -m app.ingest
exec .venv/bin/python -m uvicorn app.api:app --host 127.0.0.1 --port "${PORT:-8000}"
