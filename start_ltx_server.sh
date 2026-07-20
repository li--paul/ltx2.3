#!/usr/bin/env bash
# Launch LTX-2.3 text-to-video inference server.
#
# Usage:
#   ./start_ltx_server.sh
#
# Defaults: 0.0.0.0:8001, API token = 111.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- set defaults ---
export LTX_HOST="${LTX_HOST:-0.0.0.0}"
export LTX_PORT="${LTX_PORT:-8001}"
export LTX_API_TOKEN="${LTX_API_TOKEN:-111}"
export LTX_QUEUE_SIZE="${LTX_QUEUE_SIZE:-4}"
export LTX_MULTI_MODE="${LTX_MULTI_MODE:-8}"
export LTX_OUTPUT_DIR="${LTX_OUTPUT_DIR:-outputs/ltx-server}"
export LTX_DB="${LTX_DB:-outputs/ltx-server/jobs.sqlite3}"

# --- env ---
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Starting LTX-2.3 Video Server at ${LTX_HOST}:${LTX_PORT}"
echo "  API token: ${LTX_API_TOKEN}"
echo "  Multi mode: ${LTX_MULTI_MODE}x"
echo "  Output dir: ${LTX_OUTPUT_DIR}"
echo "  Queue size: ${LTX_QUEUE_SIZE}"

exec /home/lm/paul/ltx23-env/bin/python "$HERE/ltx_server.py"
