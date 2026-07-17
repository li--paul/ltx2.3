#!/usr/bin/env bash
# Launch LTX-2.3 text-to-video on the Intel Arc Pro B60 XPUs.
# Usage:
#   ./run.sh                      # default (1024x1024, 73 frames)
#   ./run.sh "your prompt here"   # custom prompt
#   LTX_WIDTH=896 LTX_HEIGHT=512 LTX_FRAMES=41 ./run.sh
#
# Output: output.mp4 in this directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- python ---
PYTHON="/home/lm/paul/ltx23-env/bin/python"

# --- env ---
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LTX_GEMMA_DEVICE=cpu
export LTX_WIDTH="${LTX_WIDTH:-1024}"
export LTX_HEIGHT="${LTX_HEIGHT:-1024}"
export LTX_FRAMES="${LTX_FRAMES:-121}"

# allow overriding the prompt from the first arg or an env var
if [[ $# -ge 1 ]]; then
    export LTX_PROMPT="$1"
fi

exec "$PYTHON" -u "$HERE/run_t2v_xpu_perf.py"
