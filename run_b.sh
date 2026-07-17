#!/usr/bin/env bash
# Launch LTX-2.3 text-to-video Run B: 1024x1024, 121 frames ~5.0s clip.
# Usage:
#   ./run_b.sh                      # default (1024x1024, 121 frames)
#   ./run_b.sh "your prompt here"
#
# Output: output_1024.mp4 in this directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="/home/lm/paul/ltx23-env/bin/python"

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LTX_GEMMA_DEVICE=cpu
export LTX_WIDTH="${LTX_WIDTH:-1024}"
export LTX_HEIGHT="${LTX_HEIGHT:-1024}"
export LTX_FRAMES="${LTX_FRAMES:-121}"

if [[ $# -ge 1 ]]; then
    export LTX_PROMPT="$1"
fi

exec "$PYTHON" -u "$HERE/run_t2v_xpu_perf.py"
