#!/usr/bin/env bash
# Launch LTX-2.3 text-to-video Run B: 1024x1024, 121 frames @ 24 fps (~5.0s clip).
# Uses the distilled two-stage pipeline with per-stage timing.
#
# Usage:
#   ./run_b.sh                      # default red-panda prompt
#   ./run_b.sh "your prompt here"   # custom prompt (via arg or LTX_PROMPT env)
#   LTX_PROMPT="..." ./run_b.sh
#
# Output: output_1024.mp4 in this directory.
# Expected wall time: ~2.5 min (see README "Performance / Benchmarks").
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- activate the uv venv ---
source /home/lm/ltx23-env/bin/activate

# --- env ---
export HF_HUB_OFFLINE=1                 # all weights are local; no network calls
export TOKENIZERS_PARALLELISM=false

# allow overriding the prompt from the first arg or an env var
if [[ $# -ge 1 ]]; then
    export LTX_PROMPT="$1"
fi

exec python -u "$HERE/run_t2v_xpu_perf.py"
