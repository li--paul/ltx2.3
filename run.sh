#!/usr/bin/env bash
# Launch LTX-2.3 text-to-video on the Intel Arc Pro B60 XPUs.
# Usage:
#   ./run.sh                      # default prompt (896x512, 41 frames)
#   ./run.sh "your prompt here"   # custom prompt (via LTX_PROMPT env)
#   LTX_PROMPT="..." ./run.sh
#
# Output: output.mp4 in this directory.
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

exec python -u "$HERE/run_t2v_xpu.py"
