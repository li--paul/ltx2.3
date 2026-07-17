#!/usr/bin/env bash
# Launch LTX-2.3 multi-video generation (8 prompts, all 16 XPUs).
# Pre-encodes prompts with one Gemma on CPU, then spawns 8 parallel workers
# staggered by 5 s to avoid XPU driver contention.
#
# Usage:
#   ./run_multi.sh                        # 8 default prompts
#   ./run_multi.sh prompts.json           # custom prompts JSON file
#   ./run_multi.sh --prompts-file my.json --job-dir /tmp/out
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="/home/lm/paul/ltx23-env/bin/python"

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LTX_GEMMA_DEVICE=cpu
export LTX_SPAWN_DELAY="${LTX_SPAWN_DELAY:-5}"

exec "$PYTHON" -u "$HERE/run_multi_xpu.py" "$@"