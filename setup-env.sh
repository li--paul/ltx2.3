#!/usr/bin/env bash
# Recreate the LTX-2.3 XPU Python environment with uv.
#
# Primary method: `uv sync` from pyproject.toml + uv.lock (fully reproducible).
# Fallback:      `uv pip install -r requirements.txt` (same pins, less strict).
#
# Produces a venv at $LTX_ENV (default /home/lm/ltx23-env).
#
# Usage:
#   ./setup-env.sh                       # default path
#   LTX_ENV=/path/to/venv ./setup-env.sh # custom path
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/) on PATH
#   - the Intel GPU compute stack (Level-Zero / compute runtime) installed
#     system-wide so torch+xpu can see the XPUs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
LTX_ENV="${LTX_ENV:-/home/lm/ltx23-env}"
LTX_REPO="${LTX_REPO:-/home/lm/LTX-2}"

echo ">> Syncing environment from uv.lock -> $LTX_ENV"
UV_PROJECT_ENVIRONMENT="$LTX_ENV" uv sync

# uv sync installs everything in the lockfile but NOT the LTX-2 packages
# (they're an external checkout with conflicting CUDA torch pins, so they
# are intentionally excluded from pyproject.toml). Install them manually:
echo ">> Installing LTX-2 packages editable (no-deps)"
if [[ -d "$LTX_REPO" ]]; then
    UV_PROJECT_ENVIRONMENT="$LTX_ENV" uv pip install --no-deps -e "$LTX_REPO/packages/ltx-core"
    UV_PROJECT_ENVIRONMENT="$LTX_ENV" uv pip install --no-deps -e "$LTX_REPO/packages/ltx-pipelines"
else
    echo "   (skipped: $LTX_REPO not present — clone Lightricks/LTX-2 first)"
    echo "   git clone --depth 1 https://github.com/Lightricks/LTX-2.git $LTX_REPO"
fi

echo ">> Verifying torch sees the XPUs"
# shellcheck disable=SC1091
source "$LTX_ENV/bin/activate"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("xpu available:", torch.xpu.is_available())
print("xpu device count:", torch.xpu.device_count())
if torch.xpu.is_available():
    p = torch.xpu.get_device_properties(0)
    print(f"xpu:0 {p.name} | {p.total_memory/1024**3:.1f} GB")
PY

echo ">> Done. Activate with:  source $LTX_ENV/bin/activate"
