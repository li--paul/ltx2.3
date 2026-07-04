#!/usr/bin/env bash
# Recreate the LTX-2.3 XPU Python environment with uv.
#
# Produces a venv at $LTX_ENV (default /home/lm/ltx23-env) with the exact
# pinned package set in requirements.txt, including the +xpu torch build.
#
# Usage:
#   ./setup-env.sh                       # default path /home/lm/ltx23-env
#   LTX_ENV=/path/to/venv ./setup-env.sh # custom path
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/) on PATH
#   - the Intel GPU compute stack (Level-Zero / compute runtime) installed
#     system-wide so torch+xpu can see the XPUs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LTX_ENV="${LTX_ENV:-/home/lm/ltx23-env}"
XPU_INDEX="https://download.pytorch.org/whl/xpu"

echo ">> Creating venv at $LTX_ENV (python 3.12)"
uv venv --python 3.12 "$LTX_ENV"
# shellcheck disable=SC1091
source "$LTX_ENV/bin/activate"

echo ">> Installing torch/torchvision/torchaudio (+xpu) from $XPU_INDEX"
uv pip install \
    "torch==2.12.1+xpu" "torchvision==0.27.1+xpu" "torchaudio==2.11.0+xpu" \
    --index-url "$XPU_INDEX"

echo ">> Installing pinned dependencies from requirements.txt"
# +xpu local versions resolve via the extra index; everything else from PyPI.
uv pip install -r "$HERE/requirements.txt" \
    --extra-index-url "$XPU_INDEX"

echo ">> Installing LTX-2 packages (editable, no deps to avoid CUDA torch pins)"
if [[ -d /home/lm/LTX-2 ]]; then
    uv pip install --no-deps -e /home/lm/LTX-2/packages/ltx-core
    uv pip install --no-deps -e /home/lm/LTX-2/packages/ltx-pipelines
else
    echo "   (skipped: /home/lm/LTX-2 not present — clone Lightricks/LTX-2 first)"
fi

echo ">> Verifying torch sees the XPUs"
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
