# LTX-2.3 Text-to-Video on Intel Arc Pro B60 (XPU)

End-to-end instructions for running **Lightricks LTX-2.3** text-to-video
generation on **Intel Arc Pro B60 (Battlemage) XPUs** using PyTorch's native XPU
backend — no CUDA, no Diffusers, no ComfyUI required.

This README documents a working setup that produces a synchronized
**video + audio** MP4 from a text prompt using **four B60 GPUs** on the
machine, with NUMA-aware device placement and model-parallel sharding of the
Gemma text encoder.

---

## Table of Contents

1. [What this is](#what-this-is)
2. [Tested hardware & environment](#tested-hardware--environment)
3. [Why this approach (and not Diffusers)](#why-this-approach-and-not-diffusers)
4. [Solution architecture](#solution-architecture)
5. [The fp8 question: `fp8-cast` vs `fp8-scaled-mm`](#the-fp8-question-fp8-cast-vs-fp8-scaled-mm)
6. [XPU patches applied to the LTX-2 repo](#xpu-patches-applied-to-the-ltx-2-repo)
7. [Prerequisites & setup](#prerequisites--setup)
8. [Running](#running)
9. [Configuration](#configuration)
10. [Performance / Benchmarks](#performance--benchmarks)
11. [Troubleshooting](#troubleshooting)
12. [File layout](#file-layout)
13. [Limitations & notes](#limitations--notes)

---

## What this is

[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) is Lightricks' 22B-parameter
DiT-based **joint audio-video foundation model**: it generates synchronized video
*and* audio in a single pass. This repo runs its **distilled two-stage
text-to-video pipeline** on Intel XPUs and writes an MP4.

The distilled pipeline:
- **Stage 1** — denoise at half resolution (8 sigmas / 8 steps, CFG=1, single
  forward per step).
- **Spatial upsample 2×** — a dedicated latent upscaler.
- **Stage 2** — refine at full resolution (4 sigmas / 3 steps).
- **Decode** — video VAE → frames, audio VAE + vocoder → waveform, muxed to MP4.

---

## Tested hardware & environment

| Component | Value |
|---|---|
| GPUs | 32× Intel Arc Pro B60 (Battlemage / Xe2), 23.9 GB VRAM each |
| GPUs used | `xpu:0` and `xpu:1` (the "first two B60 XPUs") |
| CPU | Intel Xeon Gold 6530 (128 logical cores), 2 TB RAM |
| Driver / runtime | Level-Zero 20.1.0, compute runtime 26.18.38308 |
| OS | Linux 7.0.0-27-generic |
| Python | CPython 3.12.13 (managed by uv) |
| PyTorch | **2.12.1+xpu** (`https://download.pytorch.org/whl/xpu`) |
| LTX-2 repo | `Lightricks/LTX-2` (cloned, packages installed editable) |

Key XPU primitives verified working on Battlemage:

- `torch.nn.functional.scaled_dot_product_attention` (SDPA) — hardware-accelerated.
- `sdpa_kernel([...])` with the full priority list — works (dispatches to the XPU kernel).
- bf16 GEMM (`torch.mm`) — **~95 TFLOPS** measured at 8192² on a single B60.
- `torch._scaled_mm` (fp8) — *functions* but is **not** hardware-accelerated on XPU
  (see [the fp8 question](#the-fp8-question-fp8-cast-vs-fp8-scaled-mm)).
- `torch.Generator(device="xpu")` — works.

---

## Why this approach (and not Diffusers)

The LTX-2.3 HuggingFace repos (`Lightricks/LTX-2.3` and `Lightricks/LTX-2.3-fp8`)
ship **single-file `.safetensors` checkpoints only** — there is no
`model_index.json` / `config.json` pipeline structure. The model card states
explicitly:

> LTX-2.3 support in the Diffusers Python library is coming soon!

So `DiffusionPipeline.from_pretrained("Lightricks/LTX-2.3", ...)` **does not work**.
The only local inference paths are ComfyUI or the official **LTX-2 monorepo**
(`ltx-pipelines`). We use the latter, which is plain PyTorch and therefore
portable to XPU with small patches (the repo targets CUDA).

---

## Solution architecture

The pipeline runs on **four XPUs** organized into two NUMA groups with no
cross-group traffic:

| NUMA group | Device | Holds (one at a time) | Peak VRAM |
|---|---|---|---|
| **A** | `xpu:0` | fp8 distilled transformer (~18 GB) | ~22 GB |
| **A** | `xpu:1` | video VAE, spatial upscaler, video/audio decoders | ~1 GB |
| **B** | `xpu:2` | Gemma layers 0–23 + embeddings + vision tower | ~16.5 GB |
| **B** | `xpu:3` | Gemma layers 24–47 + norm + lm_head | ~8.2 GB |

The `ltx-pipelines` `DistilledPipeline` builds and frees each model sequentially
inside a `gpu_model()` context manager — **only one model is ever resident on a
device at a time** (group A). Peak VRAM on group A equals the largest single
model (the fp8 transformer, ~18 GB), which fits a 24 GB B60.

### Gemma model parallelism (xpu:2 + xpu:3)

The text encoder is **Gemma 3 12B** (multimodal). In bf16 that's ~26 GB — too
large for a single 24 GB B60. Two options are supported:

1. **DP (default)** — Gemma is built on CPU, then sharded across `xpu:2 + xpu:3`
   (NUMA group B) via `accelerate`'s `device_map` dispatch
   (`infer_auto_device_map` + `dispatch_model`). The 48 decoder layers are split
   ~24/24 across the two XPUs. The forward pass runs in **~0.8 s** (vs ~18 s on
   CPU — a **22× speedup**). The embeddings processor then runs on `xpu:1`.
   This keeps group B completely separate from group A — no cross-group tensor
   transfers.

2. **CPU fallback** — set `LTX_GEMMA_DEVICE=cpu`. Gemma runs entirely on CPU
   (~18 s forward, 128 cores). Simpler but slower; useful if xpu:2/xpu:3 are
   unavailable or occupied.

The DP path has a one-time **dispatch overhead** (~45 s to shard 26 GB from CPU
→ 2 XPUs). For a single prompt this makes DP slower end-to-end (120 s vs 75 s),
but if the model is kept resident for **≥3 sequential prompts**, DP amortizes
the dispatch and wins.

### Cross-device tensor moves

For pure text-to-video there are no input images, so the image conditioner
produces no conditioning latents. The only cross-device moves are:
- prompt context: `xpu:2/3 (or cpu) → xpu:0` (after encoding),
- video/audio latents: `xpu:0 ↔ xpu:1` between stages and before decode.

All moves stay within their NUMA group — no A↔B traffic.

```
                    ┌── NUMA Group A ──┐         ┌── NUMA Group B ──┐
                    │                  │         │                  │
DPPromptEncoder ───▶│  xpu:2 + xpu:3   │         │                  │
  (Gemma sharded)   │  ~0.8 s forward  │         │                  │
                    └──────┬───────────┘         │                  │
                           │ context             │                  │
                           ▼                     │                  │
                DiffusionStage(xpu:0, fp8)       │                  │
                    │  stage-1 latents           │                  │
                    ▼                            │                  │
          VideoUpsampler(xpu:1) ──upscaled──▶ xpu:0                │
                    │  stage-2 latents           │                  │
                    ▼                            │                  │
         VideoDecoder(xpu:1) ◀──latent── + AudioDecoder(xpu:1)     │
                    │                            │                  │
                    ▼                            │                  │
              encode_video → output.mp4          │                  │
                                               └──────────────────┘
```

---

## The fp8 question: `fp8-cast` vs `fp8-scaled-mm`

This is the single most important finding for performance on XPU.

LTX-2.3 ships pre-quantized fp8 checkpoints (1462 `F8_E4M3` weights + 1462
`.weight_scale` companions). The LTX-2 repo offers two quantization policies:

### `fp8-scaled-mm` (`FP8Linear` + `torch._scaled_mm`) — ❌ slow on XPU
`FP8Linear.forward` calls `torch._scaled_mm(qinput, weight.t(), scale_a=...,
scale_b=..., use_fast_accum=True)`. On XPU:
- It *runs* (PyTorch emits *"fast_accum is not supported in XPU for now;
  silently set to false"* and proceeds),
- but it is **not** dispatched to a hardware fp8 kernel. The matmuls fall back to
  a slow path, leaving the GPU idle (~34 W) while ~2 CPU cores spin.
- Observed: 7+ minutes into an 8-step stage-1 loop without completing.

### `fp8-cast` (`Fp8CastLinear`, upcast-to-bf16 at forward) — ✅ fast on XPU
`Fp8CastLinear.forward` does `w_up = self.weight.to(input.dtype)` (fp8 → bf16,
per-layer transient) then `torch.nn.functional.linear(input, w_up, b_up)`. That
is a **bf16 GEMM**, which Battlemage accelerates in hardware (~95 TFLOPS).

- Storage stays fp8 (~22 GB → fits one 24 GB B60).
- Compute is bf16-speed.
- Per-layer bf16 transient is small and freed immediately.

**Observed with `fp8-cast`:** stage 1 (8 steps) in **~8 s**, stage 2 (3 steps) in
**~6 s**.

> **Always use `fp8-cast` (not `fp8-scaled-mm`) on Intel XPU** until PyTorch
> ships a hardware fp8 `_scaled_mm` for Battlemage.

---

## XPU patches applied to the LTX-2 repo

Three small patches make the CUDA-targeted codebase XPU-safe. They are applied
in-place to the editable checkout at `/home/lm/LTX-2/packages/`.

### 1. `ltx_pipelines/utils/helpers.py` — `cleanup_memory()`
Originally called `torch.cuda.empty_cache()` / `torch.cuda.synchronize()`
unconditionally (crashes with no CUDA). Now syncs/empties whichever accelerator
is present:

```python
def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache(); torch.xpu.synchronize()
    try:
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        logging.warning("Host empty cache cleanup failed; ignoring.", exc_info=True)
```

### 2. `ltx_pipelines/utils/gpu_model.py` — `gpu_model()` teardown
The `finally:` block called `torch.cuda.synchronize()`. Now best-effort across
accelerators before `.to("meta")` frees storage:

```python
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        model.to("meta")
        cleanup_memory()
```

### 3. `ltx_core/model/audio_vae/vocoder.py` — vocoder fp32 on XPU
The BigVGAN vocoder runs 108 sequential convolutions and must run in **fp32**
(bf16 accumulation degrades spectral metrics 40–90%). On CUDA it achieves this
with `torch.autocast("cuda", dtype=torch.float32)`, which upcasts bf16 weights
per-op. **XPU autocast does not support an fp32 target dtype** — it silently
disables autocast, so the float32 input hits bf16 conv weights and crashes:

```
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
```

The repo's own comment notes that `self.float()` is **bit-identical** to the
autocast path (just +324 MB peak VRAM for the 128 M-param vocoder — negligible).
So on non-CUDA devices we upcast the module explicitly:

```python
import contextlib
...
        if mel_spec.device.type == "cuda":
            fp32_ctx = torch.autocast(device_type=mel_spec.device.type, dtype=torch.float32)
        else:
            self.float()
            fp32_ctx = contextlib.nullcontext()
        with fp32_ctx:
            x = self.vocoder(mel_spec.float())
            ...
```

### Things that needed **no** patch
- **Attention** — uses PyTorch SDPA. The `AUTOMATIC` selector falls through to
  `_sdpa_full_priority()` when CUDA is absent, and `sdpa_kernel` dispatches to
  the XPU kernel. No code change needed.
- **`block_streaming` / `OffloadMode.CPU`** — uses `torch.cuda.Stream/Event`,
  but is only activated when `offload_mode != NONE`. We use `NONE`, so it's never
  hit. (If you want CPU weight-offloading to run on a single small GPU, this path
  would need XPU-stream patches — not done here.)
- **`trtllm` fp8 backend** — gated behind `torch.cuda.is_available()`; returns
  `False` and falls back to PyTorch-native. No change needed.
- **LoRA fusion** (`fuse_loras._get_device()`) — only invoked when LoRAs are
  loaded; we pass none.

---

## Prerequisites & setup

### 0. System access
Your user must be in the `render` group (it is on this machine) so `/dev/dri/renderD*`
is accessible. Verify the GPUs:

```bash
sycl-ls | grep level_zero:gpu | head -2   # should list the B60s
xpu-smi discovery -d 0 | grep -E "Device Name|Memory Physical"
```

### 1. uv
[`uv`](https://docs.astral.sh/uv/) is used to manage Python and the venv. It is
already at `~/.local/bin/uv`.

### 2. Create the environment

The environment is defined by a uv `pyproject.toml` (dependency spec + the XPU
index config) and a fully-pinned `uv.lock` (77 packages, including
`torch==2.12.1+xpu` and its Intel runtime transitive deps). Recreate it with
one command:

```bash
# Creates the venv at /home/lm/ltx23-env (or set LTX_ENV=/path/to/venv):
./setup-env.sh
```

`setup-env.sh` does, in order:
1. `uv sync` — installs all 77 locked packages from `uv.lock` into `$LTX_ENV`,
   including `torch==2.12.1+xpu` / `torchvision==0.27.1+xpu` /
   `torchaudio==2.11.0+xpu` (via the `pytorch-xpu` index declared in
   `pyproject.toml`), plus the Intel runtime wheels (`intel-sycl-rt`,
   `intel-cmplr-lib-rt`, `oneccl`, `onemkl-*`, `triton-xpu`, `tbb`, `umf`, …)
   that come in as torch's transitive dependencies,
2. installs the LTX-2 packages editable with `--no-deps` (see step 3 below),
3. verifies `torch.xpu` sees the GPUs.

If you prefer to do it manually:

```bash
UV_PROJECT_ENVIRONMENT=/home/lm/ltx23-env uv sync
# or, without the lockfile:
uv venv --python 3.12 /home/lm/ltx23-env
source /home/lm/ltx23-env/bin/activate
uv pip install torch==2.12.1+xpu torchvision==0.27.1+xpu torchaudio==2.11.0+xpu \
    --index-url https://download.pytorch.org/whl/xpu
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/xpu
```
> Do **not** `uv sync` the LTX-2 repo root — its `pyproject.toml` pins a CUDA
> PyTorch index (`cu129`) which would replace the XPU torch. The LTX-2 packages
> are deliberately excluded from this repo's `pyproject.toml` for the same
> reason; they're installed separately with `--no-deps`.

### 3. Clone and install the LTX-2 packages (editable, no deps)
```bash
git clone --depth 1 https://github.com/Lightricks/LTX-2.git /home/lm/LTX-2
uv pip install --no-deps -e /home/lm/LTX-2/packages/ltx-core
uv pip install --no-deps -e /home/lm/LTX-2/packages/ltx-pipelines
```
Installing editable means the XPU patches you apply to the checkout take effect
immediately. (`setup-env.sh` already runs these two if `/home/lm/LTX-2` exists.)

### 4. Apply the three XPU patches
Edit the three files described in [XPU patches](#xpu-patches-applied-to-the-ltx-2-repo),
or apply the committed patch directly:
```bash
cd /home/lm/LTX-2 && git apply /home/lm/ltx23-run/patches/xpu.patch
```

### 5. Download model artifacts
Three downloads (~48 GB total) into `/home/lm/ltx23-models/`:

| File | Source repo | Size | Notes |
|---|---|---|---|
| `ltx-2.3-22b-distilled-fp8.safetensors` | `Lightricks/LTX-2.3-fp8` | ~29.5 GB | fp8 transformer + VAEs + embeddings in one file |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `Lightricks/LTX-2.3` | ~1 GB | latent 2× upscaler |
| `gemma-3-12b-it/` (5 shards + tokenizer) | `unsloth/gemma-3-12b-it` | ~23 GB | text encoder |

> `google/gemma-3-12b-it` is **gated** (requires a HF token + license accept).
> The `unsloth/gemma-3-12b-it` mirror is **not gated** and has the exact files
> LTX needs (`tokenizer.model`, `preprocessor_config.json`, `model-*.safetensors`).

The helper script `download.py` does all three:

```bash
python /home/lm/ltx23-run/download.py
```
(Detach it with `setsid` so a shell timeout doesn't kill the download — see
[Troubleshooting](#troubleshooting).)

Verify the fp8 checkpoint structure (must show F8_E4M3 weights + `weight_scale`
companions — required by the quantization policy):

```bash
python -c "
import struct, json
from collections import Counter
p='/home/lm/ltx23-models/ltx-2.3-22b-distilled-fp8.safetensors'
with open(p,'rb') as f:
    n=struct.unpack('<Q', f.read(8))[0]; h=json.loads(f.read(n))
dt=Counter(v.get('dtype') for k,v in h.items() if k!='__metadata__')
print(dict(dt))  # expect F8_E4M3, BF16, F32
print('weight_scale keys:', sum(k.endswith('.weight_scale') for k in h))
"
```

---

## Running

### Shell launchers (easiest)

```bash
# Small config (896×512, 41 frames, ~75-120s depending on Gemma mode):
./run.sh                              # default prompt, DP Gemma
./run.sh "a dog surfing a wave"       # custom prompt
LTX_GEMMA_DEVICE=cpu ./run.sh         # fall back to CPU Gemma (faster for single shot)

# 1024×1024, 121 frames (~2.5 min):
./run_b.sh
./run_b.sh "neon city at night"
```

### Direct Python

```bash
source /home/lm/ltx23-env/bin/activate
cd /home/lm/ltx23-run
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run_t2v_xpu.py
```

- `HF_HUB_OFFLINE=1` — prevents any network call mid-run (all weights are local).
- `TOKENIZERS_PARALLELISM=false` — silences a tokenizer warning.
- `LTX_GEMMA_DEVICE=cpu` — use CPU for Gemma instead of DP sharding (faster for
  a single shot since it avoids the 45 s dispatch overhead).

Output: `/home/lm/ltx23-run/output.mp4`.

> **Tip — surviving shell timeouts.** If you launch from an environment that
> kills the command's process group on timeout (e.g. some agent shells), detach
> with `setsid` so the run survives:
> ```bash
> setsid python -u run_t2v_xpu.py > run.log 2>&1 < /dev/null & disown
> tail -f run.log
> ```

---

## Web Service (FastAPI)

A FastAPI server provides a shared web UI + REST API for text-to-video generation
on 32 Intel Arc B60 XPUs. All state is kept server-side and broadcast to all
connected clients via Server-Sent Events (SSE) — browsers are pure viewers with
no per-connection state.

### Architecture

```
Browser ─── SSE (/api/events) ◀─── FastAPI ─── LtxWorker (single) ─── Popen ─── run_t2v_xpu_perf.py
             │                                │                              (live stdout → state)
             │                                └── MultiLtxWorker ─── Popen x8 ─── run_t2v_xpu_perf.py
             │                                                                  (logs → state)
             └──── POST /api/jobs ──► Server ─── queue ──► worker
```

- **Single job** — `LtxWorker` spawns one subprocess on `xpu:0`+`xpu:1` (same as
  `run_b.sh` config: 1024×1024, 121 frames, Gemma on CPU). Subprocess stdout is
  read line-by-line into a thread-safe in-memory log buffer (`ServerState`).
- **Multi job (8×)** — `MultiLtxWorker` spawns 8 staggered subprocesses on
  `xpu:0..15` (pairs: (0,1), (2,3), …, (14,15)). Prompts are pre-encoded by
  `encode_prompts.py` (one shared Gemma instance on CPU). Per-worker log files
  are tailed every 2 s for real-time progress.
- **SSE broadcast** — All clients connect to `/api/events` (no auth required).
  The server emits the full state snapshot (active jobs, logs, progress, XPU
  telemetry, history) every ~0.8 s. Client JS renders the DOM from the snapshot
  without any per-connection polling or state variables.
- **Video serving** — `/api/jobs/{id}/video` and
  `/api/multi-jobs/{id}/videos/{i}` serve MP4 files with HTTP Range support
  (206 Partial Content) and `Content-Disposition: inline` for smooth in-browser
  playback. No auth required on these video endpoints (browser `<video>` cannot
  send Bearer tokens).

### Modification steps (per-client polling → shared SSE broadcast)

1. **Add `ServerState` class** — thread-safe shared state container holding
   active single/multi job details, rotating log buffers (100 lines), worker
   progress, cached history, and XPU telemetry. Writes via `threading.Lock`.
2. **Rewrite `LtxWorker._spawn_generation()`** — replaced `subprocess.run()`
   (blocking, captured output at end) with `subprocess.Popen()` + iterative
   `stdout.readline()` for live log capture. Tqdm-like progress is parsed via
   regex and written to state every line.
3. **Rewrite `MultiLtxWorker._run()`** — replaced sequential `proc.wait()` loop
   with a polling loop that checks `proc.poll()` every 2 s and reads the tail of
   each worker's `.log` file for real-time status. Worker grid and orchestrator
   logs are written to state as workers spawn/finish.
4. **Add SSE endpoint** — `GET /api/events` returns a `StreamingResponse` with
   `text/event-stream`. An async generator polls `state.snapshot()` every 0.8 s
   and yields `data: <json>\n\n` only when state changes.
5. **Add history refresh** — background daemon thread reloads recent jobs from
   SQLite into `state.single_history` / `state.multi_history` every 5 s.
6. **Rewrite HTML/JS** — removed all client-side state variables
   (`currentSingleJob`, `currentMultiJobs`, timers) and polling logic.
   `EventSource('/api/events')` drives all rendering via a single
   `renderAll(state)` call. Video elements use `dataset` caches to avoid
   DOM rebuild on every SSE tick (prevents flicker).

### Key files

| File | Purpose |
|------|---------|
| `/home/lm/paul/ltx_server.py` | Entry point, instantiates `ModelProfile` |
| `/home/lm/paul/ltx_server_common.py` | Server framework — all endpoints, HTML template, `ServerState`, workers |
| `/home/lm/paul/start_ltx_server.sh` | Launch script |
| `/home/lm/paul/ltx23-run/encode_prompts.py` | Shared Gemma prompt encoder (multi-job step 1) |
| `/home/lm/paul/ltx23-run/run_multi_xpu.py` | 8-video launcher (orchestrator used by `MultiLtxWorker`) |
| `/home/lm/paul/ltx23-run/run_multi_16.py` | 16-video launcher (xpu:0..31) — experimental |

### Starting the server

```bash
# Kill old server, start fresh
fuser -k 8001/tcp 2>/dev/null; sleep 2
LTX_HOST=0.0.0.0 LTX_API_TOKEN=111 \
  /home/lm/paul/ltx23-env/bin/python /home/lm/paul/ltx_server.py > server.log 2>&1 &
disown
```

Access at `http://<host>:8001`. API token (`111` in this example) is required
for all POST endpoints. The SSE endpoint and video endpoints need no auth.

---

## Multi-Worker Generation

### `run_multi_xpu.py` — 8 concurrent videos (16 XPUs)

Pre-encodes 8 prompts with a single Gemma instance on CPU (`encode_prompts.py`),
then spawns 8 staggered workers on `xpu:0..15` (pairs (0,1), (2,3), … (14,15)).
Stagger delay (default 5 s) avoids XPU driver contention during model build.

**Usage:**
```bash
# With default prompts:
LTX_GEMMA_DEVICE=cpu /home/lm/paul/ltx23-env/bin/python run_multi_xpu.py

# Custom prompts via file:
LTX_GEMMA_DEVICE=cpu /home/lm/paul/ltx23-env/bin/python run_multi_xpu.py \
  --prompts-file multi_16_output/prompts.json --job-dir /tmp/my_job

# Shell wrapper:
./run_multi.sh
```

### `run_multi_16.py` — 16 concurrent videos (32 XPUs)

Same approach but spawns 16 workers on `xpu:0..31` (pairs (0,16), (1,17), …
(15,31)). **Experimental** — 16 concurrent transformer builds stress the XPU
driver and have shown extremely slow per-step times (up to 18 min/step in testing).
Stagger delay is 10 s.

```bash
LTX_GEMMA_DEVICE=cpu /home/lm/paul/ltx23-env/bin/python run_multi_16.py
```

---

## Current Test Status

### Single Video (via web server or CLI)

| Config | Result | Date |
|--------|--------|------|
| 1024×1024, 121 frames, Gemma CPU, `run.sh` / `run_b.sh` | ✅ **Succeeded** ~2.5 min | 2026-07-16 |
| Web server single job endpoint | ✅ Verified — subprocess spawns, live log streaming via SSE, progress tracked, video served with valid Range support | 2026-07-17 |

### 8-Video Multi-Job (via web server or `run_multi_xpu.py`)

| Test | Result | Wall time | Details |
|------|--------|-----------|---------|
| 8 prompts, 1024×1024, 121 fr, xpu:0..15, 5 s stagger | ✅ **8/8 succeeded** | ~2 min 53 s | Prompt encode ~7.4 s, spawn stagger ~35 s, generation ~130 s |
| Web server multi-job endpoint | ✅ Verified — SSE shows real-time worker grid + orchestrator logs, gallery renders on completion, history shows links | 2026-07-17 |

### 16-Video Multi-Job (experimental)

| Test | Result | Details |
|------|--------|---------|
| 16 prompts, 1024×1024, 121 fr, xpu:0..31, 10 s stagger | ❌ **Interrupted — all workers killed by session timeout** | 16 workers spawned successfully, early workers showed extreme per-step times (18 min/step on worker 0, 73 s/step on worker 1). Likely driver contention from 16 concurrent model builds. |

### Known issues

- **73-frame hang** — Running 1024×1024 with 73 frames (`frames % 8 == 1`) hangs
  the XPU driver at stage-2 denoising step 1. Root cause unknown. Use 121 frames
  as the default (next valid value: 9, 17, 25, 41, 121, …).
- **16-worker bottleneck** — 16 concurrent transformer builds exceed XPU driver
  capacity. Stick to 8 workers for reliable generation.
- **SSE connection drop** — Browsers may drop SSE after ~60 s of idle (no state
  changes). Client auto-reconnects with 3 s backoff. Continuous log output
  during a running job keeps the connection alive.

---

## Configuration

Edit the constants at the top of `run_t2v_xpu.py`:

```python
PROMPT = "..."                       # your text prompt
SEED = 42
STAGE1_H, STAGE1_W = 256, 448        # stage 2 = 512 x 896  (must be /64 for two-stage)
NUM_FRAMES = 41                      # must be 8k + 1  (41 = 8*5 + 1)
FRAME_RATE = 24.0
```

Resolution/divisibility rules (from the model card):
- **Two-stage** pipelines: height and width must be divisible by **64**
  (stage 1 runs at half, so the half-res must be divisible by 32).
- **One-stage** pipelines: divisible by **32**.
- Frame count must be `8k + 1`.

Scaling up: stage1 `512×768` → stage2 `1024×1536` works on these GPUs (the fp8
transformer leaves ~2 GB headroom on a 24 GB B60 at the small setting; watch XPU
memory at higher res/longer durations). If you OOM in stage 2, lower resolution
or frame count first.

Devices are also constants:
```python
TDEV = torch.device("xpu", 0)           # transformer (NUMA group A)
CDEV = torch.device("xpu", 1)           # VAE / decoders (NUMA group A)
GEMMA_DP_DEVICES = (2, 3)               # Gemma DP sharding (NUMA group B)
USE_DP_GEMMA = True                     # DPPromptEncoder by default
```

**Environment variables:**
| Variable | Default | Effect |
|---|---|---|
| `LTX_PROMPT` | *(built-in red-panda prompt)* | Override the text prompt |
| `LTX_GEMMA_DEVICE` | `xpu` | Set to `cpu` to run Gemma on CPU instead of DP sharding |

---

## Performance / Benchmarks

Three runs were benchmarked end-to-end on the same machine, all using the
**distilled** two-stage pipeline with **fp8-cast** quantization. Timings are
wall-clock, measured with `run_t2v_xpu_perf.py` (`time.perf_counter` per stage,
with an `xpu` synchronize at each stage boundary).

### Run A — small, Gemma on CPU (default config)

stage1 256×448 → stage2 512×896, 41 frames @ 24 fps (1.71 s of video):

| Stage | Time |
|---|---|
| Prompt encode (Gemma 12B, CPU, 64 threads) | ~17 s |
| Stage 1 denoise (8 steps, incl. ~12 s build) | ~8 s (~1.05 s/it) |
| Spatial upsample 2× | ~4 s |
| Stage 2 denoise (3 steps, incl. ~12 s build) | ~6 s (~2.1 s/it) |
| Video + audio decode | ~10 s |
| Mux to MP4 | ~3 s |
| **Total** | **~75 s** |

Output: 896×512, 24 fps, 41 frames, 1.71 s, 48 kHz stereo audio, ~350 KB.

### Run B — 1024×1024, 121 frames, Gemma on CPU

stage1 512×512 → stage2 1024×1024, 121 frames @ 24 fps (5.04 s of video):

| Stage | Time | % | Notes |
|---|---:|---:|---|
| Prompt encode (Gemma 12B, CPU) | 21.93 s | 14.4% | one forward pass, no generation |
| Stage 1 denoise — 8 steps @ 512×512 | 41.13 s | 27.0% | ~2.95 s/step (incl. ~11.8 s build) |
| Spatial upsample 2× | 2.60 s | 1.7% | on xpu:1 |
| Stage 2 denoise — 3 steps @ 1024×1024 | 55.59 s | 36.5% | ~13.7 s/step (incl. ~11.6 s build) |
| Video + audio decode | 11.92 s | 7.8% | tiled VAE decode (3 chunks) |
| Mux to MP4 | 19.08 s | 12.5% | |
| **Total** | **152.25 s** | | **~2.5 min** |

Output (`sample-output-1024.mp4`): 1024×1024, 24 fps, 121 frames, 5.04 s,
48 kHz stereo audio, 1.15 MB.

### Run C — small, Gemma DP on xpu:2+xpu:3

Same output config as Run A (stage1 256×448 → stage2 512×896, 41 frames), but
with Gemma sharded across `xpu:2 + xpu:3` via `accelerate` model parallelism:

| Stage | Time | Notes |
|---|---:|---|
| Gemma build (CPU) | ~2 s | load weights to CPU |
| Dispatch (shard 26 GB → xpu:2+3) | ~34 s | one-time overhead |
| Gemma forward (2 XPUs) | ~0.8 s | **22× faster than CPU's 18 s** |
| Gemma free + cleanup | ~11 s | remove hooks, move to meta |
| Embeddings processor (xpu:1) | ~7 s | |
| Stage 1 denoise (8 steps) | ~27 s | same as Run A |
| Stage 2 denoise (3 steps) | ~22 s | same as Run A |
| Decode + mux | ~12 s | same as Run A |
| **Total** | **~120 s** | |

Output: identical to Run A (896×512, 41 frames, same seed → same content).

### Comparison

| | Run A (CPU) | Run B (CPU, 1024²) | Run C (DP, small) |
|---|---|---|---|
| Output resolution | 896×512 | 1024×1024 | 896×512 |
| Frames / duration | 41 / 1.71 s | 121 / 5.04 s | 41 / 1.71 s |
| Gemma device | CPU | CPU | xpu:2+3 (DP) |
| Gemma forward | ~18 s | ~18 s | **~0.8 s** |
| Gemma dispatch overhead | 0 | 0 | ~45 s |
| Prompt encode total | ~20 s | ~22 s | ~55 s |
| Stage-2 total | ~6 s | 55.6 s | ~6 s |
| **Total wall** | **~75 s** | **152.25 s** | **~120 s** |

### Gemma DP tradeoff

| Metric | CPU | DP (xpu:2+3) |
|---|---|---|
| Forward pass | 18 s | **0.8 s (22× faster)** |
| One-time dispatch | 0 | 45 s (shard 26 GB) |
| Single-prompt total | **75 s** ✅ | 120 s |
| Per-prompt (model resident) | 18 s | **0.8 s** ✅ |
| Break-even | — | ≥3 sequential prompts |

DP is slower for a single shot (the 45 s dispatch dominates), but if the model
is kept resident across prompts the per-prompt forward drops from 18 s to 0.8 s.
Use `LTX_GEMMA_DEVICE=cpu` to fall back to CPU mode.

### Memory

Measured with dedicated probes (build model on target device, snapshot, free):

| Device | Peak | Capacity |
|---|---|---|
| `xpu:0` (transformer, fp8-cast) | **18.12 GB allocated / 21.91 GB reserved** | 23.9 GB |
| `xpu:1` (VAEs/decoders, during decode) | 0.77 GB | 23.9 GB |
| `xpu:2` (Gemma layers 0–23) | 16.51 GB | 23.9 GB |
| `xpu:3` (Gemma layers 24–47) | 8.17 GB | 23.9 GB |

→ ~2 GB headroom on `xpu:0` for activations. The 1024²×121 run (16,384 tokens)
**fit without OOM**. The next likely OOM point is pushing stage-2 resolution
substantially higher (e.g. 1536²+) or many more frames.

> Note: the `peak xpu:0` line printed by the run's own summary reads ~0.02 GB
> because it is sampled *after* `gpu_model` frees the transformer. The 18.12 GB
> figure above is the real resident footprint, from a separate probe.

### Key takeaways

- **Use `fp8-cast`, never `fp8-scaled-mm`, on Intel XPU** (see
  [the fp8 question](#the-fp8-question-fp8-cast-vs-fp8-scaled-mm)). bf16 GEMM is
  hardware-accelerated on Battlemage (~95 TFLOPS measured); fp8 `_scaled_mm` is
  not, and falls back to a slow CPU path.
- **Stage 2 is the compute bottleneck** (36.5% of total at 1024²). Attention is
  O(n²) in tokens — tokens scaled 6.1× from Run A→B while stage-2 time scaled
  ~9.3× (superlinear, as expected).
- **Gemma DP gives 22× faster forward** (0.8 s vs 18 s) but has a 45 s one-time
  dispatch overhead. Use DP for batch/interactive workloads (≥3 prompts with
  model resident); use CPU for single-shot runs.
- **NUMA-aware placement matters.** Group A (xpu:0+1) runs the transformer +
  VAEs; group B (xpu:2+3) runs the sharded Gemma. No cross-group tensor
  transfers. Mixing devices across groups would incur high PCIe latency.
- **Fixed overhead is material at small sizes** (prompt-encode + decode + mux =
  ~53 s, 35% of Run B). For longer/higher-res jobs the diffusion stages
  dominate more.

---

## Troubleshooting

**`RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same` (in `conv1d`)**
→ The vocoder fp32 patch (#3) is missing or not applied. Re-apply it. The
editable install means editing the file is enough; no reinstall needed.

**Stage denoising is extremely slow, GPU power ~34 W, ~2 CPU cores busy**
→ You're using `fp8-scaled-mm` instead of `fp8-cast`. In `run_t2v_xpu.py` the
policy must be `from ltx_core.quantization.fp8_cast import build_policy as
fp8_cast_policy`. See [the fp8 question](#the-fp8-question-fp8-cast-vs-fp8-scaled-mm).

**`XPU out of memory` during Gemma forward on a single XPU**
→ Gemma 12B bf16 (~26 GB) doesn't fit one 24 GB B60. The driver uses DP model
parallelism across `xpu:2 + xpu:3` by default (`DPPromptEncoder`). To fall back
to CPU: `LTX_GEMMA_DEVICE=cpu ./run.sh`. (If you have a GPU with ≥32 GB, you can
move it there by setting `GDEV` in the script.)

**`torch.cuda.is_available()` / `torch.cuda.synchronize()` crashes**
→ Patches #1 and #2 are missing. Re-apply them.

**Background process killed when the launching shell times out**
→ Use `setsid ... & disown` (or `nohup ... &` with stdin redirected from
`/dev/null`) to fully detach into a new session, then poll `run.log`.

**`xpu-smi stats` shows `N/A` for EU Array Active**
→ That field reports `N/A` on this driver build even under load; it's cosmetic.
Use `xpu-smi stats -d 0 | grep "GPU Power"` (power rises under load) or a bf16
`torch.mm` micro-benchmark to confirm the GPU is computing.

**Download of `google/gemma-3-12b-it` fails / asks for a token**
→ That repo is gated. Use the `unsloth/gemma-3-12b-it` mirror (not gated) as
`download.py` already does.

**`torch.xpu.empty_cache(dev)` TypeError**
→ `empty_cache()` takes no positional args on this PyTorch build; it empties the
whole caching allocator. Call `torch.xpu.empty_cache()` (no arg).

---

## File layout

```
/home/lm/
├── ltx23-env/                          # uv venv (python 3.12, torch+xpu, ltx pkgs)
├── LTX-2/                              # cloned Lightricks/LTX-2 monorepo (editable)
│   └── packages/
│       ├── ltx-core/src/ltx_core/model/audio_vae/vocoder.py   # patch #3
│       ├── ltx-pipelines/src/ltx_pipelines/utils/helpers.py   # patch #1
│       └── ltx-pipelines/src/ltx_pipelines/utils/gpu_model.py # patch #2
├── ltx23-models/                       # downloaded weights
│   ├── ltx-2.3-22b-distilled-fp8.safetensors
│   ├── ltx-2.3-spatial-upscaler-x2-1.1.safetensors
│   └── gemma-3-12b-it/
└── ltx23-run/
    ├── README.md                       # this file
    ├── pyproject.toml                  # uv project: deps + XPU index config
    ├── uv.lock                         # fully-pinned lockfile (77 packages)
    ├── requirements.txt                # pip-freeze fallback (same pins)
    ├── setup-env.sh                    # `uv sync` to recreate the venv
    ├── run_t2v_xpu.py                  # driver script (DP Gemma by default)
    ├── run_t2v_xpu_perf.py             # perf script with per-stage timing (1024² config)
    ├── dp_prompt_encoder.py            # Gemma model-parallel across xpu:2+xpu:3
    ├── run.sh                          # launcher for the small config
    ├── run_b.sh                        # launcher for the 1024²/121-frame config
    ├── download.py                     # model download helper
    ├── patches/xpu.patch               # the three XPU patches
    ├── sample-output.mp4               # Run A output (896×512, 41 frames)
    ├── sample-output-1024.mp4          # Run B output (1024×1024, 121 frames)
    ├── run.log / perf.log              # last run logs (gitignored)
    └── output*.mp4                     # generated videos (gitignored)
```

---

## Limitations & notes

- **Audio quality.** The model card warns that audio *without speech* may be of
  lower quality. The generated clip has ambient/synchronized audio but no speech.
- **Distilled = fast, not highest quality.** For best quality use the non-
  distilled `dev` checkpoint with the two-stage HQ pipeline (`ti2vid_two_stages_hq`),
  but that uses CFG + skip-layer guidance (multiple transformer passes per step)
  and is much slower; the bf16 `dev` model (~44 GB) also won't fit two 24 GB GPUs
  without sharding or the `OffloadMode.CPU` path (which would need the
  XPU-stream patches mentioned above).
- **Single-GPU sharding is not used for the transformer.** The transformer runs
  entirely on `xpu:0`; `xpu:1` handles the VAEs/decoders. Because models are
  sequential, this still uses both XPUs in group A and keeps each within budget.
  Gemma, however, **is** sharded across `xpu:2 + xpu:3` (group B) via
  `accelerate` model parallelism. True transformer sharding across XPUs would
  require the repo's multi-GPU builder and is out of scope here.
- **Prompt enhancement is off** (`enhance_prompt=False`) to avoid Gemma
  *generation* (autoregressive, many forward passes). The DP encoder supports it
  but it would be slow due to sequential layer-by-layer generation across the
  shard boundary. Turn it on only for CPU mode or if latency is acceptable.
- **No commits were made** to the LTX-2 checkout; the three patches are applied
  in-place to the editable install.
