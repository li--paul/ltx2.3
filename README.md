# LTX-2.3 Text-to-Video on Intel Arc Pro B60 (XPU)

End-to-end instructions for running **Lightricks LTX-2.3** text-to-video
generation on **Intel Arc Pro B60 (Battlemage) XPUs** using PyTorch's native XPU
backend — no CUDA, no Diffusers, no ComfyUI required.

This README documents a working setup that produces a synchronized
**video + audio** MP4 from a text prompt using the first two B60 GPUs on the
machine.

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
10. [Expected output & timings](#expected-output--timings)
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

The `ltx-pipelines` `DistilledPipeline` builds and frees each model sequentially
inside a `gpu_model()` context manager — **only one model is ever resident on a
device at a time**. Peak VRAM therefore equals the largest single model, not
their sum. We exploit this to fit a 22B model across two 24 GB GPUs:

| Device | Holds (one at a time) | Peak VRAM |
|---|---|---|
| `xpu:0` | fp8 distilled transformer (~22 GB) | ~22 GB |
| `xpu:1` | video VAE, spatial upscaler, video decoder, audio VAE+vocoder | a few GB |
| `cpu` | Gemma-3-12B text encoder + embeddings processor | RAM only |

**Why Gemma on CPU?** The text encoder is Gemma 3 12B (multimodal). In bf16 that's
~24 GB — slightly more than a single 24 GB B60. Since the pipeline only runs one
forward pass through Gemma (no generation, `enhance_prompt=False`), running it on
CPU is the simplest robust choice (~17 s on 64 threads). It is freed before the
transformer runs, so it never competes for XPU memory.

**Cross-device tensor moves.** For pure text-to-video there are no input images,
so the image conditioner produces no conditioning latents. The only cross-device
moves are:
- prompt context: `cpu → xpu:0` (after encoding),
- video/audio latents: `xpu:0 ↔ xpu:1` between stages and before decode.

The driver script `run_t2v_xpu.py` replicates `DistilledPipeline.__call__` with
these per-block device assignments and moves.

```
PromptEncoder(cpu) ──context──▶ DiffusionStage(xpu:0, fp8)
                                      │  stage-1 latents
                                      ▼
                          VideoUpsampler(xpu:1) ──upscaled latent──▶ xpu:0
                                      │  stage-2 latents
                                      ▼
                   VideoDecoder(xpu:1) ◀──latent──  +  AudioDecoder(xpu:1)
                                      │
                                      ▼
                                encode_video → output.mp4
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
```bash
uv venv --python 3.12 /home/lm/ltx23-env
source /home/lm/ltx23-env/bin/activate
```

### 3. Install PyTorch (XPU build) and helpers
```bash
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/xpu
uv pip install einops "scipy>=1.14" av tqdm pillow OpenImageIO \
    "transformers>=4.52" accelerate sentencepiece safetensors imageio imageio-ffmpeg
```
> Do **not** `uv sync` the LTX-2 repo root — its `pyproject.toml` pins a CUDA
> PyTorch index (`cu129`) which would replace the XPU torch.

### 4. Clone and install the LTX-2 packages (editable, no deps)
```bash
git clone --depth 1 https://github.com/Lightricks/LTX-2.git /home/lm/LTX-2
uv pip install --no-deps -e /home/lm/LTX-2/packages/ltx-core
uv pip install --no-deps -e /home/lm/LTX-2/packages/ltx-pipelines
```
Installing editable means the XPU patches you apply to the checkout take effect
immediately.

### 5. Apply the three XPU patches
Edit the three files described in [XPU patches](#xpu-patches-applied-to-the-ltx-2-repo).
(They are already applied on this machine.)

### 6. Download model artifacts
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

```bash
source /home/lm/ltx23-env/bin/activate
cd /home/lm/ltx23-run
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run_t2v_xpu.py
```

- `HF_HUB_OFFLINE=1` — prevents any network call mid-run (all weights are local).
- `TOKENIZERS_PARALLELISM=false` — silences a tokenizer warning.

Output: `/home/lm/ltx23-run/output.mp4`.

> **Tip — surviving shell timeouts.** If you launch from an environment that
> kills the command's process group on timeout (e.g. some agent shells), detach
> with `setsid` so the run survives:
> ```bash
> setsid python -u run_t2v_xpu.py > run.log 2>&1 < /dev/null & disown
> tail -f run.log
> ```

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
TDEV = torch.device("xpu", 0)   # transformer
CDEV = torch.device("xpu", 1)   # VAE / decoders
GDEV = torch.device("cpu")      # Gemma text encoder
```

---

## Expected output & timings

For the default config (stage1 256×448, stage2 512×896, 41 frames @ 24 fps):

| Stage | Time |
|---|---|
| Build fp8-cast policy | <0.1 s |
| Prompt encode (Gemma 12B, CPU, 64 threads) | ~17 s |
| Stage 1 transformer build + load | ~12 s |
| Stage 1 denoise (8 steps) | ~8 s (~1.05 s/it) |
| Spatial upsample 2× | ~4 s |
| Stage 2 transformer build + load | ~12 s |
| Stage 2 denoise (3 steps) | ~6 s (~2.1 s/it) |
| Video + audio decode | ~10 s |
| Mux to MP4 | ~3 s |
| **Total** | **~75 s** |

Output file (`output.mp4`):
- Video: 896×512, 24 fps, 41 frames, 1.71 s
- Audio: 48 kHz, stereo, 1.71 s (the joint model generates synchronized audio)
- ~350 KB

---

## Troubleshooting

**`RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same` (in `conv1d`)**
→ The vocoder fp32 patch (#3) is missing or not applied. Re-apply it. The
editable install means editing the file is enough; no reinstall needed.

**Stage denoising is extremely slow, GPU power ~34 W, ~2 CPU cores busy**
→ You're using `fp8-scaled-mm` instead of `fp8-cast`. In `run_t2v_xpu.py` the
policy must be `from ltx_core.quantization.fp8_cast import build_policy as
fp8_cast_policy`. See [the fp8 question](#the-fp8-question-fp8-cast-vs-fp8-scaled-mm).

**`XPU out of memory` during Gemma forward on xpu:1**
→ Gemma 12B bf16 (~24 GB) doesn't fit one B60. Keep `GDEV = torch.device("cpu")`
for the `PromptEncoder`. (If you have a GPU with ≥32 GB, you can move it there.)

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
    ├── run_t2v_xpu.py                  # the driver script
    ├── download.py                     # model download helper
    ├── run.log                         # last run's log
    └── output.mp4                      # generated video (+ audio)
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
- **Single-GPU sharding is not used.** The transformer runs entirely on `xpu:0`;
  `xpu:1` handles the VAEs/decoders. Because models are sequential, this still
  uses both XPUs and keeps each within budget. True transformer sharding across
  XPUs would require the repo's multi-GPU builder and is out of scope here.
- **Prompt enhancement is off** (`enhance_prompt=False`) to avoid Gemma
  *generation* on CPU (slow). Turn it on only if Gemma runs on a GPU.
- **No commits were made** to the LTX-2 checkout; the three patches are applied
  in-place to the editable install.
