"""LTX-2.3 text-to-video on the first two Intel Arc Pro B60 XPUs.

Layout:
  xpu:0 -> fp8 distilled transformer (DiffusionStage, ~22 GB weights)
  xpu:1 -> video VAE / spatial upsampler / video decoder / audio VAE+vocoder
  cpu   -> Gemma-3-12B text encoder + embeddings processor

Gemma-3-12B in bf16 is ~24 GB, which does not fit a single 24 GB B60, so the
text encoder runs on CPU (one forward pass, no generation; 2 TB RAM + 128 cores).
Models are built and freed sequentially (gpu_model context), so each device only
ever holds one model at a time -> peak XPU VRAM ~= the fp8 transformer (~22 GB),
which fits a 24 GB B60. Tensors are moved across devices at stage boundaries.
For text-to-video (no input images) the image conditioner produces no
conditioning latents, so the only cross-device moves are the prompt context and
the video/audio latents between stages.
"""
import logging
import os
import time

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization.fp8_cast import build_policy as fp8_cast_policy
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import ModalitySpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ltx23")

# --- paths ---
DISTILLED_CKPT = "/home/lm/ltx23-models/ltx-2.3-22b-distilled-fp8.safetensors"
UPSCALER_CKPT = "/home/lm/ltx23-models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT = "/home/lm/ltx23-models/gemma-3-12b-it"
OUTPUT_PATH = "/home/lm/ltx23-run/output_1024.mp4"

# --- generation params (two-stage -> resolution divisible by 64) ---
PROMPT = (
    "A cinematic shot of a red panda sitting on a mossy branch in a misty bamboo forest, "
    "gentle morning light, soft bokeh, the panda turns its head and chews a bamboo leaf, "
    "photorealistic, 4k, shallow depth of field."
)
SEED = 42
STAGE1_H, STAGE1_W = 512, 512  # stage 2 -> 1024 x 1024
NUM_FRAMES = 121  # 8*15 + 1  -> ~5.0 s @ 24 fps
FRAME_RATE = 24.0

TDEV = torch.device("xpu", 0)  # transformer
CDEV = torch.device("xpu", 1)  # vae / decoders
GDEV = torch.device("cpu")  # Gemma text encoder (does not fit a single 24GB B60 in bf16)


def _mem(tag: str, dev: torch.device) -> None:
    if dev.type != "xpu":
        return
    try:
        used = torch.xpu.memory_allocated(dev) / 1024**3
        reserved = torch.xpu.memory_reserved(dev) / 1024**3
        log.info("[%s] xpu:%s allocated=%.2fGB reserved=%.2fGB", tag, dev.index, used, reserved)
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] mem query failed: %s", tag, e)


# ---- performance timing ----
_stage_timings: dict[str, float] = {}
_peak_mem: dict[int, float] = {}


def _track_peak(dev: torch.device) -> None:
    if dev.type == "xpu":
        idx = dev.index
        cur = torch.xpu.memory_allocated(dev) / 1024**3
        if cur > _peak_mem.get(idx, 0.0):
            _peak_mem[idx] = cur


class _Timer:
    def __init__(self, name: str, dev: torch.device | None = None):
        self.name = name
        self.dev = dev

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        dt = time.perf_counter() - self.t0
        _stage_timings[self.name] = _stage_timings.get(self.name, 0.0) + dt
        if self.dev is not None:
            _track_peak(self.dev)
        try:
            if self.dev is not None and self.dev.type == "xpu":
                torch.xpu.synchronize(self.dev)
        except Exception:
            pass
        log.info("⏱  %-22s %6.2f s", self.name, dt)
        return False


@torch.inference_mode()
def main() -> None:
    height, width = STAGE1_H * 2, STAGE1_W * 2
    assert_resolution(height=height, width=width, is_two_stage=True)
    dtype = torch.bfloat16
    torch.set_num_threads(64)  # CPU Gemma forward uses many cores

    log.info("devices: transformer=%s  vae/decoders=%s  text-encoder=%s", TDEV, CDEV, GDEV)
    log.info("target: %dx%d, %d frames @ %.0ffps (stage1 %dx%d)", width, height, NUM_FRAMES, FRAME_RATE, STAGE1_W, STAGE1_H)

    log.info("building fp8-cast quantization policy from %s", DISTILLED_CKPT)
    quantization = fp8_cast_policy(DISTILLED_CKPT)

    log.info("constructing pipeline blocks")
    prompt_encoder = PromptEncoder(
        checkpoint_path=DISTILLED_CKPT, gemma_root=GEMMA_ROOT, dtype=dtype, device=GDEV,
    )
    stage = DiffusionStage(
        checkpoint_path=DISTILLED_CKPT, dtype=dtype, device=TDEV,
        loras=(), quantization=quantization,
    )
    upsampler = VideoUpsampler(
        checkpoint_path=DISTILLED_CKPT, upsampler_path=UPSCALER_CKPT, dtype=dtype, device=CDEV,
    )
    video_decoder = VideoDecoder(checkpoint_path=DISTILLED_CKPT, dtype=dtype, device=CDEV)
    audio_decoder = AudioDecoder(checkpoint_path=DISTILLED_CKPT, dtype=dtype, device=CDEV)

    # noiser/generator lives on the transformer device (latents are created there)
    generator = torch.Generator(device=TDEV).manual_seed(SEED)
    noiser = GaussianNoiser(generator=generator)
    decode_generator = torch.Generator(device=CDEV).manual_seed(SEED)

    # --- prompt encoding (cpu) ---
    with _Timer("prompt-encode (cpu)", CDEV):
        log.info("encoding prompt on %s", GDEV)
        (ctx_p,) = prompt_encoder([PROMPT], enhance_first_prompt=False, enhance_prompt_image=None)
    _mem("after prompt-encode", CDEV)
    # move context to transformer device
    video_context = ctx_p.video_encoding.to(TDEV)
    audio_context = ctx_p.audio_encoding.to(TDEV)

    stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=TDEV)
    stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=TDEV)
    s1_w, s1_h = STAGE1_W, STAGE1_H

    # --- Stage 1: low-res denoise on xpu:0 ---
    log.info("stage 1: %dx%d %d frames on %s", s1_w, s1_h, NUM_FRAMES, TDEV)
    with _Timer("stage-1 denoise (8 steps)", TDEV):
        video_state, audio_state = stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas, noiser=noiser,
            width=s1_w, height=s1_h, frames=NUM_FRAMES, fps=FRAME_RATE,
            video=ModalitySpec(context=video_context, conditionings=[]),
            audio=ModalitySpec(context=audio_context),
        )
    _mem("after stage1", TDEV)

    # --- spatial upsample 2x on xpu:1 ---
    with _Timer("spatial-upsample 2x", CDEV):
        log.info("spatial upsample 2x on %s", CDEV)
        upscaled_video_latent = upsampler(video_state.latent[:1].to(CDEV))
        upscaled_video_latent = upscaled_video_latent.to(TDEV)

    # --- Stage 2: high-res refine on xpu:0 ---
    log.info("stage 2: %dx%d %d frames on %s", width, height, NUM_FRAMES, TDEV)
    with _Timer("stage-2 denoise (3 steps)", TDEV):
        video_state, audio_state = stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas, noiser=noiser,
            width=width, height=height, frames=NUM_FRAMES, fps=FRAME_RATE,
            video=ModalitySpec(
                context=video_context, conditionings=[],
                noise_scale=stage_2_sigmas[0].item(), initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context, noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent.to(TDEV),
            ),
        )
    _mem("after stage2", TDEV)

    # --- decode video + audio on xpu:1 ---
    with _Timer("video+audio decode", CDEV):
        log.info("decoding video + audio on %s", CDEV)
        tiling_config = TilingConfig.default()
        decoded_video = video_decoder(video_state.latent.to(CDEV), tiling_config, decode_generator)
        decoded_audio = audio_decoder(audio_state.latent.to(CDEV))
    _mem("after decode", CDEV)

    video_chunks_number = get_video_chunks_number(NUM_FRAMES, tiling_config)
    with _Timer("mux to mp4"):
        log.info("encoding to %s (chunks=%d)", OUTPUT_PATH, video_chunks_number)
        encode_video(
            video=decoded_video, fps=FRAME_RATE, audio=decoded_audio,
            output_path=OUTPUT_PATH, video_chunks_number=video_chunks_number,
        )
    log.info("DONE -> %s", OUTPUT_PATH)

    # ---- performance summary ----
    total = sum(_stage_timings.values())
    log.info("=" * 60)
    log.info("PERFORMANCE SUMMARY  (%dx%d, %d frames @ %.0ffps)", width, height, NUM_FRAMES, FRAME_RATE)
    log.info("-" * 60)
    for name, dt in _stage_timings.items():
        log.info("  %-24s %7.2f s  %5.1f%%", name, dt, 100.0 * dt / total if total else 0)
    log.info("  %-24s %7.2f s", "TOTAL", total)
    n_latent_frames = (NUM_FRAMES - 1) // 8 + 1
    log.info("  latent frames=%d  video tokens(stage2)~%d", n_latent_frames, n_latent_frames * (width // 32) * (height // 32))
    for idx in sorted(_peak_mem):
        log.info("  peak xpu:%d allocated = %.2f GB", idx, _peak_mem[idx])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
