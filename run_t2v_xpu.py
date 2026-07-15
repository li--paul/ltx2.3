"""LTX-2.3 text-to-video on Intel Arc Pro B60 XPUs.

Layout (NUMA-aware — two groups, no cross-group traffic):
  Group A: xpu:0 -> fp8 distilled transformer (~18 GB)
           xpu:1 -> video VAE / spatial upsampler / decoders
  Group B: xpu:2 + xpu:3 -> Gemma-3-12B text encoder (sharded via accelerate, ~26 GB)

Gemma-3-12B in bf16 is ~26 GB — too large for a single 24 GB B60. It is sharded
across xpu:2+xpu:3 using accelerate's device_map dispatch (model parallelism),
running the forward pass in ~0.8 s instead of ~17-22 s on CPU.
Models are built and freed sequentially (gpu_model context), so each device only
ever holds one model at a time. Tensors are moved across devices at stage
boundaries. For text-to-video (no input images) the image conditioner produces
no conditioning latents, so the only cross-device moves are the prompt context
and the video/audio latents between stages.

Set LTX_GEMMA_DEVICE=cpu to fall back to CPU for the text encoder.
"""
import logging
import os

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

# Optional DP prompt encoder (Gemma sharded across xpu:2+xpu:3)
try:
    from dp_prompt_encoder import DPPromptEncoder
except ImportError:
    DPPromptEncoder = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ltx23")

# --- paths ---
DISTILLED_CKPT = "/home/lm/paul/ltx23-models/ltx-2.3-22b-distilled-fp8.safetensors"
UPSCALER_CKPT = "/home/lm/paul/ltx23-models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT = "/home/lm/paul/ltx23-models/gemma-3-12b-it"
OUTPUT_PATH = os.environ.get("LTX_OUTPUT_PATH", "/home/lm/paul/ltx23-run/output.mp4")

# --- generation params (two-stage -> resolution divisible by 64) ---
_DEFAULT_PROMPT = (
    "A cinematic shot of a red panda sitting on a mossy branch in a misty bamboo forest, "
    "gentle morning light, soft bokeh, the panda turns its head and chews a bamboo leaf, "
    "photorealistic, 4k, shallow depth of field."
)
PROMPT = os.environ.get("LTX_PROMPT", _DEFAULT_PROMPT)
SEED = 42
_TARGET_W = int(os.environ.get("LTX_WIDTH", "1024"))
_TARGET_H = int(os.environ.get("LTX_HEIGHT", "1024"))
STAGE1_W, STAGE1_H = _TARGET_W // 2, _TARGET_H // 2  # stage 2 -> target
NUM_FRAMES = int(os.environ.get("LTX_FRAMES", "73"))  # must be 8k + 1
FRAME_RATE = 24.0

TDEV = torch.device("xpu", 0)  # transformer
CDEV = torch.device("xpu", 1)  # vae / decoders
GDEV = torch.device("cpu")  # fallback Gemma device
GEMMA_DP_DEVICES = (2, 3)  # NUMA group B for DP-sharded Gemma
USE_DP_GEMMA = DPPromptEncoder is not None and os.environ.get("LTX_GEMMA_DEVICE", "xpu") != "cpu"


def _mem(tag: str, dev: torch.device) -> None:
    if dev.type != "xpu":
        return
    try:
        used = torch.xpu.memory_allocated(dev) / 1024**3
        reserved = torch.xpu.memory_reserved(dev) / 1024**3
        log.info("[%s] xpu:%s allocated=%.2fGB reserved=%.2fGB", tag, dev.index, used, reserved)
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] mem query failed: %s", tag, e)


@torch.inference_mode()
def main() -> None:
    height, width = STAGE1_H * 2, STAGE1_W * 2
    assert_resolution(height=height, width=width, is_two_stage=True)
    dtype = torch.bfloat16
    torch.set_num_threads(64)  # CPU Gemma forward uses many cores

    if USE_DP_GEMMA:
        log.info("devices: transformer=%s  vae/decoders=%s  text-encoder=xpu:%s (DP sharded)",
                 TDEV, CDEV, ",".join(str(d) for d in GEMMA_DP_DEVICES))
    else:
        log.info("devices: transformer=%s  vae/decoders=%s  text-encoder=%s", TDEV, CDEV, GDEV)
    log.info("target: %dx%d, %d frames @ %.0ffps (stage1 %dx%d)", width, height, NUM_FRAMES, FRAME_RATE, STAGE1_W, STAGE1_H)

    log.info("building fp8-cast quantization policy from %s", DISTILLED_CKPT)
    quantization = fp8_cast_policy(DISTILLED_CKPT)

    log.info("constructing pipeline blocks")
    if USE_DP_GEMMA:
        prompt_encoder = DPPromptEncoder(
            checkpoint_path=DISTILLED_CKPT, gemma_root=GEMMA_ROOT, dtype=dtype,
            gemma_devices=GEMMA_DP_DEVICES, proc_device=CDEV,
        )
    else:
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

    # --- prompt encoding ---
    if USE_DP_GEMMA:
        log.info("encoding prompt on xpu:%s (DP sharded)", ",".join(str(d) for d in GEMMA_DP_DEVICES))
    else:
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
    video_state, audio_state = stage(
        denoiser=SimpleDenoiser(video_context, audio_context),
        sigmas=stage_1_sigmas, noiser=noiser,
        width=s1_w, height=s1_h, frames=NUM_FRAMES, fps=FRAME_RATE,
        video=ModalitySpec(context=video_context, conditionings=[]),
        audio=ModalitySpec(context=audio_context),
    )
    _mem("after stage1", TDEV)

    # --- spatial upsample 2x on xpu:1 ---
    log.info("spatial upsample 2x on %s", CDEV)
    upscaled_video_latent = upsampler(video_state.latent[:1].to(CDEV))
    upscaled_video_latent = upscaled_video_latent.to(TDEV)

    # --- Stage 2: high-res refine on xpu:0 ---
    log.info("stage 2: %dx%d %d frames on %s", width, height, NUM_FRAMES, TDEV)
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
    log.info("decoding video + audio on %s", CDEV)
    tiling_config = TilingConfig.default()
    decoded_video = video_decoder(video_state.latent.to(CDEV), tiling_config, decode_generator)
    decoded_audio = audio_decoder(audio_state.latent.to(CDEV))
    _mem("after decode", CDEV)

    video_chunks_number = get_video_chunks_number(NUM_FRAMES, tiling_config)
    log.info("encoding to %s (chunks=%d)", OUTPUT_PATH, video_chunks_number)
    encode_video(
        video=decoded_video, fps=FRAME_RATE, audio=decoded_audio,
        output_path=OUTPUT_PATH, video_chunks_number=video_chunks_number,
    )
    log.info("DONE -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
