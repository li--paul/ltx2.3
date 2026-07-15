"""Pre-encode all 8 prompts with one Gemma instance on CPU.

Saves each embedding as a .pt file; run this once before run_multi_xpu.py
to share a single text encoder across all generation workers.
"""

import json
import logging
import os
import sys
import tempfile
import time

import torch

from ltx_core.quantization.fp8_cast import build_policy as fp8_cast_policy
from ltx_pipelines.utils.blocks import PromptEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("encode_prompts")

DISTILLED_CKPT = "/home/lm/paul/ltx23-models/ltx-2.3-22b-distilled-fp8.safetensors"
GEMMA_ROOT = "/home/lm/paul/ltx23-models/gemma-3-12b-it"


def main() -> None:
    torch.set_num_threads(64)

    # Read prompts: from LTX_PROMPTS_FILE env, or stdin JSON, or default
    prompts_json = os.environ.get("LTX_PROMPTS_FILE", "")
    if prompts_json:
        with open(prompts_json) as f:
            prompts = json.load(f)
    elif not sys.stdin.isatty():
        prompts = json.load(sys.stdin)
    else:
        prompts = [
            "A cinematic shot of a red panda sitting on a mossy branch in a misty bamboo forest, "
            "gentle morning light, soft bokeh, the panda turns its head and chews a bamboo leaf, "
            "photorealistic, 4k, shallow depth of field.",
            "A majestic eagle soaring over a deep canyon at golden hour, warm sunlight "
            "illuminating its wings, dramatic clouds, slow camera pan, "
            "photorealistic, cinematic lighting, 8k.",
            "An underwater scene with a sea turtle swimming through a coral reef, "
            "sunbeams piercing through the water surface, colorful fish, "
            "photorealistic, volumetric lighting, 4k.",
            "A cyberpunk city at night with neon signs reflecting on wet streets, "
            "a lone figure walking under an umbrella, flying cars in the distance, "
            "cinematic, blade runner aesthetic, 8k, anamorphic lens.",
            "A serene mountain lake at sunrise with mist rising from the water, "
            "pine trees reflected in the calm surface, a wooden dock extending into the lake, "
            "photorealistic, warm golden light, hyper-realistic.",
            "A macro shot of a dragonfly perched on a dewy leaf, morning light, "
            "translucent wings with intricate vein patterns, shallow depth of field, "
            "photorealistic, 4k, ultradetailed.",
            "A medieval castle on a stormy cliff edge, lightning flashing behind it, "
            "waves crashing against the rocks, dramatic clouds, dark moody atmosphere, "
            "cinematic, epic scale, photorealistic.",
            "A futuristic greenhouse on Mars under a transparent dome, "
            "lush exotic plants, Earth visible in the twilight sky, "
            "soft artificial lighting, photorealistic, sci-fi aesthetic, 8k.",
        ]

    log.info("building fp8-cast quantization policy...")
    _ = fp8_cast_policy(DISTILLED_CKPT)

    log.info("building PromptEncoder on CPU...")
    prompt_encoder = PromptEncoder(
        checkpoint_path=DISTILLED_CKPT,
        gemma_root=GEMMA_ROOT,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    out_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="ltx_embeddings_")
    os.makedirs(out_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        log.info("encoding prompt %d/%d: %s", i + 1, len(prompts), prompt[:70])
        (ctx_p,) = prompt_encoder([prompt], enhance_first_prompt=False, enhance_prompt_image=None)
        data = {
            "video_encoding": ctx_p.video_encoding.cpu(),
            "audio_encoding": ctx_p.audio_encoding.cpu(),
        }
        path = os.path.join(out_dir, f"embeddings_{i}.pt")
        torch.save(data, path)
        log.info("  done in %.1f s -> %s", time.perf_counter() - t0, path)

    del prompt_encoder
    log.info("all done — %d files in %s", len(prompts), out_dir)
    print(out_dir)  # print path so launcher can capture it


if __name__ == "__main__":
    main()
