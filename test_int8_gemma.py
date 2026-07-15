import torch, time, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("int8_test")

from int8_prompt_encoder import Int8PromptEncoder

CKPT = "/home/lm/ltx23-models/ltx-2.3-22b-distilled-fp8.safetensors"
GEMMA = "/home/lm/ltx23-models/gemma-3-12b-it"
GDEV = torch.device("xpu", 2)
CDEV = torch.device("xpu", 1)

log.info("creating Int8PromptEncoder (gemma on %s, proc on %s)", GDEV, CDEV)
enc = Int8PromptEncoder(
    checkpoint_path=CKPT, gemma_root=GEMMA, dtype=torch.bfloat16,
    gemma_device=GDEV, proc_device=CDEV,
)

prompt = "A cinematic shot of a red panda in a misty bamboo forest, photorealistic, 4k"
log.info("encoding prompt...")
t0 = time.time()
results = enc([prompt], enhance_first_prompt=False)
dt = time.time() - t0
log.info("total encode time: %.2fs", dt)

ctx = results[0]
log.info("video_encoding: %s dtype=%s", tuple(ctx.video_encoding.shape), ctx.video_encoding.dtype)
log.info("audio_encoding: %s", tuple(ctx.audio_encoding.shape) if ctx.audio_encoding is not None else None)
log.info("attention_mask: %s", tuple(ctx.attention_mask.shape))
log.info("video_encoding device: %s", ctx.video_encoding.device)
log.info("SUCCESS")
