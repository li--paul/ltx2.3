import logging, sys
from huggingface_hub import hf_hub_download, snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dl")

# 1) fp8 distilled transformer checkpoint
log.info("downloading ltx-2.3-22b-distilled-fp8.safetensors")
p1 = hf_hub_download("Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-distilled-fp8.safetensors", local_dir="/home/lm/ltx23-models")
log.info("ckpt -> %s", p1)

# 2) spatial upsampler x2 v1.1
log.info("downloading spatial upscaler x2-1.1")
p2 = hf_hub_download("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", local_dir="/home/lm/ltx23-models")
log.info("upscaler -> %s", p2)

# 3) gemma-3-12b-it (unsloth mirror, not gated)
log.info("downloading gemma-3-12b-it snapshot")
p3 = snapshot_download("unsloth/gemma-3-12b-it", local_dir="/home/lm/ltx23-models/gemma-3-12b-it",
                       allow_patterns=["*.safetensors","*.json","*.model","*.txt","tokenizer*","preprocessor*"])
log.info("gemma -> %s", p3)
log.info("ALL DOWNLOADS COMPLETE")
