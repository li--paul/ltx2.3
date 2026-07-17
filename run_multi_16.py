"""Spawn 16 parallel LTX-2.3 video generations on xpu:0..31 (pairs).

Layout: worker i uses (xpu:i, xpu:i+16) for transformer/VAE.
Pre-encodes all prompts using encode_prompts.py (one Gemma on CPU),
then spawns concurrent subprocesses that load pre-computed embeddings.

Usage:
  python run_multi_16.py --prompts-file prompts.json --job-dir /path/to/out
  python run_multi_16.py                           # hardcoded defaults
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("multi_16_xpu")

LTX23_RUN = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get("LTX_PYTHON", "/home/lm/paul/ltx23-env/bin/python")
SCRIPT = os.path.join(LTX23_RUN, "run_t2v_xpu_perf.py")
ENCODE_SCRIPT = os.path.join(LTX23_RUN, "encode_prompts.py")
N = 16

DEFAULT_PROMPTS = [
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
    "A steaming cup of coffee on a wooden table, morning sunlight streaming through a window, "
    "cinematic, warm tones, photorealistic, 4k.",
    "A bioluminescent forest at night with glowing mushrooms and fireflies, "
    "ethereal, fantasy, cinematic lighting, 8k.",
    "A vintage steam locomotive crossing a wooden bridge over a river in autumn, "
    "fall foliage, golden hour, photorealistic, 4k.",
    "A close-up of a hummingbird hovering near a red flower, rapid wing motion, "
    "sun flare, shallow depth of field, photorealistic, 8k.",
    "A serene Japanese garden with a koi pond and cherry blossoms falling, "
    "spring, soft sunlight, meditation, photorealistic, 4k.",
    "A vast library with towering shelves of ancient books, dust motes dancing in "
    "sunbeams, mysterious atmosphere, cinematic, 8k.",
    "A lightning storm over a savanna at twilight, dramatic clouds, "
    "epic scale, cinematic, photorealistic, 4k.",
    "A cozy cabin interior with a fireplace, snow falling outside the window, "
    "warm lighting, hygge, photorealistic, 8k.",
]


def encode_prompts_subprocess(prompts_file: str) -> str:
    log.info("=" * 60)
    log.info("Encoding %d prompts via encode_prompts.py (shared Gemma)", N)
    log.info("=" * 60)
    env = os.environ.copy()
    env.update({
        "LTX_PROMPTS_FILE": prompts_file,
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    result = subprocess.run(
        [PYTHON, "-u", ENCODE_SCRIPT],
        capture_output=True, text=True, timeout=600, env=env, cwd=LTX23_RUN,
    )
    for line in (result.stdout or "").splitlines():
        log.info("  [encode] %s", line)
    for line in (result.stderr or "").splitlines():
        log.info("  [encode] %s", line)
    if result.returncode != 0:
        raise RuntimeError(
            f"encode_prompts.py failed (rc={result.returncode}): {(result.stderr or '')[-500:]}"
        )
    out_dir = (result.stdout or "").strip().splitlines()[-1]
    if not out_dir or not os.path.isdir(out_dir):
        raise RuntimeError(f"encode_prompts.py did not print a valid output dir: {out_dir!r}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="16-video LTX-2.3 generation")
    parser.add_argument("--prompts-file", help="JSON file with 16 prompts")
    parser.add_argument("--job-dir", help="Output directory")
    args = parser.parse_args()

    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
    elif os.environ.get("LTX_PROMPTS_FILE"):
        with open(os.environ["LTX_PROMPTS_FILE"]) as f:
            prompts = json.load(f)
    else:
        prompts = DEFAULT_PROMPTS

    if len(prompts) != N:
        log.warning("Expected %d prompts, got %d; proceeding anyway", N, len(prompts))

    if args.job_dir:
        output_dir = args.job_dir
    elif os.environ.get("LTX_MULTI_OUTPUT_DIR"):
        output_dir = os.environ["LTX_MULTI_OUTPUT_DIR"]
    else:
        output_dir = os.path.join(LTX23_RUN, "multi_16_output")
        log.warning("No --job-dir specified; using %s", output_dir)

    os.makedirs(output_dir, exist_ok=True)
    start_delay = int(os.environ.get("LTX_SPAWN_DELAY", "10"))

    prompts_file = os.path.join(output_dir, "prompts.json")
    with open(prompts_file, "w") as f:
        json.dump(prompts, f)

    # Step 1: encode all prompts
    log.info("Encoding %d prompts with shared Gemma", len(prompts))
    embeddings_dir = encode_prompts_subprocess(prompts_file)

    # Step 2: spawn generation jobs
    log.info("=" * 60)
    log.info("Spawning %d parallel generation jobs on xpu:0..%d", len(prompts), len(prompts) * 2 - 1)
    log.info("  Layout: worker i -> (xpu:i, xpu:i+%d)", len(prompts))
    log.info("=" * 60)

    processes = []
    for i in range(len(prompts)):
        tdev = i
        cdev = i + len(prompts)
        output_path = os.path.join(output_dir, f"video_{i}.mp4")
        log_path = os.path.join(output_dir, f"video_{i}.log")

        env = os.environ.copy()
        env.update({
            "LTX_TDEV": str(tdev),
            "LTX_CDEV": str(cdev),
            "LTX_PROMPT": prompts[i],
            "LTX_OUTPUT_PATH": output_path,
            "LTX_EMBEDDINGS_PATH": os.path.join(embeddings_dir, f"embeddings_{i}.pt"),
            "LTX_GEMMA_DEVICE": "cpu",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        })

        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            [PYTHON, "-u", SCRIPT],
            cwd=LTX23_RUN,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        if i < len(prompts) - 1 and start_delay > 0:
            log.info("  staggering next worker by %d s", start_delay)
            time.sleep(start_delay)
        processes.append({
            "idx": i, "pid": proc.pid, "proc": proc,
            "log_file": log_file, "log_path": log_path,
            "output_path": output_path, "tdev": tdev, "cdev": cdev,
        })
        log.info("video %d/%d  pid=%d  xpu:(%d,%d)  %s",
                 i + 1, len(prompts), proc.pid, tdev, cdev, output_path)

    log.info("=" * 60)
    log.info("All spawned, waiting for completion...")
    log.info("=" * 60)

    results = []
    for info in processes:
        info["proc"].wait()
        info["log_file"].close()
        rc = info["proc"].returncode
        exists = os.path.isfile(info["output_path"])
        size = os.path.getsize(info["output_path"]) if exists else 0
        status = "OK" if rc == 0 and exists and size > 0 else "FAIL"
        results.append({
            "idx": info["idx"], "rc": rc, "exists": exists, "size": size,
            "status": status, "output_path": info["output_path"],
        })
        log.info("video %d/%d done  pid=%d  rc=%d  %s  %s  (%s)",
                 info["idx"] + 1, len(prompts), info["pid"], rc,
                 "OK" if status == "OK" else "FAIL",
                 info["output_path"],
                 f"{size / 1024**2:.1f} MB" if exists else "no file")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    ok_count = sum(1 for r in results if r["status"] == "OK")
    log.info("=" * 60)
    log.info("%d / %d succeeded", ok_count, len(prompts))
    log.info("=" * 60)

    sys.exit(0 if ok_count == len(prompts) else 1)


if __name__ == "__main__":
    main()
