#!/home/lm/paul/ltx23-env/bin/python
"""Benchmark: 10 consecutive 16-video multi-jobs with different prompts."""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("bench")

BASE = os.environ.get("LTX_BENCH_URL", "http://127.0.0.1:8001")
TOKEN = os.environ.get("LTX_BENCH_TOKEN", "111")
OUTPUT = Path(os.environ.get("LTX_BENCH_OUTPUT", "/home/lm/paul/bench_results.json"))

HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

PROMPT_POOL = [
    "A red panda on a mossy branch in a misty bamboo forest, photorealistic, 4k.",
    "A majestic eagle soaring over a deep canyon at golden hour, cinematic, 8k.",
    "An underwater sea turtle swimming through a coral reef, volumetric lighting.",
    "A cyberpunk city at night with neon signs reflecting on wet streets, blade runner.",
    "A serene mountain lake at sunrise with mist rising from the water, photorealistic.",
    "A macro shot of a dragonfly perched on a dewy leaf, morning light, shallow DOF.",
    "A medieval castle on a stormy cliff edge, lightning flashing, dramatic clouds.",
    "A futuristic greenhouse on Mars under a transparent dome, lush plants, sci-fi.",
    "A steaming cup of coffee on a wooden table at sunrise, cinematic, warm tones.",
    "A wizard casting a spell in an ancient library, floating books, mystical blue light.",
    "A neon-lit sushi bar in Tokyo at midnight, rain on window, reflections, cyberpunk.",
    "A polar bear on a melting ice floe at sunset, dramatic sky, photorealistic.",
    "A race car speeding through a futuristic city tunnel, motion blur, neon reflections.",
    "A ballerina performing on an empty stage, spotlight, dust particles, emotional.",
    "A supercell thunderstorm over a prairie at twilight, lightning, rotating clouds.",
    "An astronaut floating in space overlooking Earth, stars, cosmic rays, 8k.",
    "A cat walking along a narrow rooftop at dusk, city skyline background, cinematic.",
    "A waterfall in a tropical jungle, sunlight through canopy, rainbow, photorealistic.",
    "A vintage Mustang driving through Monument Valley at sunset, warm golden light.",
    "A chef slicing fresh sushi with precision, close-up, shallow DOF, 4k.",
    "A lone wolf howling at the full moon on a snowy mountain peak, epic, cinematic.",
    "A hot air balloon floating over Cappadocia at sunrise, fairy chimneys, aerial view.",
    "A vintage record player spinning a vinyl in a dimly lit room, warm amber light.",
    "A stormtrooper walking through a burning forest on an alien planet, dramatic.",
    "A coral reef with clownfish swimming through sea anemones, sunbeams, 4k.",
    "A lone guitarist playing on a rooftop at sunset, city silhouette, emotional.",
    "A fighter jet breaking the sound barrier with vapor cone, clear sky, action.",
    "A bonsai tree on a wooden stand in a zen garden, morning dew, photorealistic.",
    "A UFO hovering over a desert highway at night, beam of light, mysterious.",
    "A cowboy riding a horse through a dust storm in the Wild West, dramatic lighting.",
    "A submarine descending into the Mariana Trench, bioluminescent creatures, 4k.",
    "A street musician playing violin in a subway station, commuters passing by.",
    "A samurai standing in a bamboo forest after rain, sword drawn, mist, cinematic.",
    "A hummingbird drinking nectar from a bright red flower, slow motion, 4k.",
    "A lava lamp with blobs of wax floating in purple liquid, retro, close-up.",
    "A pirate ship navigating through a stormy sea at night, lightning, huge waves.",
    "A glassblower shaping molten glass into a vase, workshop, orange glow, 4k.",
    "A satellite orbiting Earth with solar panels reflecting sunlight, stars, cinematic.",
    "A deer drinking from a crystal-clear mountain stream at dawn, mist, photorealistic.",
    "A Formula 1 car taking a sharp turn at high speed, tire smoke, action, 8k.",
    "A pianist playing a grand piano in an empty concert hall, spotlight, dramatic.",
    "A toucan eating a tropical fruit in the Amazon rainforest, lush green, 4k.",
    "A lighthouse beam cutting through thick fog on a rocky coast at night.",
    "A skateboarder doing a trick in an empty concrete pool at golden hour.",
    "A volcano erupting at night with lava flowing down the slope, red sky, epic.",
    "A fox walking through a snow-covered forest at twilight, soft snow, cinematic.",
    "A space elevator rising through clouds into the stratosphere, futuristic, 8k.",
    "A monk meditating in a temple courtyard with cherry blossoms falling, serene.",
    "A bullet train passing through a neon-lit city at night, motion blur, cyberpunk.",
    "A great white shark breaching the surface at sunset, dramatic, photorealistic.",
    "A night market in Bangkok with street food stalls, steam, lanterns, vibrant.",
    "A robotic hand assembling a circuit board in a high-tech factory, precision.",
    "A herd of wild horses galloping across a prairie at sunrise, dust, cinematic.",
    "A lava waterfall inside a volcanic cave, glowing orange, otherworldly, 4k.",
]


def submit_job(prompts: list[str]) -> str:
    r = requests.post(
        f"{BASE}/api/multi-jobs", headers=HEADERS, json={"prompts": prompts}, timeout=30
    )
    r.raise_for_status()
    return r.json()["id"]


def poll_until_done(job_id: str, interval: int = 15) -> dict:
    while True:
        r = requests.get(
            f"{BASE}/api/multi-jobs/{job_id}", headers=HEADERS, timeout=30
        )
        r.raise_for_status()
        job = r.json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(interval)


def main():
    log.info("Benchmark: 10 × 16-video multi-jobs")
    log.info("Server: %s", BASE)
    log.info("Token: %s", TOKEN)
    log.info("Prompt pool: %d prompts", len(PROMPT_POOL))

    results = []
    passed = 0
    failed = 0

    for run in range(10):
        log.info("=" * 60)
        log.info("Run %d/10", run + 1)

        # select 16 prompts (cycle through pool)
        offset = (run * 7) % len(PROMPT_POOL)
        prompts = []
        for i in range(16):
            idx = (offset + i) % len(PROMPT_POOL)
            prompts.append(PROMPT_POOL[idx])

        start = time.monotonic()
        start_iso = datetime.now(timezone.utc).isoformat()

        try:
            job_id = submit_job(prompts)
            log.info("  job_id: %s", job_id[:12])

            job = poll_until_done(job_id)
            elapsed = time.monotonic() - start

            ok = len(job.get("videos", []))
            total = len(prompts)
            success = job["status"] == "succeeded"

            rec = {
                "run": run + 1,
                "job_id": job_id,
                "status": job["status"],
                "prompts": prompts,
                "videos_ok": ok,
                "videos_total": total,
                "elapsed_seconds": round(elapsed, 1),
                "started_at": start_iso,
            }

            if success:
                passed += 1
                log.info(
                    "  ✓  %d/%d videos  %.1f s (%.1f min)",
                    ok, total, elapsed, elapsed / 60,
                )
            else:
                failed += 1
                rec["error"] = job.get("error", "")
                log.info("  ✗  %s  %s", job["status"], rec["error"])

            results.append(rec)

        except Exception as e:
            failed += 1
            elapsed = time.monotonic() - start
            log.error("  ✗  exception: %s", e)
            results.append({
                "run": run + 1,
                "job_id": None,
                "status": "exception",
                "error": str(e),
                "elapsed_seconds": round(elapsed, 1),
                "started_at": start_iso,
                "prompts": prompts,
                "videos_ok": 0,
                "videos_total": len(prompts),
            })

    # report
    log.info("=" * 60)
    log.info("BENCHMARK COMPLETE: %d/10 passed, %d/10 failed", passed, failed)

    times = [r["elapsed_seconds"] for r in results if r["status"] == "succeeded"]
    if times:
        avg = sum(times) / len(times)
        log.info("  Avg time: %.1f s (%.1f min)", avg, avg / 60)
        log.info("  Min time: %.1f s", min(times))
        log.info("  Max time: %.1f s", max(times))

    # per-prompt average
    total_videos = sum(r["videos_ok"] for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)
    if total_videos:
        log.info("  Total videos: %d", total_videos)
        log.info("  Avg per video: %.1f s", total_time / total_videos)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump({
            "summary": {
                "total_runs": 10,
                "passed": passed,
                "failed": failed,
                "avg_seconds": round(avg, 1) if times else None,
                "min_seconds": round(min(times), 1) if times else None,
                "max_seconds": round(max(times), 1) if times else None,
                "total_videos": total_videos,
                "avg_seconds_per_video": round(total_time / total_videos, 1) if total_videos else None,
            },
            "runs": results,
        }, f, indent=2)

    log.info("Results saved to %s", OUTPUT)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
