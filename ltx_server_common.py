"""Shared FastAPI application for LTX-2.3 text-to-video via subprocess worker.

Each generation job spawns run_t2v_xpu.py (or run_t2v_xpu_perf.py) as a
subprocess, avoiding OOM from in-process model lifecycle accumulation.
"""

import asyncio
import json
import logging
import os
import queue
import re
import secrets
import sqlite3
import subprocess
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("ltx_server")

LTX23_RUN_DIR = Path("/home/lm/paul/ltx23-run")
LTX23_ENV_PYTHON = "/home/lm/paul/ltx23-env/bin/python"
GENERATION_SCRIPT = str(LTX23_RUN_DIR / "run_t2v_xpu_perf.py")
MULTI_SCRIPT = str(LTX23_RUN_DIR / "run_multi_xpu.py")
MAX_LOG_LINES = 100


@dataclass(frozen=True)
class ModelProfile:
    display_name: str
    default_width: int
    default_height: int
    default_frames: int
    multi_mode: int = 8  # 8 or 16 concurrent videos


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    api_token: str
    queue_size: int
    output_dir: Path
    database_path: Path

    @classmethod
    def from_environment(cls) -> "ServerSettings":
        return cls(
            host=os.environ.get("LTX_HOST", "127.0.0.1"),
            port=int(os.environ.get("LTX_PORT", "8001")),
            api_token=os.environ.get("LTX_API_TOKEN", ""),
            queue_size=int(os.environ.get("LTX_QUEUE_SIZE", "2")),
            output_dir=Path(os.environ.get("LTX_OUTPUT_DIR", "outputs/ltx-server")).resolve(),
            database_path=Path(
                os.environ.get("LTX_DB", "outputs/ltx-server/jobs.sqlite3")
            ).resolve(),
        )


@dataclass(frozen=True)
class ServerApplication:
    app: FastAPI
    settings: ServerSettings
    profile: ModelProfile


def is_loopback(host: str) -> bool:
    return host.lower() in {"127.0.0.1", "localhost", "::1"}


def public_job(job: dict) -> dict:
    result = dict(job)
    result.pop("video_path", None)
    result["video_url"] = f"/api/jobs/{job['id']}/video" if job["status"] == "succeeded" else None
    return result


def generate_seed() -> int:
    return (time.time_ns() ^ secrets.randbits(63) ^ uuid4().int) & (2**63 - 1)


def validate_frames(frames: int) -> None:
    if frames < 1 or (frames - 1) % 8 != 0:
        raise HTTPException(
            status_code=422,
            detail=f"Frames must satisfy frames % 8 == 1 (got {frames}). Valid: 1, 9, 17, 25, 41, 121...",
        )


# ---------------------------------------------------------------------------
# JobStore  (SQLite)
# ---------------------------------------------------------------------------

class JobStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def initialize(self) -> None:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'queued',
                    prompt TEXT NOT NULL,
                    seed INTEGER,
                    width INTEGER,
                    height INTEGER,
                    frames INTEGER,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    video_path TEXT,
                    parameters TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS multi_jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'queued',
                    prompts TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    output_dir TEXT
                )
            """)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()

    def create_multi(self, job_id: str, prompts: list[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO multi_jobs (id, status, prompts, created_at) VALUES (?, ?, ?, ?)",
                (job_id, "queued", json.dumps(prompts), now),
            )
            conn.commit()

    def get_multi(self, job_id: str) -> dict | None:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM multi_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            r = dict(row)
            r["prompts"] = json.loads(r["prompts"])
            return r

    def list_multi(self, limit: int = 20, offset: int = 0) -> list[dict]:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM multi_jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                r["prompts"] = json.loads(r["prompts"])
                results.append(r)
            return results

    def mark_multi_running(self, job_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE multi_jobs SET status = 'running', started_at = ? WHERE id = ?",
                (now, job_id),
            )
            conn.commit()

    def mark_multi_succeeded(self, job_id: str, output_dir: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE multi_jobs SET status = 'succeeded', completed_at = ?, output_dir = ? WHERE id = ?",
                (now, output_dir, job_id),
            )
            conn.commit()

    def mark_multi_failed(self, job_id: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE multi_jobs SET status = 'failed', completed_at = ?, error = ? WHERE id = ?",
                (now, error, job_id),
            )
            conn.commit()

    def create(self, job_id: str, parameters: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """INSERT INTO jobs (id, status, prompt, seed, width, height, frames,
                   created_at, parameters) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    "queued",
                    parameters["prompt"],
                    parameters.get("seed"),
                    parameters.get("width"),
                    parameters.get("height"),
                    parameters.get("frames"),
                    now,
                    json.dumps(parameters),
                ),
            )
            conn.commit()

    def get(self, job_id: str) -> dict | None:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            r = dict(row)
            if r["parameters"]:
                r["parameters"] = json.loads(r["parameters"])
            return r

    def list(self, limit: int = 20, offset: int = 0) -> list[dict]:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                if r["parameters"]:
                    r["parameters"] = json.loads(r["parameters"])
                results.append(r)
            return results

    def mark_running(self, job_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                (now, job_id),
            )
            conn.commit()

    def mark_succeeded(self, job_id: str, video_path: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE jobs SET status = 'succeeded', completed_at = ?, video_path = ? WHERE id = ?",
                (now, video_path, job_id),
            )
            conn.commit()

    def mark_failed(self, job_id: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', completed_at = ?, error = ? WHERE id = ?",
                (now, error, job_id),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# ServerState  (thread-safe shared state, broadcast via SSE)
# ---------------------------------------------------------------------------

class ServerState:
    """Holds all active job state server-side. Updated by worker threads,
    read by the SSE endpoint. Clients are pure viewers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._single_log: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._multi_log: deque[str] = deque(maxlen=MAX_LOG_LINES)

        # active single job
        self.active_single_id: str | None = None
        self.active_single_status: str = "idle"
        self.active_single_prompt: str = ""
        self.active_single_progress: str = ""
        self.active_single_video_url: str | None = None
        self.active_single_error: str = ""

        # active multi job
        self.active_multi_id: str | None = None
        self.active_multi_status: str = "idle"
        self.active_multi_prompts: list[str] = []
        self.active_multi_workers: list[dict] = []  # [{idx, status, last_log}]
        self.active_multi_videos: list[str] = []
        self.active_multi_error: str = ""

        # cached history (refreshed from DB periodically)
        self.single_history: list[dict] = []
        self.multi_history: list[dict] = []

    # -- single job state --

    def single_start(self, job_id: str, prompt: str) -> None:
        with self._lock:
            self.active_single_id = job_id
            self.active_single_status = "running"
            self.active_single_prompt = prompt
            self.active_single_progress = ""
            self.active_single_video_url = None
            self.active_single_error = ""
            self._single_log.clear()

    def single_append_log(self, line: str) -> None:
        with self._lock:
            self._single_log.append(str(line))

    def single_update_progress(self, progress: str) -> None:
        with self._lock:
            self.active_single_progress = progress

    def single_succeeded(self, video_url: str) -> None:
        with self._lock:
            self.active_single_status = "succeeded"
            self.active_single_video_url = video_url
            self.active_single_progress = ""

    def single_failed(self, error: str) -> None:
        with self._lock:
            self.active_single_status = "failed"
            self.active_single_error = error
            self.active_single_progress = ""

    def single_clear(self) -> None:
        with self._lock:
            self.active_single_id = None
            self.active_single_status = "idle"
            self.active_single_prompt = ""
            self.active_single_progress = ""
            self.active_single_video_url = None
            self.active_single_error = ""
            self._single_log.clear()

    # -- multi job state --

    def multi_start(self, job_id: str, prompts: list[str]) -> None:
        with self._lock:
            self.active_multi_id = job_id
            self.active_multi_status = "running"
            self.active_multi_prompts = list(prompts)
            self.active_multi_workers = [
                {"idx": i, "status": "queued", "last_log": ""} for i in range(len(prompts))
            ]
            self.active_multi_videos = []
            self.active_multi_error = ""
            self._multi_log.clear()

    def multi_append_log(self, line: str) -> None:
        with self._lock:
            self._multi_log.append(str(line))

    def multi_update_worker(self, idx: int, status: str, last_log: str = "") -> None:
        with self._lock:
            if 0 <= idx < len(self.active_multi_workers):
                self.active_multi_workers[idx]["status"] = status
                if last_log:
                    self.active_multi_workers[idx]["last_log"] = last_log

    def multi_succeeded(self, videos: list[str]) -> None:
        with self._lock:
            self.active_multi_status = "succeeded"
            self.active_multi_videos = list(videos)

    def multi_failed(self, error: str) -> None:
        with self._lock:
            self.active_multi_status = "failed"
            self.active_multi_error = error

    def multi_clear(self) -> None:
        with self._lock:
            self.active_multi_id = None
            self.active_multi_status = "idle"
            self.active_multi_prompts = []
            self.active_multi_workers = []
            self.active_multi_videos = []
            self.active_multi_error = ""
            self._multi_log.clear()

    # -- snapshot for SSE broadcast --

    def snapshot(self, telemetry: dict | None = None) -> dict:
        with self._lock:
            return {
                "single": {
                    "id": self.active_single_id,
                    "status": self.active_single_status,
                    "prompt": self.active_single_prompt,
                    "progress": self.active_single_progress,
                    "log": list(self._single_log),
                    "video_url": self.active_single_video_url,
                    "error": self.active_single_error,
                },
                "multi": {
                    "id": self.active_multi_id,
                    "status": self.active_multi_status,
                    "prompts": list(self.active_multi_prompts),
                    "log": list(self._multi_log),
                    "workers": list(self.active_multi_workers),
                    "videos": list(self.active_multi_videos),
                    "error": self.active_multi_error,
                },
                "history": {
                    "single": self.single_history,
                    "multi": self.multi_history,
                },
                "telemetry": telemetry or {},
            }


# ---------------------------------------------------------------------------
# LtxWorker  (single background thread, subprocess-based)
# ---------------------------------------------------------------------------

class LtxWorker:
    """Background worker that spawns run_t2v_xpu.py per job via subprocess."""

    def __init__(
        self,
        store: JobStore,
        output_dir: Path,
        state: ServerState,
        queue_size: int = 2,
    ) -> None:
        self._store = store
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._state = state
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready = threading.Event()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def queued_count(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        self._ready.set()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ltx-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

    def submit(self, task: dict) -> None:
        self._queue.put_nowait(task)

    def _run(self) -> None:
        logger.info("Worker thread started")
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue

            job_id = task["id"]
            prompt = task["prompt"]
            seed = task.get("seed", generate_seed())
            width = task.get("width", 1024)
            height = task.get("height", 1024)
            frames = task.get("frames", 121)

            logger.info("Processing job %s: %s", job_id, prompt[:80])
            self._store.mark_running(job_id)
            self._state.single_start(job_id, prompt)

            output_filename = f"{job_id}.mp4"
            output_path = str(self._output_dir / output_filename)

            try:
                self._spawn_generation(prompt, seed, output_path, width, height, frames)
                if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                    self._store.mark_succeeded(job_id, output_path)
                    self._state.single_succeeded(f"/api/jobs/{job_id}/video")
                    logger.info("Job %s succeeded -> %s", job_id, output_path)
                else:
                    raise RuntimeError(f"Output file not created: {output_path}")
            except Exception as e:
                logger.exception("Job %s failed", job_id)
                self._store.mark_failed(job_id, str(e))
                self._state.single_failed(str(e))

    def _spawn_generation(self, prompt: str, seed: int, output_path: str, width: int = 1024, height: int = 1024, frames: int = 121) -> None:
        env = os.environ.copy()
        env.update({
            "LTX_PROMPT": prompt,
            "LTX_OUTPUT_PATH": output_path,
            "LTX_WIDTH": str(width),
            "LTX_HEIGHT": str(height),
            "LTX_FRAMES": str(frames),
            "LTX_GEMMA_DEVICE": "cpu",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        })
        logger.info("Spawning: %s", GENERATION_SCRIPT)
        self._state.single_append_log(f"[spawn] {GENERATION_SCRIPT}")
        self._state.single_append_log(f"[params] {width}x{height}, {frames}fr, seed={seed}")
        proc = subprocess.Popen(
            [LTX23_ENV_PYTHON, "-u", GENERATION_SCRIPT],
            cwd=str(LTX23_RUN_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        deadline = time.monotonic() + 1200
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip("\n\r")
            self._state.single_append_log(line)
            # extract stage progress from tqdm-like output
            m = re.search(r"(\d+)/(\d+)\s+\[(\d+):(\d+)", line)
            if m:
                cur, total = int(m.group(1)), int(m.group(2))
                step_name = "denoising" if total <= 8 else "stage 2"
                self._state.single_update_progress(f"{step_name} step {cur}/{total}")
            if time.monotonic() > deadline:
                proc.kill()
                raise RuntimeError("Generation timed out after 1200s")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Generation failed (exit code {proc.returncode})")
        logger.info("Subprocess completed (exit 0)")


# ---------------------------------------------------------------------------
# MultiLtxWorker  (single background thread, subprocess-based)
# ---------------------------------------------------------------------------

class MultiLtxWorker:
    """Background worker that directly spawns encode_prompts.py + N generation workers."""

    def __init__(self, store: JobStore, output_dir: Path, state: ServerState,
                 max_workers: int = 8) -> None:
        self._store = store
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._state = state
        self._max_workers = max_workers
        self._stagger = 5 if max_workers <= 8 else 10
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=4)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready = threading.Event()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def queued_count(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        self._ready.set()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ltx-multi-worker"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

    def submit(self, task: dict) -> None:
        self._queue.put_nowait(task)

    def _run(self) -> None:
        logger.info("Multi-worker thread started")
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue

            job_id = task["id"]
            prompts = task["prompts"]
            n = len(prompts)
            logger.info("Processing multi-job %s (%d prompts)", job_id, n)
            self._store.mark_multi_running(job_id)
            self._state.multi_start(job_id, prompts)
            self._state.multi_append_log(f"[multi] {n} prompts, job {job_id[:8]}...")

            job_dir = str(self._output_dir / job_id)
            os.makedirs(job_dir, exist_ok=True)

            prompts_file = os.path.join(job_dir, "prompts.json")
            with open(prompts_file, "w") as f:
                json.dump(prompts, f)

            try:
                # Step 1: encode prompts via encode_prompts.py (shared Gemma)
                self._state.multi_append_log("[step 1] Encoding prompts via Gemma (CPU)...")
                logger.info("Step 1/%d: encoding %d prompts via encode_prompts.py", n + 1, n)
                env = os.environ.copy()
                env.update({
                    "LTX_PROMPTS_FILE": prompts_file,
                    "LTX_GEMMA_DEVICE": "cpu",
                    "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                })
                enc_result = subprocess.run(
                    [LTX23_ENV_PYTHON, "-u", str(LTX23_RUN_DIR / "encode_prompts.py")],
                    capture_output=True, text=True, timeout=600,
                    env=env, cwd=str(LTX23_RUN_DIR),
                )
                if enc_result.returncode != 0:
                    raise RuntimeError(
                        f"encode_prompts.py failed: {(enc_result.stderr or '')[-500:]}"
                    )
                embeddings_dir = (enc_result.stdout or "").strip().splitlines()[-1]
                logger.info("Embeddings dir: %s", embeddings_dir)
                self._state.multi_append_log(f"[step 1] Done, embeddings in {embeddings_dir}")

                # Step 2: spawn generation workers with stagger
                self._state.multi_append_log(f"[step 2] Spawning {n} workers...")
                logger.info("Step 2/%d: spawning %d workers (staggered)", n + 1, n)
                processes: list[dict] = []
                for i in range(n):
                    # device assignment:
                    #   max_workers <= 8: pairs (0,1), (2,3), … (14,15)
                    #   max_workers > 8:  pairs (0,16), (1,17), … (15,31)
                    if self._max_workers <= 8:
                        tdev = i * 2
                        cdev = i * 2 + 1
                    else:
                        tdev = i
                        cdev = i + 16
                    output_path = os.path.join(job_dir, f"video_{i}.mp4")
                    log_path = os.path.join(job_dir, f"video_{i}.log")

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
                        [LTX23_ENV_PYTHON, "-u", GENERATION_SCRIPT],
                        cwd=str(LTX23_RUN_DIR),
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                    processes.append({
                        "idx": i, "proc": proc, "log_file": log_file,
                        "output_path": output_path, "log_path": log_path,
                    })
                    self._state.multi_update_worker(i, "spawned", f"pid={proc.pid} xpu:({tdev},{cdev})")
                    self._state.multi_append_log(f"  worker {i+1}/{n}  pid={proc.pid}  xpu:({tdev},{cdev})")
                    logger.info("  worker %d/%d  pid=%d  xpu:(%d,%d)", i + 1, n, proc.pid, tdev, cdev)

                    # stagger between spawns to avoid XPU driver race
                    if i < n - 1:
                        time.sleep(self._stagger)

                # Step 3: wait for all workers with periodic log tailing
                self._state.multi_append_log("[step 3] Waiting for all workers...")
                logger.info("Waiting for %d workers...", n)
                remaining = list(processes)
                results = []
                while remaining:
                    for info in list(remaining):
                        rc = info["proc"].poll()
                        if rc is not None:
                            remaining.remove(info)
                            info["log_file"].close()
                            exists = os.path.isfile(info["output_path"])
                            size = os.path.getsize(info["output_path"]) if exists else 0
                            status = "OK" if rc == 0 and exists and size > 0 else "FAIL"
                            results.append({
                                "idx": info["idx"], "rc": rc, "exists": exists,
                                "size": size, "status": status,
                                "output_path": info["output_path"],
                            })
                            self._state.multi_update_worker(info["idx"], "done" if status == "OK" else "failed")
                            self._state.multi_append_log(
                                f"  worker {info['idx']+1}/{n} done  {'OK' if status == 'OK' else f'rc={rc}'}  "
                                f"{info['output_path']} ({size/1024**2:.1f} MB)" if exists else "no file"
                            )
                            logger.info("  worker %d/%d done  rc=%d  %s  %s  (%s)",
                                        info["idx"] + 1, n, rc,
                                        "OK" if status == "OK" else f"rc={rc}",
                                        info["output_path"],
                                        f"{size / 1024**2:.1f} MB" if exists else "no file")
                    # read tail of each running worker's log
                    if remaining:
                        for info in remaining:
                            try:
                                with open(info["log_path"]) as lf:
                                    lines = lf.read().strip().splitlines()
                                    last = lines[-1] if lines else ""
                                    if last:
                                        self._state.multi_update_worker(info["idx"], "running", last[-120:])
                            except OSError:
                                pass
                        time.sleep(2)

                with open(os.path.join(job_dir, "results.json"), "w") as f:
                    json.dump(results, f, indent=2)

                ok = sum(1 for r in results if r["status"] == "OK")
                if ok == n:
                    self._store.mark_multi_succeeded(job_id, job_dir)
                    videos = [f"/api/multi-jobs/{job_id}/videos/{i}" for i in range(n)]
                    self._state.multi_succeeded(videos)
                    self._state.multi_append_log(f"[done] {ok}/{n} workers succeeded")
                    logger.info("Multi-job %s: %d/%d succeeded", job_id, ok, n)
                else:
                    raise RuntimeError(f"{ok}/{n} workers succeeded")
            except Exception as e:
                logger.exception("Multi-job %s failed", job_id)
                self._store.mark_multi_failed(job_id, str(e))
                self._state.multi_failed(str(e))
                self._state.multi_append_log(f"[error] {str(e)[:200]}")


# Multi-job public response helpers

def public_multi_job(job: dict, output_dir: str | None = None) -> dict:
    result = dict(job)
    result.pop("output_dir", None)
    if job["status"] == "succeeded":
        result["videos"] = [
            f"/api/multi-jobs/{job['id']}/videos/{i}" for i in range(len(job["prompts"]))
        ]
    else:
        result["videos"] = []
    return result


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>__MODEL_DISPLAY_NAME__</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #0c111b; color: #e8edf5; }
    main { max-width: 980px; margin: 0 auto; padding: 28px 18px 60px; }
    h1 { margin: 0 0 8px; } .muted { color: #94a3b8; }
    .card { background: #151d2b; border: 1px solid #263349; border-radius: 14px; padding: 18px; margin-top: 18px; }
    label { display: block; margin: 10px 0 5px; color: #b9c6d8; }
    textarea, input, select { box-sizing: border-box; width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #34445f; background: #0e1623; color: white; }
    textarea { min-height: 100px; resize: vertical; font-family: inherit; }
    button { margin-top: 16px; border: 0; border-radius: 9px; padding: 11px 18px; background: #4f7cff; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .55; cursor: wait; }
    .switch-row { display: flex; align-items: center; gap: 9px; margin-top: 14px; color: #b9c6d8; }
    .switch-row input { width: auto; }
    .job { padding: 10px 0; border-bottom: 1px solid #29364a; }
    .error { color: #ff8e8e; white-space: pre-wrap; }
    .note { font-size: 13px; color: #94a3b8; margin-top: 4px; }
    .multi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .multi-prompts > div { margin-bottom: 8px; }
    .multi-prompts textarea { min-height: 60px; }
    .multi-gallery { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
    .multi-video-card { min-width: 0; padding: 8px; background: #0e1623; border: 1px solid #29364a; border-radius: 8px; }
    .multi-video-card video { display: block; width: 100%; border-radius: 6px; }
    .multi-video-card .idx { color: #94a3b8; font-size: 11px; margin-bottom: 4px; }
    .telemetry-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px,1fr)); gap: 8px; margin-top: 10px; }
    .telemetry-item { background: #0e1623; border: 1px solid #29364a; border-radius: 8px; padding: 10px; text-align: center; }
    .telemetry-item .val { font-size: 20px; font-weight: 700; color: #b9d0ff; }
    .telemetry-item .lbl { font-size: 11px; color: #94a3b8; margin-top: 2px; }
    .log-box { background: #0a0f18; border: 1px solid #29364a; border-radius: 8px; padding: 10px; margin-top: 8px; max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 12px; line-height: 1.5; }
    .log-box .line { color: #94a3b8; }
    .log-box .line.info { color: #b9d0ff; }
    .log-box .line.error { color: #ff8e8e; }
    .log-box .line.done { color: #6fcf97; }
    .multi-worker-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 8px; }
    .multi-worker-item { background: #0e1623; border: 1px solid #29364a; border-radius: 6px; padding: 6px 10px; font-size: 12px; }
    .multi-worker-item .w-name { color: #94a3b8; }
    .multi-worker-item .w-status { font-weight: 700; }
    .multi-worker-item .w-status.running { color: #f1c40f; }
    .multi-worker-item .w-status.done { color: #6fcf97; }
    .multi-worker-item .w-status.failed { color: #ff8e8e; }
    .multi-worker-item .w-status.spawned { color: #b9d0ff; }
    .multi-worker-item .w-status.queued { color: #94a3b8; }
    .multi-worker-item .w-log { color: #6c7a91; font-family: monospace; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .status-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 700; }
    .status-badge.running { background: #f1c40f22; color: #f1c40f; border: 1px solid #f1c40f44; }
    .status-badge.succeeded { background: #6fcf9722; color: #6fcf97; border: 1px solid #6fcf9744; }
    .status-badge.failed { background: #ff8e8e22; color: #ff8e8e; border: 1px solid #ff8e8e44; }
    .status-badge.queued { background: #94a3b822; color: #94a3b8; border: 1px solid #94a3b844; }
    .status-badge.idle { background: #4f7cff22; color: #4f7cff; border: 1px solid #4f7cff44; }
    .progress-bar { background: #29364a; border-radius: 6px; height: 6px; margin-top: 8px; overflow: hidden; }
    .progress-bar .fill { height: 100%; border-radius: 6px; background: #4f7cff; transition: width 0.5s; }
    @media (max-width: 700px) { .multi-gallery { grid-template-columns: 1fr 1fr; } .multi-grid { grid-template-columns: 1fr; } .multi-worker-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body><main>
  <h1>__MODEL_DISPLAY_NAME__</h1>
  <div class="muted">All state is server-side. Every connected client sees the same view. Multi mode: __MULTI_MODE__×.</div>

  <section class="card">
    <label>API Token</label><input id="token" type="password" autocomplete="off" placeholder="Required for submitting jobs">
  </section>

  <h2>Single Video Generation (1024×1024, 121 frames)</h2>
  <section class="card">
    <label>Prompt</label>
    <textarea id="singlePrompt" placeholder="Enter your prompt...">A cinematic shot of a red panda sitting on a mossy branch in a misty bamboo forest, photorealistic, 4k.</textarea>
    <button id="singleSubmit">Generate Video</button>
    <div id="singleStatus" class="note" style="margin-top:12px"></div>
    <div id="singleProgress" class="progress-bar" style="display:none"><div class="fill" style="width:0%"></div></div>
    <div id="singleLog" class="log-box" style="display:none"></div>
    <div id="singleVideoContainer" style="margin-top:8px"></div>
    <div id="singleError" class="error"></div>
  </section>

  <h2>Multi Video Generation (__MULTI_MODE__×)</h2>
  <section class="card">
    <div class="note" style="margin-bottom:10px">__MULTI_MODE__ concurrent 1024×1024, 121-frame videos.</div>
    __MULTI_PROMPTS_HTML__
    <button id="multiSubmit">Generate __MULTI_MODE__ Videos</button>
    <div id="multiStatus" class="note" style="margin-top:12px"></div>
    <div id="multiWorkerPanel" class="multi-worker-grid" style="margin-top:8px"></div>
    <div id="multiLog" class="log-box" style="display:none"></div>
    <div id="multiError" class="error"></div>
    <div id="multiGallery" class="multi-gallery"></div>
  </section>

  <section class="card">
    <h2 style="margin:0 0 8px">Server Log <span class="note">(shared, last __MAX_LOG_LINES__ lines)</span></h2>
    <div id="serverLog" class="log-box"></div>
  </section>

  <section class="card">
    <h2 style="margin:0 0 8px">XPU Telemetry</h2>
    <label class="switch-row"><input id="showTelemetry" type="checkbox" checked> Show hardware telemetry</label>
    <div id="xpuTelemetry" class="telemetry-grid"></div>
  </section>

  <section class="card"><h2>Recent Single Jobs</h2><div id="singleHistory"></div></section>
  <section class="card"><h2>Recent Multi-Jobs</h2><div id="multiHistory"></div></section>
</main>
<script>
const $ = id => document.getElementById(id);
const STATE_LABELS = {idle:'Idle',queued:'Queued',running:'Running',succeeded:'Completed',failed:'Failed'};
const DEFAULT_PROMPTS = [
  'A cinematic shot of a red panda sitting on a mossy branch in a misty bamboo forest, photorealistic, 4k.',
  'A majestic eagle soaring over a deep canyon at golden hour, warm sunlight, cinematic, 8k.',
  'An underwater scene with a sea turtle swimming through a coral reef, volumetric lighting, 4k.',
  'A cyberpunk city at night with neon signs reflecting on wet streets, blade runner aesthetic, 8k.',
  'A serene mountain lake at sunrise with mist rising from the water, photorealistic, warm golden light.',
  'A macro shot of a dragonfly perched on a dewy leaf, morning light, shallow depth of field, 4k.',
  'A medieval castle on a stormy cliff edge, lightning flashing, dramatic clouds, epic scale.',
  'A futuristic greenhouse on Mars under a transparent dome, lush exotic plants, sci-fi, 8k.',
  'A steaming cup of coffee on a wooden table at sunrise, cinematic, warm tones, photorealistic, 4k.',
  'A wizard casting a spell in an ancient library, floating books, mystical blue light, cinematic.',
  'A neon-lit sushi bar in Tokyo at midnight, rain on window, reflections, cyberpunk aesthetic.',
  'A polar bear on a melting ice floe at sunset, dramatic sky, climate change, photorealistic.',
  'A race car speeding through a futuristic city tunnel, motion blur, neon reflections, 8k.',
  'A ballerina performing on an empty stage, spotlight, dust particles, emotional, cinematic.',
  'A supercell thunderstorm over a prairie at twilight, lightning, rotating clouds, epic scale.',
  'An astronaut floating in space overlooking Earth, stars, cosmic rays, photorealistic, 8k.'
];
const ST = sessionStorage;
$('token').value = ST.getItem('ltx_token') || '';
function headers(json) {
  const t = $('token').value.trim(); ST.setItem('ltx_token', t);
  const h = {}; if (t) h.Authorization = 'Bearer ' + t; if (json) h['Content-Type'] = 'application/json'; return h;
}
async function api(path, opts) {
  const r = await fetch(path, opts || {});
  if (!r.ok) { let d; try { d = (await r.json()).detail; } catch { d = r.statusText; } throw new Error(typeof d === 'string' ? d : JSON.stringify(d)); }
  return r;
}

// fill default prompts
document.querySelectorAll('.mp').forEach(ta => { ta.value = DEFAULT_PROMPTS[+ta.dataset.idx]; });

// ---- SSE: all state comes from server ----
let evtSource = null;
function connectSSE() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource('/api/events');
  evtSource.onmessage = function(event) {
    try { renderAll(JSON.parse(event.data)); } catch(e) { console.error(e); }
  };
  evtSource.onerror = function() {
    evtSource.close();
    setTimeout(connectSSE, 3000);
  };
}

function renderAll(state) {
  renderSingle(state.single);
  renderMulti(state.multi);
  renderHistory(state.history);
  renderTelemetry(state.telemetry);
  renderServerLog(state);
}

// ---- single job ----
function renderSingle(s) {
  const statusEl = $('singleStatus');
  const logEl = $('singleLog');
  const progressEl = $('singleProgress');
  const progressFill = progressEl.querySelector('.fill');
  const videoContainer = $('singleVideoContainer');
  const errorEl = $('singleError');

  if (s.status === 'idle') {
    statusEl.textContent = '';
    logEl.style.display = 'none';
    progressEl.style.display = 'none';
    videoContainer.replaceChildren();
    errorEl.textContent = '';
    return;
  }

  let html = `<span class="status-badge ${s.status}">${STATE_LABELS[s.status] || s.status}</span>`;
  if (s.progress) html += ` <span style="color:#b9d0ff;font-size:13px">${s.progress}</span>`;
  statusEl.innerHTML = html;

  if (s.error) errorEl.textContent = s.error; else errorEl.textContent = '';

  // log
  if (s.log && s.log.length) {
    logEl.style.display = '';
    logEl.replaceChildren();
    s.log.forEach(line => {
      const d = document.createElement('div'); d.className = 'line';
      d.textContent = line;
      logEl.appendChild(d);
    });
    logEl.scrollTop = logEl.scrollHeight;
  } else {
    logEl.style.display = 'none';
  }

  // progress bar
  if (s.progress) {
    progressEl.style.display = '';
    const m = s.progress.match(/(\d+)\/(\d+)/);
    if (m) progressFill.style.width = Math.round(parseInt(m[1])/parseInt(m[2])*100) + '%';
  } else if (s.status === 'running') {
    progressEl.style.display = '';
    progressFill.style.width = '100%';
    progressFill.style.animation = 'pulse 1.5s infinite';
  } else {
    progressEl.style.display = 'none';
  }

  // video - only rebuild if url changed (avoid flicker from SSE re-render)
  if (s.status === 'succeeded' && s.video_url && s.video_url !== videoContainer.dataset.videoUrl) {
    videoContainer.replaceChildren();
    videoContainer.dataset.videoUrl = s.video_url;
    const video = document.createElement('video');
    video.src = s.video_url; video.controls = true; video.preload = 'metadata';
    video.style.maxWidth = '100%'; video.style.maxHeight = '480px';
    videoContainer.appendChild(video);
  } else if (s.status !== 'succeeded') {
    videoContainer.replaceChildren();
    delete videoContainer.dataset.videoUrl;
  }
}

// ---- multi job ----
function renderMulti(m) {
  const statusEl = $('multiStatus');
  const logEl = $('multiLog');
  const workerPanel = $('multiWorkerPanel');
  const gallery = $('multiGallery');
  const errorEl = $('multiError');

  if (m.status === 'idle') {
    statusEl.textContent = '';
    logEl.style.display = 'none';
    workerPanel.replaceChildren();
    gallery.replaceChildren();
    errorEl.textContent = '';
    return;
  }

  let html = `<span class="status-badge ${m.status}">${STATE_LABELS[m.status] || m.status}</span>`;
  if (m.prompts) html += ` <span style="color:#94a3b8;font-size:13px">${m.prompts.length} prompts</span>`;
  if (m.videos && m.videos.length) {
    html += ' ';
    m.videos.forEach((url, i) => { html += `<a href="${url}" target="_blank" style="color:#4f7cff;margin-right:4px">[vid${i}]</a>`; });
  }
  statusEl.innerHTML = html;

  if (m.error) errorEl.textContent = m.error; else errorEl.textContent = '';

  // worker grid - only update text/classes, not DOM (avoid flicker)
  if (m.workers && m.workers.length) {
    while (workerPanel.children.length < m.workers.length) {
      const div = document.createElement('div'); div.className = 'multi-worker-item';
      div.innerHTML = '<span class="w-name"></span> <span class="w-status"></span><div class="w-log"></div>';
      workerPanel.appendChild(div);
    }
    while (workerPanel.children.length > m.workers.length) {
      workerPanel.lastChild.remove();
    }
    m.workers.forEach((w, i) => {
      const div = workerPanel.children[i];
      div.querySelector('.w-name').textContent = 'Worker ' + (w.idx+1);
      const ws = div.querySelector('.w-status');
      ws.textContent = w.status;
      ws.className = 'w-status ' + w.status;
      div.querySelector('.w-log').textContent = w.last_log || '';
    });
  } else {
    workerPanel.replaceChildren();
  }

  // log
  if (m.log && m.log.length) {
    logEl.style.display = '';
    logEl.replaceChildren();
    m.log.forEach(line => {
      const d = document.createElement('div'); d.className = 'line';
      if (line.startsWith('[error]')) d.classList.add('error');
      else if (line.startsWith('[done]')) d.classList.add('done');
      else if (line.startsWith('[')) d.classList.add('info');
      d.textContent = line;
      logEl.appendChild(d);
    });
    logEl.scrollTop = logEl.scrollHeight;
  } else {
    logEl.style.display = 'none';
  }

  // gallery - only rebuild when video URLs change
  const newVideos = m.status === 'succeeded' && m.videos ? m.videos.join(',') : '';
  if (newVideos && newVideos !== gallery.dataset.videosKey) {
    gallery.replaceChildren();
    gallery.dataset.videosKey = newVideos;
    (m.prompts || []).forEach((p, i) => {
      const card = document.createElement('div'); card.className = 'multi-video-card';
      const label = document.createElement('div'); label.className = 'idx';
      label.textContent = 'Video ' + (i+1) + ': ' + p.slice(0, 60);
      card.appendChild(label);
      const video = document.createElement('video');
      video.src = m.videos[i]; video.controls = true; video.preload = 'metadata';
      video.style.maxHeight = '240px'; video.style.width = '100%';
      card.appendChild(video);
      gallery.appendChild(card);
    });
  } else if (!newVideos) {
    gallery.replaceChildren();
    delete gallery.dataset.videosKey;
  }
}

// ---- history ----
function renderHistory(h) {
  // single history
  const singleRoot = $('singleHistory');
  singleRoot.replaceChildren();
  if (h && h.single) {
    h.single.forEach(job => {
      const row = document.createElement('div'); row.className = 'job';
      row.textContent = (job.created_at || '') + ' \u00b7 ' + STATE_LABELS[job.status] + ' \u00b7 ' + (job.prompt || '').slice(0, 100);
      if (job.status === 'succeeded' && job.video_url) {
        const a = document.createElement('a'); a.href = job.video_url; a.target = '_blank';
        a.textContent = ' [video]'; a.style.color = '#4f7cff'; a.style.marginLeft = '8px';
        row.appendChild(a);
      }
      singleRoot.appendChild(row);
    });
  }

  // multi history
  const multiRoot = $('multiHistory');
  multiRoot.replaceChildren();
  if (h && h.multi) {
    h.multi.forEach(job => {
      const row = document.createElement('div'); row.className = 'job';
      const first = (job.prompts || [])[0] || '';
      const label = document.createElement('span');
      label.textContent = (job.created_at || '') + ' \u00b7 ' + STATE_LABELS[job.status] + ' \u00b7 ' + first.slice(0, 100);
      row.appendChild(label);
      if (job.status === 'succeeded' && job.videos) {
        job.videos.forEach((url, i) => {
          const a = document.createElement('a'); a.href = url; a.target = '_blank';
          a.textContent = ' [vid' + i + ']'; a.style.color = '#4f7cff'; a.style.marginRight = '4px';
          row.appendChild(a);
        });
      }
      multiRoot.appendChild(row);
    });
  }
}

// ---- server log (combines single + multi logs) ----
function renderServerLog(state) {
  const el = $('serverLog');
  const lines = [];
  if (state.single && state.single.log) state.single.log.forEach(l => lines.push(l));
  if (state.multi && state.multi.log) state.multi.log.forEach(l => lines.push(l));
  if (!lines.length) { el.style.display = 'none'; return; }
  el.style.display = '';
  el.replaceChildren();
  lines.slice(-40).forEach(line => {
    const d = document.createElement('div'); d.className = 'line';
    if (line.includes('error') || line.includes('fail')) d.classList.add('error');
    d.textContent = line;
    el.appendChild(d);
  });
  el.scrollTop = el.scrollHeight;
}

// ---- telemetry ----
const savedTelemetry = ST.getItem('ltx_show_telemetry');
$('showTelemetry').checked = savedTelemetry === null ? true : savedTelemetry === 'true';
function renderTelemetry(t) {
  const panel = $('xpuTelemetry');
  panel.hidden = !$('showTelemetry').checked;
  if (panel.hidden || !t || !t.cards) return;
  const items = [
    {val: t.cards, lbl: 'XPU Cards'},
    {val: t.pkg_temp_avg_c != null ? t.pkg_temp_avg_c + '°C' : 'N/A', lbl: 'Pkg Temp (avg)'},
    {val: t.pkg_temp_max_c != null ? t.pkg_temp_max_c + '°C' : 'N/A', lbl: 'Pkg Temp (max)'},
    {val: t.vram_temp_avg_c != null ? t.vram_temp_avg_c + '°C' : 'N/A', lbl: 'VRAM Temp (avg)'},
    {val: t.vram_temp_max_c != null ? t.vram_temp_max_c + '°C' : 'N/A', lbl: 'VRAM Temp (max)'},
    {val: t.energy_total_j != null ? (t.energy_total_j/3600).toFixed(2) + ' Wh' : 'N/A', lbl: 'Energy (total)'},
    {val: t.fan_rpm_avg != null ? t.fan_rpm_avg + ' RPM' : 'N/A', lbl: 'Fan (avg)'},
  ];
  panel.replaceChildren();
  items.forEach(it => {
    const d = document.createElement('div'); d.className = 'telemetry-item';
    d.innerHTML = '<div class="val">' + it.val + '</div><div class="lbl">' + it.lbl + '</div>';
    panel.appendChild(d);
  });
}
$('showTelemetry').onchange = () => {
  ST.setItem('ltx_show_telemetry', String($('showTelemetry').checked));
};

// ---- submit ----
async function submitSingle() {
  $('singleSubmit').disabled = true;
  const p = $('singlePrompt').value.trim() || DEFAULT_PROMPTS[0];
  try {
    await api('/api/jobs', { method: 'POST', headers: headers(true), body: JSON.stringify({prompt: p, width: 1024, height: 1024, frames: 121}) });
  } catch(e) { $('singleError').textContent = e.message; }
  finally { $('singleSubmit').disabled = false; }
}
async function submitMulti() {
  $('multiSubmit').disabled = true;
  const prompts = [];
  document.querySelectorAll('.mp').forEach(ta => { const v = ta.value.trim(); prompts[+ta.dataset.idx] = v || DEFAULT_PROMPTS[+ta.dataset.idx]; });
  try {
    await api('/api/multi-jobs', { method: 'POST', headers: headers(true), body: JSON.stringify({prompts: prompts}) });
  } catch(e) { $('multiError').textContent = e.message; }
  finally { $('multiSubmit').disabled = false; }
}

$('singleSubmit').onclick = submitSingle;
$('multiSubmit').onclick = submitMulti;

connectSSE();
</script></body></html>"""


# ---------------------------------------------------------------------------
# XPU hardware monitoring
# ---------------------------------------------------------------------------

def _find_hwmon_dirs(max_cards: int = 16) -> list[tuple[int, str]]:
    """Return sorted list of (xpu_index, hwmon_dir_path) for the first *max_cards* XPU cards."""
    results = []
    for entry in sorted(os.listdir("/sys/class/drm")):
        if not entry.startswith("card"):
            continue
        card_num = entry[len("card"):]
        if not card_num.isdigit():
            continue
        idx = int(card_num)
        # card0 is the display card; skip it
        if idx == 0:
            continue
        if len(results) >= max_cards:
            break
        hwmon_dir = f"/sys/class/drm/{entry}/device/hwmon"
        if not os.path.isdir(hwmon_dir):
            continue
        for hw_entry in sorted(os.listdir(hwmon_dir)):
            hw_path = f"{hwmon_dir}/{hw_entry}"
            name_file = f"{hw_path}/name"
            if os.path.isfile(name_file):
                with open(name_file) as f:
                    if f.read().strip() == "xe":
                        results.append((idx, hw_path))
                        break
    return results


def _read_sysfs(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def xpu_stats() -> dict:
    """Return aggregate XPU hardware metrics."""
    cards = _find_hwmon_dirs()
    pkg_temps: list[int] = []
    vram_temps: list[int] = []
    energies: list[int] = []
    fans: list[int] = []

    for idx, hw in cards:
        for fname in os.listdir(hw):
            fpath = f"{hw}/{fname}"
            if not os.path.isfile(fpath):
                continue
            # temperature: temp2_input = pkg, temp3_input = vram
            if fname == "temp2_input":
                v = _read_sysfs(fpath)
                if v is not None:
                    pkg_temps.append(v)
            elif fname == "temp3_input":
                v = _read_sysfs(fpath)
                if v is not None:
                    vram_temps.append(v)
            elif fname == "energy1_input":
                v = _read_sysfs(fpath)
                if v is not None:
                    energies.append(v)
            elif fname.startswith("fan") and fname.endswith("_input"):
                v = _read_sysfs(fpath)
                if v is not None:
                    fans.append(v)

    def avg(vals: list[int]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    def max_val(vals: list[int]) -> int | None:
        return max(vals) if vals else None

    return {
        "cards": len(cards),
        "pkg_temp_avg_c": round(avg(pkg_temps) / 1000, 1) if pkg_temps else None,
        "pkg_temp_max_c": round(max_val(pkg_temps) / 1000, 1) if pkg_temps else None,
        "vram_temp_avg_c": round(avg(vram_temps) / 1000, 1) if vram_temps else None,
        "vram_temp_max_c": round(max_val(vram_temps) / 1000, 1) if vram_temps else None,
        "energy_total_j": round(sum(energies) / 1e6, 1) if energies else None,
        "fan_rpm_avg": round(avg(fans)) if fans else None,
    }


def _generate_prompts_html(n: int) -> str:
    """Generate the multi-prompt textarea grid for *n* prompts (8 or 16)."""
    if n <= 8:
        cols = 2
        rows_per = n // cols
    else:
        cols = 4
        rows_per = n // cols
    parts = ['<div class="multi-grid">']
    for c in range(cols):
        parts.append('<div class="multi-prompts">')
        for r in range(rows_per):
            i = c * rows_per + r
            parts.append(
                f'<div><label>Prompt {i+1}</label>'
                f'<textarea class="mp" data-idx="{i}"></textarea></div>'
            )
        parts.append('</div>')
    parts.append('</div>')
    return "\n".join(parts)


def render_html(profile: ModelProfile) -> str:
    prompts_html = _generate_prompts_html(profile.multi_mode)
    return (
        HTML_TEMPLATE.replace("__MODEL_DISPLAY_NAME__", profile.display_name)
        .replace("__DEFAULT_WIDTH__", str(profile.default_width))
        .replace("__DEFAULT_HEIGHT__", str(profile.default_height))
        .replace("__DEFAULT_FRAMES__", str(profile.default_frames))
        .replace("__MAX_LOG_LINES__", str(MAX_LOG_LINES))
        .replace("__MULTI_PROMPTS_HTML__", prompts_html)
        .replace("__MULTI_MODE__", str(profile.multi_mode))
    )


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_server(profile: ModelProfile) -> ServerApplication:
    settings = ServerSettings.from_environment()
    store = JobStore(settings.database_path)
    state = ServerState()
    worker: LtxWorker | None = None
    multi_worker: MultiLtxWorker | None = None
    _history_timer: threading.Event | None = None

    def refresh_history() -> None:
        """Load recent history from DB into ServerState (runs periodically)."""
        try:
            state.single_history = [
                public_job(j) for j in store.list(limit=10, offset=0)
            ]
            state.multi_history = [
                public_multi_job(j) for j in store.list_multi(limit=10, offset=0)
            ]
        except Exception as e:
            logger.warning("refresh_history error: %s", e)

    class JobRequest(BaseModel):
        prompt: str = Field(min_length=1, max_length=4096)
        seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
        width: int = Field(default=profile.default_width, ge=64, le=2048)
        height: int = Field(default=profile.default_height, ge=64, le=2048)
        frames: int = Field(default=profile.default_frames, ge=1, le=1024)

    class MultiJobRequest(BaseModel):
        prompts: list[str] = Field(min_length=1, max_length=profile.multi_mode)

    def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        if not settings.api_token:
            return
        scheme, _, value = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(value, settings.api_token):
            raise HTTPException(status_code=401, detail="Invalid or missing API token")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal worker, multi_worker, _history_timer
        if not is_loopback(settings.host) and not settings.api_token:
            raise RuntimeError(
                "LTX_API_TOKEN is required when LTX_HOST is not a loopback address."
            )
        store.initialize()
        refresh_history()
        worker = LtxWorker(
            store=store,
            output_dir=settings.output_dir,
            state=state,
            queue_size=settings.queue_size,
        )
        worker.start()
        app.state.worker = worker
        multi_worker = MultiLtxWorker(
            store=store,
            output_dir=settings.output_dir,
            state=state,
            max_workers=profile.multi_mode,
        )
        multi_worker.start()
        app.state.multi_worker = multi_worker
        # background history refresher every 5s
        _stop = threading.Event()
        _history_timer = _stop

        def _history_loop():
            while not _stop.wait(5):
                refresh_history()

        threading.Thread(target=_history_loop, daemon=True).start()
        try:
            yield
        finally:
            _stop.set()
            worker.stop()
            multi_worker.stop()

    app = FastAPI(
        title=f"{profile.display_name} Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.profile = profile
    app.state.settings = settings
    app.state.store = store
    html_page = render_html(profile)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return html_page

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/events")
    async def event_stream():
        """SSE endpoint — broadcasts server state to all connected clients."""
        async def _generate():
            last_json = ""
            while True:
                snap = state.snapshot(xpu_stats())
                cur = json.dumps(snap, default=str)
                # only send if state changed
                if cur != last_json:
                    yield f"data: {cur}\n\n"
                    last_json = cur
                await asyncio.sleep(0.8)
        return StreamingResponse(_generate(), media_type="text/event-stream")

    @app.get("/api/xpu-stats")
    def get_xpu_stats():
        return xpu_stats()

    @app.get("/ready", dependencies=[Depends(require_token)])
    def ready():
        w = app.state.worker
        if not w.ready:
            raise HTTPException(
                status_code=503,
                detail={"status": "loading_or_failed", "queued": w.queued_count},
            )
        return {"status": "ready", "queued": w.queued_count}

    @app.post("/api/jobs", status_code=202, dependencies=[Depends(require_token)])
    def create_job(request: JobRequest):
        if request.width % 64 or request.height % 64:
            raise HTTPException(
                status_code=422,
                detail="Width and height must be divisible by 64 for two-stage pipeline.",
            )
        validate_frames(request.frames)
        w = app.state.worker
        if not w.ready:
            raise HTTPException(status_code=503, detail="Worker is not ready.")
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=422, detail="Prompt must not be blank.")
        parameters = request.model_dump()
        parameters["prompt"] = prompt
        if parameters["seed"] is None:
            parameters["seed"] = generate_seed()
        job_id = uuid4().hex
        store.create(job_id, parameters)
        task = {"id": job_id, **parameters}
        try:
            worker.submit(task)
        except queue.Full:
            store.mark_failed(job_id, "Inference queue is full")
            raise HTTPException(status_code=429, detail="Inference queue is full.")
        return public_job(store.get(job_id))

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(require_token)])
    def get_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return public_job(job)

    @app.get("/api/jobs", dependencies=[Depends(require_token)])
    def list_jobs(
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        return {
            "items": [public_job(job) for job in store.list(limit, offset)],
            "limit": limit,
            "offset": offset,
        }

    @app.get("/api/jobs/{job_id}/video")
    def get_video(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != "succeeded" or not job["video_path"]:
            raise HTTPException(status_code=409, detail="Video is not available")
        video_path = (settings.output_dir / Path(job["video_path"]).name).resolve()
        if video_path.parent != settings.output_dir or not video_path.is_file():
            raise HTTPException(status_code=404, detail="Video file not found")
        return FileResponse(video_path, media_type="video/mp4",
                            headers={"Content-Disposition": "inline"})

    @app.post("/api/multi-jobs", status_code=202, dependencies=[Depends(require_token)])
    def create_multi_job(request: MultiJobRequest):
        mw = app.state.multi_worker
        if not mw.ready:
            raise HTTPException(status_code=503, detail="Multi-worker is not ready.")
        prompts = [p.strip() for p in request.prompts if p.strip()]
        if not prompts:
            raise HTTPException(status_code=422, detail="At least one non-blank prompt required.")
        if len(prompts) > profile.multi_mode:
            raise HTTPException(status_code=422, detail=f"Maximum {profile.multi_mode} prompts.")
        job_id = uuid4().hex
        store.create_multi(job_id, prompts)
        try:
            mw.submit({"id": job_id, "prompts": prompts})
        except queue.Full:
            store.mark_multi_failed(job_id, "Multi-job queue is full")
            raise HTTPException(status_code=429, detail="Multi-job queue is full.")
        return public_multi_job(store.get_multi(job_id))

    @app.get("/api/multi-jobs", dependencies=[Depends(require_token)])
    def list_multi_jobs():
        return [public_multi_job(j) for j in store.list_multi(limit=20, offset=0)]

    @app.get("/api/multi-jobs/{job_id}", dependencies=[Depends(require_token)])
    def get_multi_job(job_id: str):
        job = store.get_multi(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Multi-job not found")
        return public_multi_job(job)

    @app.get("/api/multi-jobs/{job_id}/videos/{index:int}")
    def get_multi_video(job_id: str, index: int):
        job = store.get_multi(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Multi-job not found")
        if job["status"] != "succeeded" or not job["output_dir"]:
            raise HTTPException(status_code=409, detail="Videos not available")
        if index < 0 or index >= len(job["prompts"]):
            raise HTTPException(status_code=422, detail=f"Index out of range: 0..{len(job['prompts'])-1}")
        video_path = Path(job["output_dir"]) / f"video_{index}.mp4"
        if not video_path.is_file():
            raise HTTPException(status_code=404, detail="Video file not found")
        return FileResponse(video_path, media_type="video/mp4",
                            headers={"Content-Disposition": "inline"})

    return ServerApplication(app=app, settings=settings, profile=profile)


def run_server(server: ServerApplication) -> None:
    import uvicorn
    uvicorn.run(server.app, host=server.settings.host, port=server.settings.port, workers=1)
