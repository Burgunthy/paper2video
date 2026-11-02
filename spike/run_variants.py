#!/usr/bin/env python3
"""Spike orchestrator: render 5 quality-variant videos of the same paper.

Runs the paper2video CLI twice concurrently (4-core box), shortest variants
first for fast feedback. Logs to spike/logs/<run_id>.log.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENV_FILE = Path("/home/jth/taehyeon-assistant/.env")
SOURCE = "https://arxiv.org/pdf/2410.21845"

VARIANTS = [
    # (run_id, variant, keypoints) — ordered fastest-first
    ("spike-v5-compact",     "compact",     10),
    ("spike-v2-figurewalk",  "figurewalk",  10),
    ("spike-v1-deep",        "deep",        12),
    ("spike-v3-hybrid",      "hybrid",      14),
    ("spike-v4-deepdive",    "deepdive",    18),
]

DESCRIPTIONS = {
    "deep":       "V1 딥스크립트 — manim만, 전문 기반 12-14씬 (깊이 개선 격리)",
    "figurewalk": "V2 피규어 워크스루 — 실제 논문 피규어 확대/하이라이트 중심",
    "hybrid":     "V3 하이브리드 — 개념 manim + 방법·결과 실제 피규어",
    "deepdive":   "V4 풀 딥다이브 — 전 섹션(실험·어블레이션·한계) + 피규어",
    "compact":    "V5 컴팩트 하이브리드 — V3 레시피 압축판",
}


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def build_env(base: dict[str, str], keypoints: int) -> dict[str, str]:
    env = {**os.environ, **base}
    env.update({
        "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
        "ANTHROPIC_API_KEY": base["ZAI_API_KEY"],
        "PAPER2VIDEO_MODEL": "glm-5.3",
        "PAPER2VIDEO_MAX_TOKENS": "16384",
        "PAPER2VIDEO_TEXT_CAP": "150000",
        "PAPER2VIDEO_KEYPOINTS": str(keypoints),
        "PAPER2VIDEO_LANG": "ko",
        "PAPER2VIDEO_TTS": "edge",
    })
    return env


def main() -> int:
    logs = REPO / "spike" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    base = load_env()
    remaining = list(VARIANTS)
    running: dict[str, subprocess.Popen] = {}
    done: dict[str, int] = {}
    max_parallel = 2

    while remaining or running:
        while remaining and len(running) < max_parallel:
            run_id, variant, kp = remaining.pop(0)
            log_path = logs / f"{run_id}.log"
            env = build_env(base, kp)
            env["PAPER2VIDEO_VARIANT"] = variant
            cmd = [
                str(REPO / ".venv/bin/paper2video"), SOURCE,
                "--no-search", "--width", "1280", "--height", "720",
                "--run-id", run_id,
            ]
            lf = open(log_path, "ab")
            lf.write(f"\n===== {time.strftime('%H:%M:%S')} launch {variant} =====\n".encode())
            lf.flush()
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                    stdout=lf, stderr=subprocess.STDOUT)
            running[run_id] = proc
            print(f"[launch] {run_id} (pid {proc.pid})", flush=True)

        time.sleep(15)
        for run_id, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[run_id]
            done[run_id] = rc
            video = REPO / "out" / run_id / "video.mp4"
            ok = rc == 0 and video.exists()
            size = f"{video.stat().st_size/1e6:.1f}MB" if video.exists() else "-"
            print(f"[{'OK  ' if ok else 'FAIL'}] {run_id} rc={rc} video={size}", flush=True)

    print("\n=== summary ===")
    for run_id, rc in done.items():
        video = REPO / "out" / run_id / "video.mp4"
        size = f"{video.stat().st_size/1e6:.1f}MB" if video.exists() else "missing"
        print(f"  {run_id}: rc={rc} {size}")
    return 0 if all(rc == 0 for rc in done.values()) and len(done) == len(VARIANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
