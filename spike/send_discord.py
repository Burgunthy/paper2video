#!/usr/bin/env python3
"""Send the 5 spike variant videos to the owner's Discord DM + a pick-one poll."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
ENV_FILE = Path("/home/jth/taehyeon-assistant/.env")
API = "https://discord.com/api/v10"
UPLOAD_LIMIT = 8 * 1024 * 1024  # bot upload safe margin

RUNS = [
    ("spike-v1-deep",       "V1 딥스크립트",   "manim만 · 전문 기반 12-14씬 — 깊이 개선만 반영"),
    ("spike-v2-figurewalk", "V2 피규어 워크스루", "논문 피규어 확대/하이라이트 중심 — 피규어 활용만 반영"),
    ("spike-v3-hybrid",     "V3 하이브리드",   "개념 manim + 방법·결과 실제 피규어 — 결합형"),
    ("spike-v4-deepdive",   "V4 풀 딥다이브",  "전 섹션 커버(실험·어블레이션·한계) + 피규어 — 최대 깊이"),
    ("spike-v5-compact",    "V5 컴팩트 하이브리드", "V3 레시피 압축판 — 짧고 밀도 있게"),
]


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def shrink(video: Path) -> Path:
    """Re-encode until under the Discord bot upload limit."""
    if video.stat().st_size <= UPLOAD_LIMIT:
        return video
    for h, crf in [(540, 30), (480, 32), (360, 34)]:
        out = video.with_name(f"{video.stem}-{h}p.mp4")
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
             "-vf", f"scale=-2:{h}", "-c:v", "libx264", "-crf", str(crf),
             "-preset", "veryfast", "-c:a", "aac", "-b:a", "64k", str(out)],
            capture_output=True, timeout=900,
        )
        if r.returncode == 0 and out.exists() and out.stat().st_size <= UPLOAD_LIMIT:
            return out
    raise RuntimeError(f"could not shrink {video} under limit")


def main() -> int:
    env = load_env()
    token = env["DISCORD_BOT_TOKEN"]
    owner = env["DISCORD_OWNER_IDS"].replace(",", " ").split()[0]
    headers = {"Authorization": f"Bot {token}"}

    r = requests.post(f"{API}/users/@me/channels", headers=headers,
                      json={"recipient_id": int(owner)}, timeout=30)
    r.raise_for_status()
    channel_id = r.json()["id"]
    print(f"DM channel: {channel_id}")

    intro = requests.post(
        f"{API}/channels/{channel_id}/messages", headers=headers,
        json={"content": (
            "**p2v 개선안 5개 변형** 나왔어요 — 같은 논문(HIL-SERL, 2410.21845)으로 만들었습니다.\n"
            "공통: 논문 전문 반영(기존엔 8%만 사용) + GLM-5.3 스크립트.\n"
            "변형별 차이는 각 영상 메시지에. 다 보고 투표해주시면 그 방향을 새 표준으로 잡습니다."
        )}, timeout=30)
    intro.raise_for_status()

    sent = []
    for run_id, label, desc in RUNS:
        video = REPO / "out" / run_id / "video.mp4"
        if not video.exists():
            print(f"[skip] {run_id}: no video", file=sys.stderr)
            continue
        asset = shrink(video)
        dur = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)], capture_output=True, text=True
        ).stdout.strip()
        mins = f"{float(dur)/60:.1f}분" if dur else "?분"
        payload = {"content": f"**{label}** ({mins}, {asset.stat().st_size/1e6:.1f}MB)\n{desc}"}
        with open(asset, "rb") as f:
            resp = requests.post(
                f"{API}/channels/{channel_id}/messages", headers=headers,
                data={"payload_json": json.dumps(payload)},
                files={"files[0]": (f"{label.replace(' ', '_')}.mp4", f, "video/mp4")},
                timeout=300,
            )
        resp.raise_for_status()
        sent.append(label)
        print(f"[sent] {label} ({mins})")
        time.sleep(2)

    poll = {
        "poll": {
            "question": {"media": {"text": "p2v 새 표준 — 어느 방향? (V1~V5)"}},
            "answers": [
                {"answer_id": i + 1, "poll_media": {"text": f"{label} — {desc.split('—')[0].strip()}"}}
                for i, (_rid, label, desc) in enumerate(RUNS)
            ],
            "allow_multiselect": False,
            "duration": 48,
        }
    }
    resp = requests.post(f"{API}/channels/{channel_id}/messages", headers=headers,
                         json=poll, timeout=30)
    resp.raise_for_status()
    print(f"[sent] poll ({len(sent)} videos delivered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
