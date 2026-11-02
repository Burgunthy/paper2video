"""Deterministic renderer for figure scenes — no LLM, no manim.

Composes the paper figure onto a black canvas (matching manim scenes), adds
an optional highlight box, then animates a slow zoom-in with ffmpeg zoompan
plus a caption strip and a source tag. Output is a silent mp4 whose duration
covers the narration; the pipeline muxes audio and trims to length.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..types import Scene

BG = (0, 0, 0)
ACCENT = (255, 201, 64)        # amber highlight
MARGIN_X = 56
MARGIN_TOP = 44
CAPTION_STRIP_H = 84
TAG_H = 54

REGIONS = {
    "full":        (0.0, 0.0, 1.0, 1.0),
    "left":        (0.0, 0.0, 0.5, 1.0),
    "right":       (0.5, 0.0, 1.0, 1.0),
    "top":         (0.0, 0.0, 1.0, 0.5),
    "bottom":      (0.0, 0.5, 1.0, 1.0),
    "center":      (0.25, 0.25, 0.75, 0.75),
    "top-left":    (0.0, 0.0, 0.5, 0.5),
    "top-right":   (0.5, 0.0, 1.0, 0.5),
    "bottom-left": (0.0, 0.5, 0.5, 1.0),
    "bottom-right":(0.5, 0.5, 1.0, 1.0),
}


class FigureRenderError(Exception):
    pass


def _find_font() -> str:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    found = shutil.which("fc-match")
    if found:
        out = subprocess.run(["fc-match", "-f", "%{file}", "Noto Sans CJK KR"],
                             capture_output=True, text=True).stdout.strip()
        if out and Path(out).exists():
            return out
    raise FigureRenderError("no CJK font found for drawtext")


def _region(name: str | None) -> tuple[float, float, float, float]:
    return REGIONS.get((name or "").strip().lower(), REGIONS["full"])


def _load_manifest(fig_dir: Path) -> list[dict]:
    mf = fig_dir / "figures.json"
    if not mf.exists():
        raise FigureRenderError(f"figures.json not found in {fig_dir}")
    return json.loads(mf.read_text(encoding="utf-8"))


def _compose_base(scene: Scene, fig_dir: Path, width: int, height: int, out_png: Path) -> tuple[float, float]:
    """Draw figure+highlight on canvas. Returns focus point (fx, fy) in 0..1."""
    from PIL import Image, ImageDraw

    spec = scene.visual_spec or {}
    fig_id = str(spec.get("figure_id", "")).strip()
    manifest = _load_manifest(fig_dir)
    entry = next((f for f in manifest if f["id"] == fig_id), None)
    if entry is None:
        raise FigureRenderError(f"figure_id {fig_id!r} not in manifest (have {[f['id'] for f in manifest]})")

    fig = Image.open(fig_dir / entry["path"]).convert("RGB")

    # Focus crop of the source figure
    fx0, fy0, fx1, fy1 = _region(spec.get("focus"))
    if (fx0, fy0, fx1, fy1) != REGIONS["full"]:
        w, h = fig.size
        fig = fig.crop((int(w * fx0), int(h * fy0), int(w * fx1), int(h * fy1)))

    canvas = Image.new("RGB", (width, height), BG)
    avail_w = width - 2 * MARGIN_X
    avail_h = height - MARGIN_TOP - CAPTION_STRIP_H - 16
    scale = min(avail_w / fig.width, avail_h / fig.height)
    new_size = (max(1, int(fig.width * scale)), max(1, int(fig.height * scale)))
    fig = fig.resize(new_size, Image.LANCZOS)
    pos_x = (width - fig.width) // 2
    pos_y = MARGIN_TOP + max(0, (avail_h - fig.height) // 2)
    canvas.paste(fig, (pos_x, pos_y))

    # Highlight box over a sub-region of the placed figure
    hl = str(spec.get("highlight", "") or "").strip().lower()
    if hl and hl != "none" and hl != "full":
        hx0, hy0, hx1, hy1 = _region(hl)
        box = (pos_x + int(fig.width * hx0), pos_y + int(fig.height * hy0),
               pos_x + int(fig.width * hx1), pos_y + int(fig.height * hy1))
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle(box, fill=ACCENT + (34,), outline=ACCENT + (255,), width=5)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    # Focus point for zoompan = center of focus/highlight region on canvas
    cx = (fx0 + fx1) / 2 if (fx0, fy0, fx1, fy1) != REGIONS["full"] else 0.5
    cy = (fy0 + fy1) / 2 if (fx0, fy0, fx1, fy1) != REGIONS["full"] else 0.5
    if hl and hl in REGIONS:
        hx0, hy0, hx1, hy1 = REGIONS[hl]
        cx, cy = (hx0 + hx1) / 2, (hy0 + hy1) / 2

    canvas.save(out_png)
    return cx, cy


def render_figure_scene(
    scene: Scene,
    duration_sec: float,
    fig_dir: Path,
    out_dir: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_png = out_dir / f"figscene_{scene.id:03d}.png"
    out_mp4 = out_dir / f"figscene_{scene.id:03d}.mp4"
    fx, fy = _compose_base(scene, fig_dir, width, height, base_png)

    dur = max(4.0, duration_sec) + 0.8
    frames = int(dur * fps)
    spec = scene.visual_spec or {}
    caption = str(spec.get("caption", "")).replace(":", "\\:").replace("'", "\\'")
    fig_id = str(spec.get("figure_id", ""))
    manifest = _load_manifest(fig_dir)
    page = next((f["page"] for f in manifest if f["id"] == fig_id), 0)
    tag = f"Figure {fig_id.replace('fig', '')} · p.{page}".replace(":", "\\:")
    font = _find_font()

    # Upscale 4x first so zoompan steps stay sub-pixel smooth.
    up_w = width * 4
    cx_expr = f"(iw-iw/zoom)*{fx:.4f}"
    cy_expr = f"(ih-ih/zoom)*{fy:.4f}"
    zoom_expr = "'min(1+0.0016*on,1.14)'"
    vf = (
        f"scale={up_w}:-2,"
        f"zoompan=z={zoom_expr}:x={cx_expr}:y={cy_expr}:d={frames}:s={width}x{height}:fps={fps},"
        f"drawtext=fontfile={font}:text='{tag}':x=28:y=h-th-28:fontsize=26:fontcolor=0xE8E8EF:"
        f"box=1:boxcolor=0x000000@0.55:boxborderw=10,"
        f"drawtext=fontfile={font}:text='{caption}':x=(w-text_w)/2:y=h-CAPTION:fontsize=34:fontcolor=0xFFD666"
        .replace("CAPTION", "92")
    )
    if caption:
        pass  # caption strip already drawn above
    vf += ",fade=t=in:st=0:d=0.45,format=yuv420p"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(base_png),
        "-vf", vf, "-frames:v", str(frames),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not out_mp4.exists():
        raise FigureRenderError(f"ffmpeg failed: {proc.stderr[-800:]}")
    return out_mp4
