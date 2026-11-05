"""Deterministic renderer for figure scenes — no LLM, no manim.

The figure is composed onto its own layer (rendered at 2x for zoom headroom)
and only that layer is animated with a slow ffmpeg zoompan, then overlaid onto
a static canvas. Because the zoom lives inside the figure box, the figure can
never expand into the caption strip at the bottom — subtitles stay readable.
A caption strip and a source tag are drawn (both on opaque boxes) after the
overlay, so text is always legible. Output is a silent mp4 whose duration
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
LAYER_SCALE = 2                # figure layer supersampling for smooth zoompan

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


def _compose_layers(
    scene: Scene, fig_dir: Path, width: int, height: int, out_dir: Path,
) -> tuple[Path, Path, tuple[int, int, int, int], tuple[float, float]]:
    """Build the static background canvas and the 2x supersampled figure layer.

    Returns (bg_png, layer_png, box(x, y, w, h) on the final canvas,
    focus point (fx, fy) in 0..1 on the layer).
    """
    from PIL import Image

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

    # Figure box on the final canvas — must NEVER overlap the caption strip.
    avail_w = width - 2 * MARGIN_X
    avail_h = height - MARGIN_TOP - CAPTION_STRIP_H - 16
    scale = min(avail_w / fig.width, avail_h / fig.height)
    box_w = max(2, int(fig.width * scale)) & ~1   # even dims for yuv420p safety
    box_h = max(2, int(fig.height * scale)) & ~1
    box_x = (width - box_w) // 2
    box_y = MARGIN_TOP + max(0, (avail_h - box_h) // 2)

    bg = Image.new("RGB", (width, height), BG)

    # Figure layer at LAYER_SCALE x — zoompan crops inside it, so extra
    # resolution directly becomes zoom sharpness.
    layer = Image.new("RGB", (box_w * LAYER_SCALE, box_h * LAYER_SCALE), BG)
    s2 = min(layer.width / fig.width, layer.height / fig.height)
    fw, fh = max(1, int(fig.width * s2)), max(1, int(fig.height * s2))
    fig_scaled = fig.resize((fw, fh), Image.LANCZOS)
    fx_off = (layer.width - fw) // 2
    fy_off = (layer.height - fh) // 2
    layer.paste(fig_scaled, (fx_off, fy_off))

    # NOTE: no highlight boxes. Fractional-region outlines never line up with
    # real panel boundaries and read as arbitrary vertical/horizontal lines.
    # The zoom centers on the focus region only.

    # Focus point for the zoom = center of the focus region
    cx = (fx0 + fx1) / 2
    cy = (fy0 + fy1) / 2

    out_dir.mkdir(parents=True, exist_ok=True)
    bg_png = out_dir / f"figscene_{scene.id:03d}_bg.png"
    layer_png = out_dir / f"figscene_{scene.id:03d}_layer.png"
    bg.save(bg_png)
    layer.save(layer_png)

    # Flattened preview at 1x for debugging/QA
    preview = bg.copy()
    preview.paste(layer.resize((box_w, box_h), Image.LANCZOS), (box_x, box_y))
    preview.save(out_dir / f"figscene_{scene.id:03d}.png")

    return bg_png, layer_png, (box_x, box_y, box_w, box_h), (cx, cy)


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
    out_mp4 = out_dir / f"figscene_{scene.id:03d}.mp4"
    bg_png, layer_png, (bx, by, bw, bh), (fx, fy) = _compose_layers(
        scene, fig_dir, width, height, out_dir)

    dur = max(4.0, duration_sec) + 0.8
    frames = int(dur * fps)
    spec = scene.visual_spec or {}
    caption = str(spec.get("caption", "")).replace(":", "\\:").replace("'", "\\'")
    fig_id = str(spec.get("figure_id", ""))
    manifest = _load_manifest(fig_dir)
    page = next((f["page"] for f in manifest if f["id"] == fig_id), 0)
    tag = f"Figure {fig_id.replace('fig', '')} · p.{page}".replace(":", "\\:")
    font = _find_font()

    # Zoom ONLY the figure layer; the layer is always scaled back to the figure
    # box, so the figure can never leave the box (caption strip stays clean).
    zoom_expr = "'min(1+0.0016*on,1.14)'"
    chain = (
        f"[1:v]zoompan=z={zoom_expr}"
        f":x='(iw-iw/zoom)*{fx:.4f}':y='(ih-ih/zoom)*{fy:.4f}'"
        f":d={frames}:s={bw}x{bh}:fps={fps}[fig];"
        f"[0:v][fig]overlay={bx}:{by}:shortest=1[comp];"
        f"[comp]drawtext=fontfile={font}:text='{tag}'"
        f":x=w-tw-28:y=10:fontsize=24:fontcolor=0xE8E8EF"
        f":box=1:boxcolor=0x000000@0.55:boxborderw=8"
    )
    if caption:
        chain += (
            f",drawtext=fontfile={font}:text='{caption}'"
            f":x=(w-text_w)/2:y=h-92:fontsize=34:fontcolor=0xFFD666"
            f":box=1:boxcolor=0x000000@0.75:boxborderw=12"
        )
    chain += ",fade=t=in:st=0:d=0.45,format=yuv420p[v]"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(fps), "-i", str(bg_png),
        "-loop", "1", "-framerate", str(fps), "-i", str(layer_png),
        "-filter_complex", chain,
        "-map", "[v]",
        "-frames:v", str(frames),
        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not out_mp4.exists():
        raise FigureRenderError(f"ffmpeg failed: {proc.stderr[-800:]}")
    return out_mp4
