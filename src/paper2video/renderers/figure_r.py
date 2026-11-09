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

# 캡션/출처 태그는 상단(y=10, fontsize 26, box 여백 10)에 그려진다 — 텍스트 하단이
# y≈61까지 내려온다. 그림 박스는 캡션이 있을 때 이만큼 더 아래에서 시작해야
# 캡션 박스가 그림 상단을 덮지 않는다(2026-11-10 QA: "그림/글 겹침").
CAPTION_BAND_H = 40


def snap_crop_x(arr, x0: float, x1: float, tol: float = 0.20) -> tuple[float, float, bool]:
    """Snap a fractional horizontal crop to real panel boundaries.

    Multi-panel figures have vertical whitespace gutters; fractional crops
    (left = 0-50%) cut panels mid-content. Boundaries = centers of contiguous
    near-uniform column runs; each crop edge snaps to the nearest boundary
    within `tol`. Returns (x0, x1, both_snapped) — when either edge cannot
    snap to a detected boundary there are no reliable panel lines, and the
    caller must fall back to the FULL figure (a mid-panel cut is worse).
    """
    import numpy as np

    h, w = arr.shape
    col_range = arr.max(axis=0).astype(int) - arr.min(axis=0).astype(int)
    gutter = col_range < 12
    bounds: list[float] = []
    run_start = None
    for x in range(w):
        if gutter[x]:
            if run_start is None:
                run_start = x
        elif run_start is not None:
            if x - run_start >= max(6, int(w * 0.015)):
                bounds.append(((run_start + x) / 2) / w)
            run_start = None
    if run_start is not None and w - run_start >= max(6, int(w * 0.015)):
        bounds.append(((run_start + w) / 2) / w)
    if not bounds:
        return x0, x1, False

    def snap(v: float) -> float | None:
        if v <= 0.02 or v >= 0.98:
            # 요청 자체가 그림 가장자리 — 내부 거터로 스냅하면 안 된다
            # (하단 내용이 끝까지 차 있는 그림에서 y1=1.0이 스냅 실패하던 버그).
            return v
        best = min(bounds, key=lambda b: abs(b - v))
        return best if abs(best - v) <= tol else None

    s0, s1 = snap(x0), snap(x1)
    if s0 is None or s1 is None:
        return x0, x1, False
    return s0, s1, True


def snap_crop_y(arr, y0: float, y1: float, tol: float = 0.20) -> tuple[float, float, bool]:
    """Same as snap_crop_x for horizontally stacked panels (row gutters)."""
    import numpy as np

    rotated = np.asarray(arr).T
    return snap_crop_x(rotated, y0, y1, tol)


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

    # Focus crop of the source figure, snapped to real panel gutters so a
    # "left" crop of a 3-panel figure is exactly panel 1 (not 0-50%, which
    # cuts the middle panel in half).
    fx0, fy0, fx1, fy1 = _region(spec.get("focus"))
    if (fx0, fy0, fx1, fy1) != REGIONS["full"]:
        import numpy as np

        arr = np.array(fig.convert("L"))
        w_img, h_img = fig.size
        # 범위가 이미 전체(0..1)인 축은 자를 필요가 없다 — 스냅도 불요.
        # (내부 거터만 있는 그림에서 "bottom"(전체 폭)이 x축 스냅 실패로
        # full 폴백하던 버그: 2026-11-10 재생성 검증에서 발견)
        want_x = not (fx0 <= 0.0 and fx1 >= 1.0)
        want_y = not (fy0 <= 0.0 and fy1 >= 1.0)
        sx0, sx1, ok_x = (snap_crop_x(arr, fx0, fx1) if want_x else (fx0, fx1, True))
        sy0, sy1, ok_y = (snap_crop_y(arr, fy0, fy1) if want_y else (fy0, fy1, True))
        if ok_x and ok_y:
            fig = fig.crop((int(w_img * sx0), int(h_img * sy0),
                            int(w_img * sx1), int(h_img * sy1)))
        # else: no reliable panel boundaries — keep the FULL figure. A crop
        # that cuts panels mid-content looks broken; full view never does.

    # Figure box on the final canvas — must NEVER overlap the caption strip
    # (bottom) nor the caption/tag text band (top, y≈10..61 when caption exists).
    has_caption = bool(str((scene.visual_spec or {}).get("caption", "")).strip())
    top_margin = MARGIN_TOP + (CAPTION_BAND_H if has_caption else 0)
    avail_w = width - 2 * MARGIN_X
    avail_h = height - top_margin - CAPTION_STRIP_H - 16
    scale = min(avail_w / fig.width, avail_h / fig.height)
    box_w = max(2, int(fig.width * scale)) & ~1   # even dims for yuv420p safety
    box_h = max(2, int(fig.height * scale)) & ~1
    box_x = (width - box_w) // 2
    box_y = top_margin + max(0, (avail_h - box_h) // 2)

    bg = Image.new("RGB", (width, height), BG)

    # Figure layer at box size. Scenes render STATIC (no zoompan): a zoom
    # push-in always crops edge content — on multi-panel figures the outer
    # panels get cut mid-panel ("애매하게 잘림"). Static + panel-snapped crops
    # never cut content.
    layer = Image.new("RGB", (box_w, box_h), BG)
    s2 = min(layer.width / fig.width, layer.height / fig.height)
    fw, fh = max(1, int(fig.width * s2)), max(1, int(fig.height * s2))
    fig_scaled = fig.resize((fw, fh), Image.LANCZOS)
    fx_off = (layer.width - fw) // 2
    fy_off = (layer.height - fh) // 2
    layer.paste(fig_scaled, (fx_off, fy_off))

    # NOTE: no highlight boxes (fractional outlines read as arbitrary lines),
    # no zoom (crops edge panels).

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

    # Static composition — no zoompan. The figure layer is pasted at its box;
    # a zoom would crop edge panels on multi-panel figures.
    chain = (
        f"[0:v][1:v]overlay={bx}:{by}:shortest=1[comp];"
        f"[comp]drawtext=fontfile={font}:text='{tag}'"
        f":x=w-tw-28:y=10:fontsize=24:fontcolor=0xE8E8EF"
        f":box=1:boxcolor=0x000000@0.55:boxborderw=8"
    )
    if caption:
        # 상단 좌측 — 하단 스트립은 큰 내레이션 자막의 영역
        chain += (
            f",drawtext=fontfile={font}:text='{caption}'"
            f":x=28:y=10:fontsize=26:fontcolor=0xFFFFFF"
            f":box=1:boxcolor=0x000000@0.55:boxborderw=10"
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
