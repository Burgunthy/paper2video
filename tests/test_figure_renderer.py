"""Figure-scene renderer: zoom containment + caption readability + encode quality."""
import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from paper2video.renderers.figure_r import render_figure_scene
from paper2video.types import Scene


def _make_fig_dir(tmp_path: Path) -> Path:
    """A manifest with one TALL white figure (worst case for vertical overflow)."""
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    img = Image.new("RGB", (800, 1100), (255, 255, 255))
    img.save(fig_dir / "fig1.png")
    manifest = [{
        "id": "fig1", "number": 1, "page": 3,
        "caption": "A tall white test figure", "path": "fig1.png",
        "width_px": 800, "height_px": 1100,
    }]
    (fig_dir / "figures.json").write_text(json.dumps(manifest), encoding="utf-8")
    return fig_dir


def _scene(caption: str = "테스트 캡션") -> Scene:
    return Scene(
        id=1, narration="narration text", visual_type="figure",
        visual_spec={"figure_id": "fig1", "focus": "full", "highlight": "", "caption": caption},
        duration_hint_sec=5.0,
    )


def _frame_at(mp4: Path, t: float) -> Image.Image:
    out = mp4.parent / f"probe_{t:.2f}.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(mp4),
         "-frames:v", "1", str(out)],
        check=True,
    )
    return Image.open(out).convert("RGB")


def _white_fraction(img: Image.Image, y0: int, y1: int) -> float:
    """Fraction of near-white pixels in a row band (x 320..1240).

    Caption glyphs are thin strokes (~<15% of a band); a solid white figure
    block invading the band reads ~100%. Threshold separates the two.
    """
    px = img.load()
    n = tot = 0
    for y in range(y0, y1):
        for x in range(320, 1240, 2):
            r, g, b = px[x, y]
            tot += 1
            if r > 200 and g > 200 and b > 200:
                n += 1
    return n / tot


def test_zoom_never_invades_overlay_zone(tmp_path):
    """The animated zoom must stay inside the figure box: at MAX zoom the
    solid-white figure must not show as a dense block below y=620. White
    caption GLYPHS are expected in the strip — density tells them apart."""
    fig_dir = _make_fig_dir(tmp_path)
    out_dir = tmp_path / "figure_out"
    mp4 = render_figure_scene(_scene(), duration_sec=5.0, fig_dir=fig_dir, out_dir=out_dir)

    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(mp4)]))
    # zoom reaches its cap (~1.14) before 3s; check both mid and last frame
    for t in (dur * 0.6, dur - 0.15):
        frame = _frame_at(mp4, t)
        assert frame.size == (1280, 720)
        for y0 in range(622, 716, 8):
            frac = _white_fraction(frame, y0, min(y0 + 8, 716))
            assert frac <= 0.45, \
                f"dense white block at y={y0} (frac={frac:.2f}, t={t}) — figure invaded the overlay zone"


def test_caption_band_free_of_figure_pixels(tmp_path):
    """The caption band must hold only the (white) caption glyphs on their
    opaque box — never dense figure content behind them."""
    fig_dir = _make_fig_dir(tmp_path)
    out_dir = tmp_path / "figure_out"
    mp4 = render_figure_scene(_scene(), duration_sec=5.0, fig_dir=fig_dir, out_dir=out_dir)

    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(mp4)]))
    frame = _frame_at(mp4, dur - 0.15)
    frac = _white_fraction(frame, 626, 670)
    assert frac <= 0.45, f"dense content in caption band (frac={frac:.2f})"


def test_figure_scene_high_quality_encode(tmp_path, monkeypatch):
    """The figure scene itself must be encoded at high quality (crf<=18) since
    it is the first generation of the chain."""
    fig_dir = _make_fig_dir(tmp_path)
    captured: dict = {}

    class FakeProc:
        returncode = 0
        stderr = ""

    def fake_run(cmd, *a, **kw):
        captured["cmd"] = cmd
        out = cmd[cmd.index(str(cmd[-1]))]
        Path(out).write_bytes(b"x")  # existence stub
        return FakeProc()

    monkeypatch.setattr("paper2video.renderers.figure_r.subprocess.run", fake_run)
    render_figure_scene(_scene(), duration_sec=5.0, fig_dir=fig_dir,
                        out_dir=tmp_path / "out2")
    cmd = captured["cmd"]
    assert "-crf" in cmd
    assert int(cmd[cmd.index("-crf") + 1]) <= 18


def test_no_highlight_box_lines(tmp_path):
    """Fractional highlight boxes don't match real panel boundaries and read
    as arbitrary vertical/horizontal lines — the renderer must not draw them."""
    fig_dir = _make_fig_dir(tmp_path)
    out_dir = tmp_path / "figure_out"
    scene = Scene(
        id=2, narration="n", visual_type="figure",
        visual_spec={"figure_id": "fig1", "focus": "left", "highlight": "left", "caption": "캡"},
        duration_hint_sec=5.0,
    )
    from paper2video.renderers.figure_r import _compose_layers
    _compose_layers(scene, fig_dir, 1280, 720, out_dir)
    from PIL import Image
    import numpy as np
    layer = np.array(Image.open(out_dir / "figscene_002_layer.png").convert("RGB"))
    amber = (np.abs(layer.astype(int) - np.array([255, 201, 64])).sum(axis=2) < 90)
    assert amber.sum() == 0, f"highlight lines drawn: {amber.sum()} amber px"

def test_caption_is_white(tmp_path):
    """Viewer preference: captions white, not amber."""
    fig_dir = _make_fig_dir(tmp_path)
    out_dir = tmp_path / "figure_out"
    mp4 = render_figure_scene(_scene(), duration_sec=5.0, fig_dir=fig_dir, out_dir=out_dir)
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(mp4)]))
    frame = _frame_at(mp4, dur - 0.15)
    px = frame.load()
    # caption glyph pixels: near-white, NOT amber (255,214,102)
    white_glyphs = any(
        r > 230 and g > 230 and b > 225
        for y in range(626, 670) for x in range(340, 1220, 2)
        for r, g, b in [px[x, y]]
    )
    amber_glyphs = any(
        r > 230 and 190 < g < 228 and b < 150
        for y in range(626, 670) for x in range(340, 1220, 2)
        for r, g, b in [px[x, y]]
    )
    assert white_glyphs and not amber_glyphs, "caption not white"


def test_no_zoom_animation(tmp_path, monkeypatch):
    """Zoom crops figure content (edge panels at 1.14x) — figure scenes must be
    static: no zoompan in the ffmpeg filter."""
    fig_dir = _make_fig_dir(tmp_path)
    captured: dict = {}

    class FakeProc:
        returncode = 0
        stderr = ""

    def fake_run(cmd, *a, **kw):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"x")
        return FakeProc()

    monkeypatch.setattr("paper2video.renderers.figure_r.subprocess.run", fake_run)
    render_figure_scene(_scene(), duration_sec=5.0, fig_dir=fig_dir,
                        out_dir=tmp_path / "o1")
    assert all("zoompan" not in str(arg) for arg in captured["cmd"]), \
        "figure scene must render static (no zoompan)"


def _make_three_panel_fig(path: Path):
    """White canvas, 3 dark blocks with clear vertical gutters (like fig2)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (900, 300), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i in range(3):
        x0 = 30 + i * 300
        d.rectangle((x0, 30, x0 + 240, 270), fill=(40, 40, 40))
    img.save(path)


def test_region_crop_snaps_to_panel_gutters(tmp_path):
    """Fractional crops (left=0-50%) cut multi-panel figures mid-panel.
    The crop must snap to detected gutter boundaries: on a 3-panel figure,
    left -> panel 1, center -> panel 2, right -> panel 3."""
    from paper2video.renderers.figure_r import snap_crop_x
    import numpy as np
    from PIL import Image

    fig_dir = _make_fig_dir(tmp_path)
    _make_three_panel_fig(fig_dir / "fig1.png")
    arr = np.array(Image.open(fig_dir / "fig1.png").convert("L"))

    x0, x1, ok = snap_crop_x(arr, 0.0, 0.5)   # left
    assert ok and abs(x1 - 1 / 3) < 0.05, f"left crop should end at panel boundary, got {x1:.3f}"
    x0, x1, ok = snap_crop_x(arr, 0.25, 0.75)  # center
    assert ok and abs(x0 - 1 / 3) < 0.05 and abs(x1 - 2 / 3) < 0.05, \
        f"center crop should be the middle panel, got {x0:.3f}-{x1:.3f}"


def test_no_gutters_means_full_view(tmp_path):
    """Gutter-less figures (dense architecture diagrams) must NOT get a
    fractional crop — both crop edges must snap to detected boundaries or the
    scene falls back to the FULL figure. A mid-panel cut is worse than no cut."""
    import numpy as np
    from PIL import Image
    from paper2video.renderers.figure_r import snap_crop_x

    # Dense figure: ink in every column (no whitespace gutters at all)
    arr = (np.random.default_rng(0).random((300, 900)) * 100).astype("uint8")
    x0, x1, snapped = snap_crop_x(arr, 0.25, 0.75)
    assert snapped is False
    # And the composer must treat that as full-view (no crop):
    from paper2video.renderers.figure_r import _compose_layers
    fig_dir = _make_fig_dir(tmp_path)
    img = Image.fromarray(arr.astype("uint8")).convert("RGB")
    img.save(fig_dir / "fig1.png")
    scene = Scene(id=3, narration="n", visual_type="figure",
                  visual_spec={"figure_id": "fig1", "focus": "center", "highlight": "", "caption": "캡"},
                  duration_hint_sec=5.0)
    _, layer_png, _, _ = _compose_layers(scene, fig_dir, 1280, 720, tmp_path / "o")
    layer = Image.open(layer_png)
    # full figure (landscape 3:1) in a landscape box -> layer should be wide,
    # not a narrow center-crop sliver
    assert layer.width / layer.height > 1.5, \
        f"expected full-figure view, got narrow crop {layer.size}"
