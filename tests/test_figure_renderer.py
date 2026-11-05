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


def test_zoom_never_invades_overlay_zone(tmp_path):
    """The animated zoom must stay inside the figure box: at MAX zoom the figure
    must not render below the safe-area bottom edge (y=620), and the caption
    strip must show an opaque background rather than figure pixels."""
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
        px = frame.load()
        # Bottom band below the safe area (y>=622), excluding the tag zone on
        # the left (x<320) and caption glyphs themselves: no WHITE figure pixels.
        for y in range(622, 716, 2):
            for x in range(320, 1240, 4):
                r, g, b = px[x, y]
                assert not (r > 200 and g > 200 and b > 200), \
                    f"figure pixels leaked into overlay zone at (x={x}, y={y}, t={t})"


def test_caption_band_free_of_figure_pixels(tmp_path):
    """Rows where the amber caption text lives must contain the caption and its
    opaque box only — never raw figure content behind the glyphs."""
    fig_dir = _make_fig_dir(tmp_path)
    out_dir = tmp_path / "figure_out"
    mp4 = render_figure_scene(_scene(), duration_sec=5.0, fig_dir=fig_dir, out_dir=out_dir)

    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(mp4)]))
    frame = _frame_at(mp4, dur - 0.15)
    px = frame.load()
    for y in range(626, 670, 2):
        for x in range(340, 1220, 4):
            r, g, b = px[x, y]
            assert not (r > 200 and g > 200 and b > 200), \
                f"figure content visible in caption band at (x={x}, y={y})"


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
