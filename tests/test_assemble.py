import subprocess
from pathlib import Path
from paper2video.assemble import build_concat_list, assemble_video, _ffmpeg, _ffprobe_duration
from paper2video.tts import FakeTTS, synthesize_scene_audio
from paper2video.renderers.slide import render_slide
from paper2video.types import Scene


def test_build_concat_list_format(tmp_path):
    lines = build_concat_list([tmp_path / "a.mp4", tmp_path / "b.mp4"])
    assert "file '" in lines
    assert "a.mp4" in lines
    assert "b.mp4" in lines


def test_assemble_video_produces_mp4(tmp_path):
    scenes = [
        Scene(id=1, narration="first scene", visual_type="slide",
              visual_spec={"title": "One", "bullets": ["a"]}, duration_hint_sec=3.0),
        Scene(id=2, narration="second scene", visual_type="slide",
              visual_spec={"title": "Two", "bullets": ["b"]}, duration_hint_sec=3.0),
    ]
    img_dir = tmp_path / "img"
    images = [render_slide(s, img_dir) for s in scenes]
    audio_dir = tmp_path / "audio"
    engine = FakeTTS()
    audios = [synthesize_scene_audio(s, engine, audio_dir) for s in scenes]
    durations = [a.duration_sec for a in audios]

    out = assemble_video(
        images=images,
        audio_paths=[a.audio_path for a in audios],
        durations=durations,
        out_path=tmp_path / "final.mp4",
        work_dir=tmp_path / "work",
    )
    assert out.exists()
    assert out.stat().st_size > 0
    total = _ffprobe_duration(out)
    if total is not None:
        assert total > sum(durations) - 0.5


def _make_test_clip(path: Path, w: int = 1280, h: int = 720, dur: float = 4.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c=gray:s={w}x{h}:r=30:d={dur}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def _make_test_wav(path: Path, dur: float = 3.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono:d={dur}",
         "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return path


def test_should_stream_copy_decision(tmp_path):
    from paper2video.assemble import should_stream_copy

    clip = _make_test_clip(tmp_path / "match.mp4")
    wav = _make_test_wav(tmp_path / "a.wav", dur=3.0)

    assert should_stream_copy(clip, wav, width=1280, height=720, fps=30) is True
    # Wrong target resolution -> must re-encode
    assert should_stream_copy(clip, wav, width=1920, height=1080, fps=30) is False
    # Video shorter than the audio duration -> needs tpad, must re-encode
    short = _make_test_clip(tmp_path / "short.mp4", dur=1.0)
    assert should_stream_copy(short, wav, width=1280, height=720, fps=30) is False


def test_mux_stream_copies_when_params_match(tmp_path, monkeypatch):
    from paper2video import assemble

    clip = _make_test_clip(tmp_path / "v.mp4")
    wav = _make_test_wav(tmp_path / "a.wav", dur=3.0)

    captured: dict = {}
    real_run = subprocess.run

    def spy_run(cmd, *a, **kw):
        if cmd and str(cmd[0]).endswith("ffmpeg"):
            captured["cmd"] = cmd
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(assemble.subprocess, "run", spy_run)
    out = assemble.mux_scene_clip(
        video_path=clip, audio_path=wav, duration_sec=3.0,
        out_path=tmp_path / "o.mp4", width=1280, height=720, fps=30,
    )

    cmd = captured["cmd"]
    i = cmd.index("-c:v")
    assert cmd[i + 1] == "copy", f"expected stream copy for matching clip, got: {cmd}"
    # Output trimmed to the audio duration
    total = _ffprobe_duration(out)
    assert total is not None and abs(total - 3.0) < 0.3


def test_mux_reencodes_when_params_mismatch(tmp_path, monkeypatch):
    from paper2video import assemble

    clip = _make_test_clip(tmp_path / "small.mp4", w=640, h=360)
    wav = _make_test_wav(tmp_path / "a.wav", dur=3.0)

    captured: dict = {}
    real_run = subprocess.run

    def spy_run(cmd, *a, **kw):
        if cmd and str(cmd[0]).endswith("ffmpeg"):
            captured["cmd"] = cmd
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(assemble.subprocess, "run", spy_run)
    assemble.mux_scene_clip(
        video_path=clip, audio_path=wav, duration_sec=3.0,
        out_path=tmp_path / "o.mp4", width=1280, height=720, fps=30,
    )

    cmd = captured["cmd"]
    i = cmd.index("-c:v")
    assert cmd[i + 1] == "libx264", f"expected re-encode for mismatched clip, got: {cmd}"


def test_concat_clips_sets_high_quality_crf(tmp_path, monkeypatch):
    from paper2video import assemble

    clip = _make_test_clip(tmp_path / "c1.mp4")
    captured: dict = {}
    real_run = subprocess.run

    def spy_run(cmd, *a, **kw):
        if cmd and str(cmd[0]).endswith("ffmpeg"):
            captured["cmd"] = cmd
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(assemble.subprocess, "run", spy_run)
    assemble.concat_clips([clip], tmp_path / "cat.mp4", tmp_path / "w")

    cmd = captured["cmd"]
    i = cmd.index("-crf")
    assert int(cmd[i + 1]) <= 18, f"concat must encode at crf<=18, got: {cmd}"
