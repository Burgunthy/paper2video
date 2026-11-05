import wave
from paper2video.tts import FakeTTS, synthesize_scene_audio
from paper2video.types import Scene


def test_fake_tts_writes_wav(tmp_path):
    engine = FakeTTS(sample_rate=22050)
    out = engine.synthesize("hello world", tmp_path / "a.wav")
    assert out.exists()
    with wave.open(str(out)) as w:
        assert w.getframerate() == 22050
        assert w.getnframes() > 0


def test_synthesize_scene_audio_returns_path_and_duration(tmp_path):
    engine = FakeTTS()
    scene = Scene(id=1, narration="hi there", visual_type="slide",
                  visual_spec={"title": "t"}, duration_hint_sec=3.0)
    result = synthesize_scene_audio(scene, engine, tmp_path)
    assert result.audio_path.exists()
    assert result.duration_sec > 0


def test_fake_tts_duration_proportional_to_text(tmp_path):
    engine = FakeTTS()
    short = engine.synthesize("hi", tmp_path / "short.wav")
    long = engine.synthesize("this is a much longer sentence with many words", tmp_path / "long.wav")
    with wave.open(str(short)) as s, wave.open(str(long)) as l:
        assert l.getnframes() > s.getnframes()


def test_tts_pronunciation_glossary():
    """Korean TTS mangles tech terms (SAM3 -> 삼삼, Ego4D -> 에고4디).
    Narration must be phonetically normalized before synthesis."""
    from paper2video.tts import normalize_for_tts

    src = "SAM3가 객체를 분할하고 Ego4D와 Ego-Exo4D, EPIC-KITCHENS를 씁니다. HaWoR 트래커, Depth Anything 3, VITRA 코퍼스."
    out = normalize_for_tts(src)
    assert "샘쓰리" in out and "SAM3" not in out
    assert "에고포디" in out and "Ego4D" not in out
    assert "에고엑소포디" in out and "Ego-Exo4D" not in out
    assert "에픽키친즈" in out
    assert "하워" in out and "HaWoR" not in out
    assert "뎁스애니띵쓰리" in out
    assert "비트라" in out
    # Unrelated digits/words untouched
    assert normalize_for_tts("오늘 500개를 3시간에") == "오늘 500개를 3시간에"
