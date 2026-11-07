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


def test_http_tts_roundtrip(tmp_path):
    """HTTPTTS POSTs {text, voice} and writes the returned WAV bytes."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from paper2video.tts import HTTPTTS

    seen: dict = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            seen["payload"] = _json.loads(self.rfile.read(n))
            seen["path"] = self.path
            body = b"RIFF-fake-wav-bytes"
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        engine = HTTPTTS(url=f"http://127.0.0.1:{srv.server_port}/tts", voice="Sohee")
        out = tmp_path / "s.wav"
        engine.synthesize("안녕하세요", out)
        assert out.read_bytes() == b"RIFF-fake-wav-bytes"
        assert seen["path"] == "/tts"
        assert seen["payload"] == {"text": "안녕하세요", "voice": "Sohee"}
    finally:
        srv.shutdown()


def test_http_tts_error_raises(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import pytest
    from paper2video.tts import HTTPTTS

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        engine = HTTPTTS(url=f"http://127.0.0.1:{srv.server_port}/tts")
        with pytest.raises(RuntimeError, match="500"):
            engine.synthesize("x", tmp_path / "s.wav")
    finally:
        srv.shutdown()
