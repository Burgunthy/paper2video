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


def _scene(nid: int = 1) -> Scene:
    return Scene(id=nid, narration="테스트 내레이션", visual_type="slide",
                 visual_spec={"title": "t"}, duration_hint_sec=3.0)


def test_scene_audio_reuses_cached_wav(tmp_path):
    """같은 run-id 재실행 시 이미 합성된 장면 오디오는 건너뛴다."""
    calls = []

    class CountingTTS(FakeTTS):
        def synthesize(self, text, out_path):
            calls.append(text)
            return super().synthesize(text, out_path)

    scene = _scene()
    first = synthesize_scene_audio(scene, CountingTTS(), tmp_path)
    second = synthesize_scene_audio(scene, CountingTTS(), tmp_path)
    assert len(calls) == 1
    assert first.audio_path == second.audio_path
    assert first.duration_sec == second.duration_sec


def test_scene_audio_skips_partial_wav(tmp_path):
    """크래시로 잘린 wav는 무효 — 재합성한다."""
    scene = _scene()
    (tmp_path / "scene_001.wav").write_bytes(b"RIFF\x00\x00")  # 깨진 헤더
    result = synthesize_scene_audio(scene, FakeTTS(), tmp_path)
    assert result.duration_sec > 0


def test_scene_audio_retries_transient_failure(tmp_path):
    """일시 엔진 실패는 재시도로 회복 — 한 번의 실패로 전체가 죽지 않는다."""
    class FlakyTTS(FakeTTS):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def synthesize(self, text, out_path):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("connection refused")
            return super().synthesize(text, out_path)

    engine = FlakyTTS()
    result = synthesize_scene_audio(_scene(), engine, tmp_path, retry_waits=(0, 0, 0))
    assert engine.calls == 3
    assert result.duration_sec > 0


def test_scene_audio_raises_after_exhausted_retries(tmp_path):
    """계속 실패하면 마지막 예외를 그대로 올린다."""
    class DeadTTS(FakeTTS):
        def synthesize(self, text, out_path):
            raise RuntimeError("connection refused")

    import pytest
    with pytest.raises(RuntimeError, match="connection refused"):
        synthesize_scene_audio(_scene(), DeadTTS(), tmp_path, retry_waits=(0, 0))
    assert not (tmp_path / "scene_001.wav").exists()  # 부분 출력 정리됨


def test_http_tts_instruct_forwarded(tmp_path, monkeypatch):
    """PAPER2VIDEO_TTS_INSTRUCT가 지정되면 payload에 instruct가 실린다."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from paper2video.tts import HTTPTTS

    seen: dict = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            seen["payload"] = _json.loads(self.rfile.read(n))
            body = b"RIFF-fake-wav-bytes"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    monkeypatch.setenv("PAPER2VIDEO_TTS_INSTRUCT", "빠르고 에너지 있는 톤으로")
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        engine = HTTPTTS(url=f"http://127.0.0.1:{srv.server_port}/tts", voice="Sohee")
        engine.synthesize("안녕하세요", tmp_path / "s.wav")
        assert seen["payload"]["instruct"] == "빠르고 에너지 있는 톤으로"
        assert seen["payload"]["voice"] == "Sohee"
    finally:
        srv.shutdown()
