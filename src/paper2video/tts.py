from __future__ import annotations
import asyncio
import os
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from .types import Scene


class TTSEngine(Protocol):
    sample_rate: int
    def synthesize(self, text: str, out_path: Path) -> Path: ...


@dataclass
class SceneAudio:
    scene_id: int
    audio_path: Path
    duration_sec: float


class FakeTTS:
    """Silent WAV generator for tests. Duration scales with text length."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def synthesize(self, text: str, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        duration_sec = max(0.5, len(text) / 40.0)
        n_frames = int(duration_sec * self.sample_rate)
        silence = struct.pack("<h", 0) * n_frames
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(silence)
        return out_path


class KokoroTTS:
    """Kokoro-82M (Apache 2.0, ~4.2 MOS). Lazy-imports torch/kokoro."""

    def __init__(self, voice: str = "af_heart", lang_code: str = "a"):
        from kokoro import KPipeline  # type: ignore
        self._pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.sample_rate = 24000

    def synthesize(self, text: str, out_path: Path) -> Path:
        import numpy as np
        import soundfile as sf  # type: ignore

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        chunks: list[np.ndarray] = []
        for _, _, audio in self._pipeline(text, voice=self.voice):
            chunks.append(audio)
        if not chunks:
            audio_full = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
        else:
            audio_full = np.concatenate(chunks).astype(np.float32)
        sf.write(str(out_path), audio_full, self.sample_rate)
        return out_path


# Korean TTS mispronounces common tech terms (SAM3 -> 삼삼, Ego4D -> 에고4디).
# Map them to the intended spoken form before synthesis. Longest-first so
# "Ego-Exo4D" wins over "Ego4D". Word-boundary regex keeps "4D" inside other
# tokens untouched.
_TTS_GLOSSARY: list[tuple[str, str]] = [
    ("Ego-Exo4D", "에고엑소포디"),
    ("Ego4D", "에고포디"),
    ("Ego-Exo", "에고엑소"),
    ("EPIC-KITCHENS", "에픽키친즈"),
    ("EPIC KITCHENS", "에픽키친즈"),
    ("Something-Something", "썸싱썸싱"),
    ("Depth Anything 3", "뎁스애니띵쓰리"),
    ("Depth Anything", "뎁스애니띵"),
    ("ZoeDepth", "조우뎁스"),
    ("SAM3", "샘쓰리"),
    ("SAM 3", "샘쓰리"),
    ("HaWoR", "하워"),
    ("WiLoR", "윌로"),
    ("VITRA", "비트라"),
    ("MANO", "마노"),
    ("ARCTIC", "아틱틱"),
    ("HOT3D", "핫쓰리디"),
    ("HACO", "하코"),
    ("DexYCB", "덱스와이씨비"),
]


def normalize_for_tts(text: str) -> str:
    import re

    out = text
    for term, spoken in sorted(_TTS_GLOSSARY, key=lambda t: -len(t[0])):
        out = re.sub(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", spoken, out)
    return out


class EdgeTTS:
    """Online TTS via Microsoft Edge voices (Korean etc.). Writes 24kHz mono WAV.

    Kokoro does not support Korean, so this engine covers ko (and any other
    edge-tts voice) at neural quality without an API key.
    """

    def __init__(self, voice: str | None = None):
        import edge_tts  # type: ignore
        self._edge_tts = edge_tts
        self.voice = voice or os.environ.get("PAPER2VIDEO_TTS_VOICE", "ko-KR-SunHiNeural")
        self.sample_rate = 24000

    def synthesize(self, text: str, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = Path(f.name)
        try:
            asyncio.run(self._edge_tts.Communicate(text, self.voice).save(str(tmp)))
            ffmpeg = os.environ.get("PAPER2VIDEO_FFMPEG", "ffmpeg")
            subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-i", str(tmp),
                 "-ar", str(self.sample_rate), "-ac", "1", "-sample_fmt", "s16",
                 str(out_path)],
                check=True, stdin=subprocess.DEVNULL,
            )
        finally:
            tmp.unlink(missing_ok=True)
        return out_path


def _wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / float(w.getframerate())


def synthesize_scene_audio(scene: Scene, engine: TTSEngine, out_dir: Path) -> SceneAudio:
    out_dir = Path(out_dir)
    path = out_dir / f"scene_{scene.id:03d}.wav"
    engine.synthesize(normalize_for_tts(scene.narration), path)
    return SceneAudio(scene_id=scene.id, audio_path=path, duration_sec=_wav_duration(path))
