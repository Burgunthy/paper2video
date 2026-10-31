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
    engine.synthesize(scene.narration, path)
    return SceneAudio(scene_id=scene.id, audio_path=path, duration_sec=_wav_duration(path))
