"""Text-to-Speech: cross-platform backends with auto-detection."""

import asyncio
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from config import TTSConfig

EDGE_VOICE_MAP = {
    "zosia": "pl-PL-ZofiaNeural",
    "zofia": "pl-PL-ZofiaNeural",
    "agnieszka": "pl-PL-AgnieszkaNeural",
    "marek": "pl-PL-MarekNeural",
}


def _pyttsx3_available() -> bool:
    try:
        import pyttsx3  # noqa: F401

        return True
    except ImportError:
        return False


def _edge_available() -> bool:
    try:
        import edge_tts  # noqa: F401

        return True
    except ImportError:
        return False


def _detect_tts_backend() -> str:
    """Pick the best available TTS backend for the current platform."""
    if sys.platform == "darwin":
        return "macos"
    if _pyttsx3_available():
        return "pyttsx3"
    if _edge_available():
        return "edge"
    raise RuntimeError(
        "Brak dostępnego backendu TTS. Zainstaluj: pip install pyttsx3 edge-tts"
    )


def _resolve_edge_voice(voice: str) -> str:
    if voice.startswith("pl-PL-"):
        return voice
    return EDGE_VOICE_MAP.get(voice.lower(), "pl-PL-ZofiaNeural")


class MacSayTTS:
    """Built-in macOS TTS — no download, works offline."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.voice = config.voice
        self._sample_rate = 22050

    def load_model(self):
        print(f"[TTS] Używam macOS say (głos: {self.voice})")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            path = tmp.name

        try:
            subprocess.run(
                ["say", "-v", self.voice, "-o", path, text],
                check=True,
                capture_output=True,
            )
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio, sr
        finally:
            Path(path).unlink(missing_ok=True)

    def synthesize_to_file(self, text: str, output_path: str):
        audio, sr = self.synthesize(text)
        sf.write(output_path, audio, sr)
        return output_path

    def synthesize_streaming(self, text: str, chunk_size: int = 4800):
        audio, sr = self.synthesize(text)
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size], sr


class Pyttsx3TTS:
    """Offline TTS via pyttsx3 — Windows/Linux/macOS system voices."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.voice = config.voice
        self._sample_rate = config.sample_rate

    def load_model(self):
        print(f"[TTS] Używam pyttsx3 ({platform.system()}, głos: {self.voice})")

    def _create_engine(self):
        import pyttsx3

        engine = pyttsx3.init()
        for voice in engine.getProperty("voices"):
            name = (voice.name or "").lower()
            vid = (voice.id or "").lower()
            target = self.voice.lower()
            if target in name or target in vid:
                engine.setProperty("voice", voice.id)
                break
        return engine

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name

        try:
            engine = self._create_engine()
            engine.save_to_file(text, path)
            engine.runAndWait()
            engine.stop()
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio, sr
        finally:
            Path(path).unlink(missing_ok=True)

    def synthesize_to_file(self, text: str, output_path: str):
        audio, sr = self.synthesize(text)
        sf.write(output_path, audio, sr)
        return output_path

    def synthesize_streaming(self, text: str, chunk_size: int = 4800):
        audio, sr = self.synthesize(text)
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size], sr


class EdgeTTS:
    """Microsoft Edge TTS — free, cross-platform, good Polish voices (requires network)."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.voice = _resolve_edge_voice(config.voice)
        self._sample_rate = config.sample_rate

    def load_model(self):
        print(f"[TTS] Używam edge-tts (głos: {self.voice})")

    async def _synthesize_async(self, text: str, output_path: str):
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        import librosa

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name

        try:
            asyncio.run(self._synthesize_async(text, path))
            audio, sr = librosa.load(path, sr=None, mono=True)
            return audio.astype(np.float32), sr
        finally:
            Path(path).unlink(missing_ok=True)

    def synthesize_to_file(self, text: str, output_path: str):
        audio, sr = self.synthesize(text)
        sf.write(output_path, audio, sr)
        return output_path

    def synthesize_streaming(self, text: str, chunk_size: int = 4800):
        audio, sr = self.synthesize(text)
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size], sr


class ChatterboxTTS:
    """Chatterbox TTS — requires ~2 GB model download from HuggingFace."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.model = None
        self.device = config.device

    def load_model(self):
        print("[TTS] Ładowanie modelu Chatterbox (pobieranie ~2 GB)...")
        import torch
        from chatterbox.tts import ChatterboxTTS as CBModel

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = CBModel.from_pretrained(device=self.device)
        print(f"[TTS] Chatterbox załadowany na: {self.device}")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not self.model:
            self.load_model()

        import torch

        with torch.no_grad():
            wav = self.model.generate(text)

        if isinstance(wav, torch.Tensor):
            audio_np = wav.squeeze().cpu().numpy()
        else:
            audio_np = np.array(wav).squeeze()

        if audio_np.max() > 1.0:
            audio_np = audio_np / np.abs(audio_np).max()

        return audio_np, self.config.sample_rate

    def synthesize_to_file(self, text: str, output_path: str):
        audio, sr = self.synthesize(text)
        sf.write(output_path, audio, sr)
        return output_path

    def synthesize_streaming(self, text: str, chunk_size: int = 4800):
        audio, sr = self.synthesize(text)
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size], sr


def create_tts(config: TTSConfig):
    """Create TTS engine based on config.backend."""
    backend = config.backend
    if backend == "auto":
        backend = _detect_tts_backend()

    if backend == "chatterbox":
        return ChatterboxTTS(config)
    if backend == "macos":
        if sys.platform != "darwin":
            raise RuntimeError("Backend 'macos' wymaga systemu macOS.")
        return MacSayTTS(config)
    if backend == "pyttsx3":
        if not _pyttsx3_available():
            raise RuntimeError("Backend 'pyttsx3' wymaga: pip install pyttsx3")
        return Pyttsx3TTS(config)
    if backend == "edge":
        if not _edge_available():
            raise RuntimeError("Backend 'edge' wymaga: pip install edge-tts")
        return EdgeTTS(config)

    raise ValueError(
        f"Nieznany backend TTS: {config.backend!r}. "
        "Dozwolone: auto, macos, pyttsx3, edge, chatterbox"
    )
