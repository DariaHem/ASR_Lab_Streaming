"""Audio I/O handling for microphone capture and speaker playback."""

import queue
import threading
import numpy as np

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

from config import AudioConfig


class AudioIO:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.pa = None
        self.input_stream = None
        self.output_stream = None
        self._recording = False
        self._paused = False
        self.audio_queue: queue.Queue[bytes] = queue.Queue()

    def start_recording(self) -> str:
        """Start mic capture. Returns human-readable device name."""
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError(
                "PyAudio nie jest dostępne — tryb live wymaga mikrofonu.\n"
                "Zainstaluj PyAudio (patrz README: sekcja Windows/Linux/macOS) "
                "lub użyj trybu laboratoryjnego bez mikrofonu:\n"
                "  python main.py text\n"
                "  python labs/asr_eval.py --audio-dir labs/test_audio --references labs/test_audio/references.csv"
            )

        self.pa = pyaudio.PyAudio()
        default = self.pa.get_default_input_device_info()
        device_name = default.get("name", "nieznane")
        device_index = int(default.get("index", -1))
        print(f"[Audio] Mikrofon: {device_name} (index={device_index})")

        self.input_stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            input_device_index=device_index if device_index >= 0 else None,
            frames_per_buffer=self.config.chunk_size,
            stream_callback=self._audio_callback,
        )
        self._recording = True
        self._paused = False
        self._drain_queue()
        self.input_stream.start_stream()
        print("[Audio] Nagrywanie rozpoczęte...")
        return device_name

    def stop_recording(self):
        self._recording = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.pa:
            self.pa.terminate()
        print("[Audio] Nagrywanie zatrzymane.")

    def pause_recording(self):
        """Pause mic capture (e.g. while agent speaks via TTS)."""
        self._paused = True
        self._drain_queue()

    def resume_recording(self):
        """Resume mic capture after TTS."""
        self._paused = False
        self._drain_queue()

    def _drain_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self._recording and not self._paused:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def play_audio(self, audio: np.ndarray, sample_rate: int):
        """Play audio array through speakers."""
        if not PYAUDIO_AVAILABLE:
            print("[Audio] PyAudio niedostępne - pomijam odtwarzanie.")
            return

        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = (audio * 32767).astype(np.int16)

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
        )
        stream.write(audio.tobytes())
        stream.stop_stream()
        stream.close()
        pa.terminate()

    def get_audio_chunk(self, timeout: float = 1.0) -> bytes | None:
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_paused(self) -> bool:
        return self._paused
