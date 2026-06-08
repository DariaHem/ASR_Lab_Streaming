"""Streaming ASR using Faster-Whisper with Voice Activity Detection."""

import queue
from dataclasses import dataclass

import numpy as np
import librosa
from faster_whisper import WhisperModel
import webrtcvad

from config import ASRConfig, AudioConfig


@dataclass
class TranscriptionResult:
    text: str
    avg_logprob: float | None
    no_speech_prob: float | None
    segment_count: int

POLISH_MEDICAL_PROMPT = (
    "Rozmowa w recepcji placówki medycznej. Pacjent umawia wizytę u lekarza."
)


class StreamingASR:
    def __init__(self, asr_config: ASRConfig, audio_config: AudioConfig):
        self.config = asr_config
        self.audio_config = audio_config
        self.model = None
        self.vad = webrtcvad.Vad(2)  # aggressiveness 0-3
        self._audio_buffer = []
        self._is_speaking = False
        self._silence_frames = 0
        self._min_silence_frames = int(
            asr_config.min_silence_duration_ms / (30)  # 30ms per VAD frame
        )
        self.transcript_queue: queue.Queue[str] = queue.Queue()
        self._running = False

    def load_model(self):
        print(f"[ASR] Ładowanie modelu Whisper '{self.config.model_size}'...")
        self.model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        print("[ASR] Model załadowany.")

    def process_audio_chunk(self, audio_chunk: bytes) -> str | None:
        """Process a raw audio chunk (16-bit PCM, 16kHz mono).
        Returns transcription when a speech segment is complete."""
        frame_duration_ms = 30
        frame_size = int(self.audio_config.sample_rate * frame_duration_ms / 1000) * 2

        for i in range(0, len(audio_chunk) - frame_size + 1, frame_size):
            frame = audio_chunk[i : i + frame_size]
            if len(frame) < frame_size:
                break

            is_speech = self.vad.is_speech(frame, self.audio_config.sample_rate)

            if is_speech:
                if not self._is_speaking:
                    self._is_speaking = True
                    self._silence_frames = 0
                self._audio_buffer.append(frame)
            else:
                if self._is_speaking:
                    self._silence_frames += 1
                    self._audio_buffer.append(frame)

                    if self._silence_frames >= self._min_silence_frames:
                        transcript = self._transcribe_buffer()
                        self._audio_buffer = []
                        self._is_speaking = False
                        self._silence_frames = 0
                        if transcript:
                            return transcript
        return None

    def _transcribe_buffer(self) -> str | None:
        if not self._audio_buffer or not self.model:
            return None

        audio_bytes = b"".join(self._audio_buffer)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) < self.audio_config.sample_rate * 0.3:
            return None

        segments, info = self.model.transcribe(
            audio_np,
            language=self.config.language,
            beam_size=5,
            vad_filter=True,
        )

        text = " ".join(segment.text.strip() for segment in segments)
        return text if text.strip() else None

    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize and resample audio to 16 kHz mono float32 for Whisper."""
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sample_rate != self.audio_config.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.audio_config.sample_rate,
            )

        # Boost quiet captures so Whisper gets usable signal
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if 0 < peak < 0.12:
            audio = audio * (0.85 / peak)

        return audio

    def _run_transcribe(
        self,
        audio: np.ndarray,
        *,
        vad_filter: bool,
        beam_size: int | None = None,
    ) -> TranscriptionResult:
        segments, _ = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=beam_size if beam_size is not None else self.config.beam_size,
            vad_filter=vad_filter,
            initial_prompt=POLISH_MEDICAL_PROMPT,
            condition_on_previous_text=False,
        )
        seg_list = list(segments)
        text = " ".join(segment.text.strip() for segment in seg_list).strip()
        if not seg_list:
            return TranscriptionResult(
                text=text,
                avg_logprob=None,
                no_speech_prob=None,
                segment_count=0,
            )

        return TranscriptionResult(
            text=text,
            avg_logprob=sum(segment.avg_logprob for segment in seg_list) / len(seg_list),
            no_speech_prob=max(segment.no_speech_prob for segment in seg_list),
            segment_count=len(seg_list),
        )

    def transcribe_array(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a numpy audio buffer (any sample rate, mono/stereo)."""
        if not self.model:
            self.load_model()

        audio = self._prepare_audio(audio, sample_rate)
        min_samples = int(self.audio_config.sample_rate * 0.25)

        if len(audio) < min_samples:
            print(
                f"[ASR] Za krótkie audio: {len(audio)} próbek "
                f"({len(audio) / self.audio_config.sample_rate:.2f}s, min {min_samples / self.audio_config.sample_rate:.2f}s)"
            )
            return ""

        result = self._run_transcribe(audio, vad_filter=True)
        if not result.text:
            print("[ASR] Pusta transkrypcja z VAD — ponawiam beam=5 bez vad_filter")
            result = self._run_transcribe(audio, vad_filter=False, beam_size=5)
        return result.text

    def transcribe_array_detailed(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        allow_aggressive_retry: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio and return text with Whisper confidence metadata."""
        if not self.model:
            self.load_model()

        audio = self._prepare_audio(audio, sample_rate)
        min_samples = int(self.audio_config.sample_rate * 0.25)

        if len(audio) < min_samples:
            print(
                f"[ASR] Za krótkie audio: {len(audio)} próbek "
                f"({len(audio) / self.audio_config.sample_rate:.2f}s, min {min_samples / self.audio_config.sample_rate:.2f}s)"
            )
            return TranscriptionResult(text="", avg_logprob=None, no_speech_prob=None, segment_count=0)

        result = self._run_transcribe(audio, vad_filter=True)
        if not result.text and allow_aggressive_retry:
            print("[ASR] Pusta transkrypcja z VAD — ponawiam beam=5 bez vad_filter")
            result = self._run_transcribe(audio, vad_filter=False, beam_size=5)
        return result

    def transcribe_array_aggressive(self, audio: np.ndarray, sample_rate: int) -> str:
        """Last-resort transcription: no VAD, explicit beam search, padded audio."""
        if not self.model:
            self.load_model()

        audio = self._prepare_audio(audio, sample_rate)
        min_samples = int(self.audio_config.sample_rate * 0.25)
        if len(audio) < min_samples:
            pad = np.zeros(min_samples - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])

        print(
            f"[ASR] Agresywna transkrypcja: {len(audio) / self.audio_config.sample_rate:.2f}s, "
            f"peak={np.abs(audio).max():.3f}, beam=5, vad_filter=False"
        )
        return self._run_transcribe(audio, vad_filter=False, beam_size=5).text

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe a complete audio file (for testing)."""
        if not self.model:
            self.load_model()
        segments, _ = self.model.transcribe(
            audio_path,
            language=self.config.language,
            beam_size=self.config.beam_size,
            initial_prompt=POLISH_MEDICAL_PROMPT,
        )
        return " ".join(segment.text.strip() for segment in segments)
