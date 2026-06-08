"""Hands-free live voice call session (mic → ASR → LLM → TTS loop)."""

import platform
import sys
import threading
import time
import copy
import numpy as np

from agent.asr import TranscriptionResult
from agent.audio_io import PYAUDIO_AVAILABLE
from agent.pipeline import MedicalVoiceAgent
from config import LiveConfig
from medical.prompts import GREETING


def _normalize_transcript(text: str) -> str:
    return text.strip().lower().rstrip("!.?,")


def _reject_transcript(
    transcript: str,
    *,
    duration_s: float,
    peak_rms: float,
    utterance_rms: float,
    asr_result: TranscriptionResult,
    live: LiveConfig,
) -> str | None:
    """Return rejection reason, or None if transcript should be accepted."""
    threshold = live.speech_rms_threshold
    min_peak = threshold * live.min_peak_rms_multiplier

    if peak_rms < min_peak:
        return f"peak_rms {peak_rms:.0f} < {min_peak:.0f}"

    if not transcript.strip():
        return "empty_transcript"

    if asr_result.avg_logprob is not None and asr_result.avg_logprob < live.min_avg_logprob:
        return f"avg_logprob {asr_result.avg_logprob:.2f} < {live.min_avg_logprob}"

    if asr_result.no_speech_prob is not None and asr_result.no_speech_prob > 0.6:
        if asr_result.avg_logprob is None or asr_result.avg_logprob < live.min_avg_logprob + 0.3:
            return f"no_speech_prob {asr_result.no_speech_prob:.2f}"

    normalized = _normalize_transcript(transcript)
    words = normalized.split()
    if len(words) <= 2:
        hallucination_words = {word.lower() for word in live.silence_hallucination_words}
        if normalized in hallucination_words or any(word in hallucination_words for word in words):
            if (
                duration_s < live.silence_hallucination_max_duration_s
                or peak_rms < live.silence_hallucination_max_rms
                or utterance_rms < threshold
            ):
                return f"silence_hallucination '{transcript}'"

    if len(transcript.strip()) < 8 and peak_rms < threshold * 2:
        return f"short_low_energy (len={len(transcript.strip())}, peak={peak_rms:.0f})"

    return None


def _live_audio_status() -> tuple[bool, str]:
    """Return (available, user-facing status markdown)."""
    if not PYAUDIO_AVAILABLE:
        os_name = platform.system()
        install_hint = {
            "Windows": "pip install pyaudio (lub pobierz wheel z https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)",
            "Linux": "sudo apt install portaudio19-dev && pip install pyaudio",
            "Darwin": "brew install portaudio && pip install pyaudio",
        }.get(os_name, "pip install pyaudio")
        return False, (
            f"⚠️ **Tryb live niedostępny** — brak PyAudio ({os_name})\n\n"
            f"**Instalacja:** {install_hint}\n\n"
            "**Alternatywa (bez mikrofonu):**\n"
            "- `python main.py text` — tryb tekstowy\n"
            "- `python labs/asr_eval.py` — ewaluacja ASR na plikach WAV"
        )
    return True, ""

class LiveCallSession:
    STATUS_IDLE = "🔴 Rozmowa nieaktywna"
    STATUS_LISTENING = "🎤 Słucham... mów swobodnie"
    STATUS_PROCESSING = "⏳ Przetwarzam..."
    STATUS_SPEAKING = "🔊 Ania odpowiada..."

    def __init__(self, agent: MedicalVoiceAgent):
        self.agent = agent
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.active = False
        self.status = self.STATUS_IDLE
        self.chat_history: list[dict] = [{"role": "assistant", "content": GREETING}]
        self.emotion_md = "Oczekiwanie na rozmowę..."
        self.state_md = "## Stan rozmowy\n_Oczekiwanie..._"
        self.mic_md = "_Mikrofon nieaktywny_"
        self.last_tts_audio: tuple[int, np.ndarray] | None = None
        self._ui_version = 0
        self._tts_version = 0
        self._mic_level = 0.0
        self._user_speaking = False
        self._last_ui_bump = 0.0

    def start(self) -> str:
        if self.active and self._thread and self._thread.is_alive():
            return self.STATUS_LISTENING

        available, status_md = _live_audio_status()
        if not available:
            self.mic_md = status_md
            self.status = "⚠️ Mikrofon niedostępny — użyj trybu tekstowego lub labs/asr_eval.py"
            self._bump_ui(force=True)
            return self.status

        self._stop.clear()
        self.active = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return self.STATUS_LISTENING

    def stop(self) -> str:
        self._stop.set()
        self.active = False
        self.agent.audio_io.stop_recording()
        self.status = self.STATUS_IDLE
        self.mic_md = "_Mikrofon nieaktywny_"
        self._bump_ui()
        return self.STATUS_IDLE

    def _bump_ui(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_ui_bump < 0.15:
            return
        self._last_ui_bump = now
        with self._lock:
            self._ui_version += 1

    def _bump_tts(self, audio: tuple[int, np.ndarray]):
        with self._lock:
            self.last_tts_audio = audio
            self._tts_version += 1
            self._ui_version += 1

    def _update_mic_display(self, rms: float, in_speech: bool):
        self._mic_level = rms
        self._user_speaking = in_speech
        pct = min(100, max(0, (rms - 40) / 2500 * 100))
        bar_len = 24
        filled = int(pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        if rms < 15:
            perm_hint = (
                "Sprawdź: Ustawienia → Prywatność → Mikrofon → włącz dla **Terminal/Cursor/Python**."
                if sys.platform == "darwin"
                else "Sprawdź uprawnienia mikrofonu dla aplikacji terminalowej / Python w ustawieniach systemu."
            )
            self.mic_md = (
                f"⚠️ **Brak sygnału z mikrofonu!**\n\n"
                f"{perm_hint}\n\n"
                f"`[{bar}]` {pct:.0f}%"
            )
        elif in_speech:
            self.mic_md = f"🟢 **Wykrywam mowę** — mów dalej...\n\n`[{bar}]` **{pct:.0f}%**"
        else:
            self.mic_md = f"🎤 **Nasłuchuję** — zacznij mówić\n\n`[{bar}]` {pct:.0f}%"

        self._bump_ui()

    def get_snapshot(self) -> dict:
        with self._lock:
            return {
                "chat": copy.deepcopy(self.chat_history),
                "emotion": self.emotion_md,
                "state": self.state_md,
                "status": self.status,
                "mic": self.mic_md,
                "tts": self.last_tts_audio,
                "ui_version": self._ui_version,
                "tts_version": self._tts_version,
            }

    def _run_loop(self):
        try:
            device = self.agent.audio_io.start_recording()
            self.status = self.STATUS_SPEAKING
            self.mic_md = f"🎤 Mikrofon: **{device}**"
            self._bump_ui(force=True)
            self._speak_greeting()

            live = self.agent.config.live
            sr = self.agent.config.audio.sample_rate
            chunk_samples = self.agent.config.audio.chunk_size
            chunk_duration = chunk_samples / sr
            silence_chunks_needed = max(1, int(live.silence_seconds / chunk_duration))
            relative_silence_chunks_needed = max(
                1, int(live.relative_silence_seconds / chunk_duration)
            )
            min_speech_chunks = max(1, int(live.min_speech_seconds / chunk_duration))
            speech_threshold = live.speech_rms_threshold

            speech_buffer: list[bytes] = []
            silence_count = 0
            relative_silence_count = 0
            speech_chunks = 0
            in_utterance = False
            utterance_start = 0.0
            max_rms_seen = 0.0
            utterance_max_rms = 0.0
            listen_start = time.time()
            last_status_log = time.time()
            status_log_interval = live.status_log_interval_s

            print(
                f"[LiveSession] Nasłuch aktywny — próg RMS={speech_threshold}, "
                f"min_mowa={live.min_speech_seconds}s, cisza_koniec={live.silence_seconds}s, "
                f"max_wypowiedź={live.max_utterance_seconds}s, "
                f"względna_cisza={live.relative_silence_drop_ratio:.0%}"
            )

            while not self._stop.is_set():
                if self.agent.audio_io.is_paused:
                    if time.time() - last_status_log >= status_log_interval:
                        print("[LiveSession] Mikrofon wstrzymany (TTS/ASR) — czekam...")
                        last_status_log = time.time()
                    time.sleep(0.05)
                    continue

                self.status = self.STATUS_LISTENING
                chunk = self.agent.audio_io.get_audio_chunk(timeout=0.3)
                if chunk is None:
                    continue

                audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(audio_np**2))) if len(audio_np) else 0.0
                max_rms_seen = max(max_rms_seen, rms)
                if in_utterance:
                    utterance_max_rms = max(utterance_max_rms, rms)

                is_speech = rms >= speech_threshold
                self._update_mic_display(rms, in_utterance or is_speech)

                now = time.time()
                if now - last_status_log >= status_log_interval:
                    state = "MOWA" if in_utterance else "CISZA"
                    print(
                        f"[LiveSession] status rms={rms:.0f} max={max_rms_seen:.0f} "
                        f"utter_rms={utterance_max_rms:.0f} state={state} "
                        f"silence={silence_count}/{silence_chunks_needed} "
                        f"rel_silence={relative_silence_count}/{relative_silence_chunks_needed} "
                        f"paused={self.agent.audio_io.is_paused} thread=alive"
                    )
                    last_status_log = now

                end_reason = None
                if in_utterance:
                    utterance_elapsed = now - utterance_start
                    if utterance_elapsed >= live.max_utterance_seconds:
                        end_reason = f"max_duration ({utterance_elapsed:.1f}s)"
                    elif (
                        utterance_max_rms > speech_threshold
                        and rms < utterance_max_rms * (1 - live.relative_silence_drop_ratio)
                    ):
                        relative_silence_count += 1
                        if relative_silence_count >= relative_silence_chunks_needed:
                            end_reason = (
                                f"relative_silence (rms={rms:.0f} peak={utterance_max_rms:.0f})"
                            )
                    else:
                        relative_silence_count = 0

                if is_speech:
                    if not in_utterance:
                        utterance_start = now
                    speech_buffer.append(chunk)
                    speech_chunks += 1
                    silence_count = 0
                    in_utterance = True
                elif in_utterance:
                    speech_buffer.append(chunk)
                    silence_count += 1
                    if silence_count >= silence_chunks_needed:
                        end_reason = f"silence ({silence_count} chunks)"

                if in_utterance and end_reason:
                    full_audio = b"".join(speech_buffer)
                    seg_duration = len(full_audio) / (2 * sr)
                    seg_max_rms = utterance_max_rms
                    print(
                        f"[LiveSession] SEGMENT_END reason={end_reason} "
                        f"duration={seg_duration:.2f}s rms_max={seg_max_rms:.0f} "
                        f"speech_chunks={speech_chunks}"
                    )

                    speech_buffer = []
                    silence_count = 0
                    relative_silence_count = 0
                    speech_chunks_saved = speech_chunks
                    speech_chunks = 0
                    in_utterance = False
                    utterance_max_rms = 0.0
                    listen_start = time.time()
                    max_rms_seen = 0.0

                    min_peak_rms = speech_threshold * live.min_peak_rms_multiplier
                    if speech_chunks_saved >= min_speech_chunks and seg_max_rms >= min_peak_rms:
                        self.status = self.STATUS_PROCESSING
                        self.mic_md = "⏳ **Przetwarzam wypowiedź...**"
                        self._bump_ui(force=True)
                        try:
                            self._handle_utterance(
                                full_audio,
                                seg_duration=seg_duration,
                                seg_max_rms=seg_max_rms,
                            )
                        except Exception as e:
                            print(f"[LiveSession] Błąd tury: {e}")
                            import traceback
                            traceback.print_exc()
                            self.mic_md = f"❌ Błąd: {e} — możesz mówić dalej"
                            self.agent.audio_io.resume_recording()
                            self._bump_ui(force=True)
                    elif speech_chunks_saved < min_speech_chunks:
                        print(
                            f"[LiveSession] Za krótka wypowiedź — odrzucam "
                            f"({seg_duration:.2f}s, {speech_chunks_saved} chunks, RMS max={seg_max_rms:.0f})"
                        )
                    else:
                        print(
                            f"[LiveSession] Za niski szczyt RMS — odrzucam "
                            f"(peak={seg_max_rms:.0f} < {min_peak_rms:.0f}, {seg_duration:.2f}s)"
                        )
                elif time.time() - listen_start > 8 and max_rms_seen < live.speech_rms_threshold:
                    perm_hint = (
                        "Sprawdź uprawnienia mikrofonu dla Python/Terminal w Ustawieniach macOS."
                        if sys.platform == "darwin"
                        else "Sprawdź, czy właściwe urządzenie wejściowe jest wybrane w ustawieniach dźwięku."
                    )
                    self.mic_md = (
                        f"⚠️ **Mikrofon słyszy tylko ciszę** (max {max_rms_seen:.0f})\n\n"
                        f"Urządzenie: {device}\n"
                        f"{perm_hint}"
                    )
                    self._bump_ui(force=True)

        except Exception as e:
            print(f"[LiveSession] Błąd: {e}")
            import traceback
            traceback.print_exc()
            self.status = f"❌ Błąd: {e}"
            self._bump_ui(force=True)
        finally:
            self.agent.audio_io.stop_recording()
            if not self._stop.is_set():
                self.active = False
                self.status = self.STATUS_IDLE
                self._bump_ui(force=True)

    def _handle_utterance(
        self,
        raw_audio: bytes,
        *,
        seg_duration: float | None = None,
        seg_max_rms: float | None = None,
    ) -> str:
        self.agent.audio_io.pause_recording()
        transcript = ""
        try:
            audio_np = np.frombuffer(raw_audio, dtype=np.int16)
            sr = self.agent.config.audio.sample_rate
            duration_s = seg_duration if seg_duration is not None else (len(audio_np) / sr if sr else 0.0)
            utterance_rms = (
                float(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))
                if len(audio_np)
                else 0.0
            )
            peak_rms = seg_max_rms if seg_max_rms is not None else utterance_rms

            self.status = self.STATUS_PROCESSING
            self.mic_md = "⏳ **Przetwarzam wypowiedź...**"
            self._bump_ui(force=True)

            live = self.agent.config.live
            min_peak_rms = live.speech_rms_threshold * live.min_peak_rms_multiplier

            print(
                f"[LiveSession] ASR_START duration={duration_s:.2f}s "
                f"rms_avg={utterance_rms:.0f} rms_max={peak_rms:.0f} bytes={len(raw_audio)}"
            )

            if peak_rms < min_peak_rms:
                print(
                    f"[LiveSession] Odrzucam przed ASR — za niski szczyt RMS "
                    f"({peak_rms:.0f} < {min_peak_rms:.0f})"
                )
                self.mic_md = (
                    "🟡 **Nie wykryto mowy** — mów głośniej i wyraźniej\n\n"
                    f"_Poziom {peak_rms:.0f} (wymagane ≥ {min_peak_rms:.0f})_"
                )
                self._bump_ui(force=True)
                return ""

            allow_aggressive_retry = peak_rms >= min_peak_rms * 1.1
            asr_result = self.agent.asr.transcribe_array_detailed(
                audio_np,
                sr,
                allow_aggressive_retry=allow_aggressive_retry,
            )
            transcript = asr_result.text

            print(
                f"[LiveSession] ASR_RESULT transcript={transcript!r} "
                f"avg_logprob={asr_result.avg_logprob} no_speech_prob={asr_result.no_speech_prob}"
            )

            reject_reason = _reject_transcript(
                transcript,
                duration_s=duration_s,
                peak_rms=peak_rms,
                utterance_rms=utterance_rms,
                asr_result=asr_result,
                live=live,
            )
            if reject_reason:
                print(
                    f"[LiveSession] Odrzucona transkrypcja ({reject_reason}) — "
                    f"rms={utterance_rms:.0f}, max={peak_rms:.0f}, len={len(raw_audio)}B, {duration_s:.2f}s"
                )
                self.mic_md = (
                    "🟡 **Nie rozpoznano mowy** — mów głośniej i wyraźniej po polsku\n\n"
                    f"_Nagranie: {duration_s:.1f}s, poziom {peak_rms:.0f} "
                    f"(próg {live.speech_rms_threshold})_"
                )
                self._bump_ui(force=True)
                return ""

            with self._lock:
                self.chat_history.append({"role": "user", "content": transcript})
            self._bump_ui(force=True)
            print(f"[LiveSession] UI updated with user transcript: {transcript!r}")

            self._handle_turn(transcript, raw_audio, user_already_in_chat=True)
            return transcript
        finally:
            self.agent.audio_io.resume_recording()
            self.status = self.STATUS_LISTENING
            self.mic_md = "🎤 **Twoja kolej** — mów teraz"
            self._bump_ui(force=True)
            print("[LiveSession] Gotowe — nasłuchuję kolejnej wypowiedzi")

    def _speak_greeting(self):
        with self._lock:
            self.chat_history = [{"role": "assistant", "content": GREETING}]
        self.agent.audio_io.pause_recording()
        try:
            self._speak(GREETING)
        finally:
            self.agent.audio_io.resume_recording()

    def _handle_turn(
        self,
        transcript: str,
        raw_audio: bytes,
        *,
        user_already_in_chat: bool = False,
    ):
        audio_np = np.frombuffer(raw_audio, dtype=np.int16)

        self.status = "🤖 Ania myśli..."
        self.mic_md = "🤖 **Ania przygotowuje odpowiedź...**"
        self._bump_ui(force=True)

        print(f"[LiveSession] LLM_START transcript={transcript!r}")
        try:
            self.agent.llm.check_connection()
        except Exception as e:
            print(f"[LiveSession] LLM connection check failed: {e}")
            err_msg = (
                f"❌ Brak połączenia z Ollama ({self.agent.config.llm.base_url}). "
                "Uruchom: `ollama serve`"
            )
            with self._lock:
                if not user_already_in_chat:
                    self.chat_history.append({"role": "user", "content": transcript})
                self.chat_history.append({"role": "assistant", "content": err_msg})
            self._bump_ui(force=True)
            return

        result = self.agent.process_text_turn(transcript, audio_np)
        print(f"[LiveSession] LLM_DONE response={result['response'][:120]!r}...")

        with self._lock:
            if not user_already_in_chat:
                self.chat_history.append({"role": "user", "content": transcript})
            self.chat_history.append({"role": "assistant", "content": result["response"]})

            for action in result.get("actions", []):
                if action.get("action") == "appointment_booked":
                    conf = action["result"].get("confirmation", "")
                    if conf:
                        self.chat_history.append({"role": "assistant", "content": f"✅ {conf}"})
                elif action.get("action") == "transfer_to_operator":
                    self.chat_history.append({"role": "assistant", "content": "📞 Przekazuję do operatora..."})
                elif action.get("action") == "emergency":
                    self.chat_history.append({"role": "assistant", "content": f"🚨 {action['message']}"})

            self.emotion_md = _format_emotion(result.get("emotion", {}))
            self.state_md = _format_state(result.get("state", {}))

        self._bump_ui(force=True)
        self._speak(result["response"])

    def _speak(self, text: str):
        """Synthesize TTS for Gradio playback (mic already muted by caller)."""
        self.status = self.STATUS_SPEAKING
        self._bump_ui(force=True)
        try:
            audio, sr = self.agent.tts.synthesize(text)
            self._bump_tts((sr, audio))
            time.sleep(len(audio) / sr + 0.4)
        except Exception as e:
            print(f"[TTS] Błąd: {e}")


def _format_emotion(emotion: dict) -> str:
    if not emotion:
        return "Brak danych"
    em = emotion.get("emotion", "?")
    intensity = emotion.get("emotion_intensity", 0)
    bar = "█" * intensity + "░" * (10 - intensity)
    lines = [
        f"**Emocja:** {em}",
        f"**Intensywność:** [{bar}] {intensity}/10",
        f"**Pilność:** {emotion.get('urgency', '?')}/10",
        f"**Wiek:** {emotion.get('estimated_age_group', '?')}",
        f"**Płeć:** {emotion.get('estimated_gender', '?')}",
    ]
    if emotion.get("needs_calming"):
        lines.append("⚠️ **WYMAGA USPOKOJENIA**")
    return "\n".join(lines)


def _format_state(state: dict) -> str:
    lines = ["## Stan rozmowy\n"]
    if state.get("patient_name"):
        lines.append(f"**Pacjent:** {state['patient_name']}")
    if state.get("complaint"):
        lines.append(f"**Dolegliwość:** {state['complaint']}")
    if state.get("specialty"):
        lines.append(f"**Specjalista:** {state['specialty']}")
    if state.get("appointment_booked"):
        lines.append("✅ **Wizyta umówiona**")
    if state.get("transferred"):
        lines.append("📞 **Przekazano do operatora**")
    if len(lines) == 1:
        lines.append("_Zbieranie danych..._")
    return "\n".join(lines)
