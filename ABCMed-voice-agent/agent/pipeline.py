"""Main voice agent pipeline: ASR → LLM → TTS with emotion monitoring."""

import re
import time
import threading
import numpy as np
from dataclasses import dataclass, field

from config import AppConfig
from agent.asr import StreamingASR
from agent.llm import LocalLLM, LLMConnectionError, LLMTimeoutError, LLMError
from agent.tts import create_tts
from agent.emotion import VoiceEmotionAnalyzer
from agent.audio_io import AudioIO
from medical.scheduler import AppointmentScheduler
from medical.prompts import get_system_prompt, GREETING, TRANSFER_TO_OPERATOR, EMERGENCY_RESPONSE
from medical.specialties import get_specialty_by_name


@dataclass
class ConversationState:
    patient_name: str | None = None
    complaint: str | None = None
    suggested_specialty: str | None = None
    appointment_booked: bool = False
    transferred_to_operator: bool = False
    is_emergency: bool = False
    emotion_history: list = field(default_factory=list)
    turn_count: int = 0


class MedicalVoiceAgent:
    def __init__(self, config: AppConfig):
        self.config = config
        self.asr = StreamingASR(config.asr, config.audio)
        self.llm = LocalLLM(config.llm)
        self.tts = create_tts(config.tts)
        self.emotion_analyzer = VoiceEmotionAnalyzer(config.audio.sample_rate)
        self.audio_io = AudioIO(config.audio)
        self.scheduler = AppointmentScheduler()
        self.state = ConversationState()
        self._running = False
        self._callbacks: dict[str, list] = {
            "on_transcription": [],
            "on_response": [],
            "on_emotion_update": [],
            "on_action": [],
            "on_tts_start": [],
            "on_tts_done": [],
        }

    def on(self, event: str, callback):
        """Register event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, data):
        for cb in self._callbacks.get(event, []):
            cb(data)

    def initialize(self):
        """Load all models."""
        print("=" * 50)
        print("  ABCMed - Asystent Głosowy Call Center")
        print("=" * 50)
        print()
        self.asr.load_model()
        if self.config.tts.backend == "chatterbox":
            self.tts.load_model()
        else:
            self.tts.load_model()
        print(f"[LLM] Model: {self.llm.model}")
        print(f"[ASR] Whisper {self.config.asr.model_size} beam={self.config.asr.beam_size}")

        available_dates = self.scheduler.get_available_dates_text()
        system_prompt = get_system_prompt(available_dates)
        self.llm.set_system_prompt(system_prompt)
        print("\n[System] Wszystkie modele załadowane. Gotowy do rozmowy.\n")

    def start_live(self):
        """Start live voice conversation with microphone."""
        self.initialize()
        self._running = True

        # Play greeting
        self._speak(GREETING)
        self._emit("on_response", GREETING)

        # Start recording
        self.audio_io.start_recording()
        audio_buffer = b""

        print("\n[System] Mów do mikrofonu... (Ctrl+C aby zakończyć)\n")

        try:
            while self._running:
                chunk = self.audio_io.get_audio_chunk(timeout=0.5)
                if chunk is None:
                    continue

                audio_buffer += chunk

                # Process when we have enough audio (480ms chunks for VAD)
                min_process_size = int(self.config.audio.sample_rate * 0.48) * 2
                if len(audio_buffer) >= min_process_size:
                    transcript = self.asr.process_audio_chunk(audio_buffer)
                    if transcript:
                        self._handle_turn(transcript, audio_buffer)
                        audio_buffer = b""
                    else:
                        # Keep last 100ms for continuity
                        keep_bytes = int(self.config.audio.sample_rate * 0.1) * 2
                        audio_buffer = audio_buffer[-keep_bytes:]

        except KeyboardInterrupt:
            print("\n[System] Rozmowa zakończona.")
        finally:
            self._running = False
            self.audio_io.stop_recording()

    def process_text_turn(self, text: str, audio_data: np.ndarray | None = None) -> dict:
        """Process a single conversation turn (for Gradio/text mode).
        Returns dict with response, emotion, actions."""
        self.state.turn_count += 1

        # Emotion analysis from audio if available
        emotion_data = None
        if audio_data is not None:
            emotion_data = self.emotion_analyzer.analyze_audio(audio_data)
            self._emit("on_emotion_update", emotion_data)
            self.state.emotion_history.append(emotion_data)

        # LLM emotion analysis from text
        text_emotion = self.llm.analyze_emotion(text)

        # Combine voice + text emotion
        combined_emotion = self._combine_emotions(emotion_data, text_emotion)

        self._emit("on_transcription", text)

        # Get LLM response
        try:
            response = self.llm.chat(text, timeout=self.config.llm.timeout_seconds)
        except LLMTimeoutError as e:
            print(f"[LLM] Timeout: {e}")
            return self._llm_error_result(
                text,
                combined_emotion,
                f"⏱️ Ania nie odpowiedziała na czas ({self.config.llm.timeout_seconds:.0f}s). "
                "Sprawdź, czy Ollama działa i spróbuj ponownie.",
            )
        except LLMConnectionError as e:
            print(f"[LLM] Połączenie: {e}")
            return self._llm_error_result(
                text,
                combined_emotion,
                f"❌ Brak połączenia z Ollama ({self.config.llm.base_url}). "
                "Uruchom: `ollama serve`",
            )
        except LLMError as e:
            print(f"[LLM] Błąd: {e}")
            return self._llm_error_result(
                text,
                combined_emotion,
                f"❌ Błąd LLM: {e}",
            )

        # Parse actions from response
        actions = self._parse_actions(response)
        clean_response = self._clean_response(response)

        # Execute actions
        action_results = []
        for action in actions:
            result = self._execute_action(action)
            action_results.append(result)
            self._emit("on_action", result)

        self._emit("on_response", clean_response)

        return {
            "response": clean_response,
            "emotion": combined_emotion,
            "actions": action_results,
            "turn": self.state.turn_count,
            "state": {
                "patient_name": self.state.patient_name,
                "complaint": self.state.complaint,
                "specialty": self.state.suggested_specialty,
                "appointment_booked": self.state.appointment_booked,
                "transferred": self.state.transferred_to_operator,
            },
        }

    def _handle_turn(self, transcript: str, raw_audio: bytes):
        """Handle a full conversation turn in live mode."""
        print(f"\n🗣️  Pacjent: {transcript}")
        self._emit("on_transcription", transcript)

        # Analyze voice emotion
        audio_np = np.frombuffer(raw_audio, dtype=np.int16)
        emotion = self.emotion_analyzer.analyze_audio(audio_np)
        self._emit("on_emotion_update", emotion)

        # Get response
        result = self.process_text_turn(transcript, audio_np)

        print(f"🤖 Ania: {result['response']}")
        if result["emotion"]:
            em = result["emotion"]
            print(f"   📊 Emocje: {em.get('emotion', '?')} (intensywność: {em.get('emotion_intensity', '?')}/10)")

        # Speak response
        self._speak(result["response"])

    def _speak(self, text: str):
        """Convert text to speech and play."""
        self._emit("on_tts_start", text)
        try:
            audio, sr = self.tts.synthesize(text)
            self.audio_io.play_audio(audio, sr)
        except Exception as e:
            print(f"[TTS] Błąd syntezy: {e}")
        self._emit("on_tts_done", text)

    def _llm_error_result(self, text: str, emotion: dict, message: str) -> dict:
        """Return a turn result when LLM fails."""
        self._emit("on_response", message)
        return {
            "response": message,
            "emotion": emotion,
            "actions": [],
            "turn": self.state.turn_count,
            "state": {
                "patient_name": self.state.patient_name,
                "complaint": self.state.complaint,
                "specialty": self.state.suggested_specialty,
                "appointment_booked": self.state.appointment_booked,
                "transferred": self.state.transferred_to_operator,
            },
        }

    def _combine_emotions(self, voice_emotion: dict | None, text_emotion: dict) -> dict:
        """Combine voice-based and text-based emotion analysis."""
        result = dict(text_emotion)

        if voice_emotion:
            arousal = voice_emotion.get("arousal", 0.5)
            # Boost intensity if voice indicates high arousal
            if arousal > 0.7:
                result["emotion_intensity"] = min(10, result.get("emotion_intensity", 5) + 2)
            # Use voice profile for gender/age if available
            profile = voice_emotion.get("voice_profile", {})
            if profile.get("estimated_gender") != "nieokreślone":
                result["estimated_gender"] = profile["estimated_gender"]
            if profile.get("estimated_age_group") != "dorosły":
                result["estimated_age_group"] = profile["estimated_age_group"]
            result["voice_arousal"] = arousal
            result["voice_stress"] = voice_emotion.get("voice_stress_indicators", {})

        return result

    def _parse_actions(self, response: str) -> list[dict]:
        """Parse action commands from LLM response."""
        actions = []
        pattern = r'\[AKCJA:\s*(\w+)\s*(.*?)\]'
        matches = re.finditer(pattern, response)

        for match in matches:
            action_type = match.group(1)
            params_str = match.group(2)
            params = {}
            param_pattern = r'(\w+)="([^"]*)"'
            for p_match in re.finditer(param_pattern, params_str):
                params[p_match.group(1)] = p_match.group(2)
            actions.append({"type": action_type, "params": params})

        return actions

    def _clean_response(self, response: str) -> str:
        """Remove action tags from response text."""
        cleaned = re.sub(r'\[AKCJA:.*?\]', '', response)
        return cleaned.strip()

    def _execute_action(self, action: dict) -> dict:
        """Execute a parsed action."""
        action_type = action["type"]
        params = action["params"]

        if action_type == "UMÓW_WIZYTĘ":
            result = self.scheduler.book_appointment(
                patient_name=params.get("pacjent", "Nieznany"),
                specialty=params.get("specjalista", ""),
                doctor=params.get("lekarz", ""),
                date=params.get("data", ""),
                time=params.get("godzina", ""),
                reason=params.get("powód", ""),
            )
            if result["success"]:
                self.state.appointment_booked = True
                self.state.patient_name = params.get("pacjent")
            return {"action": "appointment_booked", "result": result}

        elif action_type == "PRZEKAŻ_DO_OPERATORA":
            self.state.transferred_to_operator = True
            return {
                "action": "transfer_to_operator",
                "reason": params.get("powód", "Na prośbę pacjenta"),
            }

        elif action_type == "NAGŁY_PRZYPADEK":
            self.state.is_emergency = True
            return {"action": "emergency", "message": EMERGENCY_RESPONSE}

        return {"action": "unknown", "raw": action}
