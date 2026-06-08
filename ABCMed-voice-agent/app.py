"""Gradio Web UI for the Medical Voice Agent demo."""

import time
import numpy as np
import gradio as gr

from config import AppConfig
from agent.pipeline import MedicalVoiceAgent
from agent.live_session import LiveCallSession
from medical.prompts import GREETING


# Global agent instance
agent: MedicalVoiceAgent | None = None
live_session: LiveCallSession | None = None
conversation_log: list[dict] = []
_last_tts_version = -1


def initialize_agent():
    global agent
    if agent is None:
        config = AppConfig()
        agent = MedicalVoiceAgent(config)
        agent.initialize()
    return agent


def format_emotion_display(emotion: dict) -> str:
    """Format emotion data for display."""
    if not emotion:
        return "Brak danych"

    em = emotion.get("emotion", "nieokreślone")
    intensity = emotion.get("emotion_intensity", 0)
    age = emotion.get("estimated_age_group", "?")
    gender = emotion.get("estimated_gender", "?")
    urgency = emotion.get("urgency", 0)
    needs_calming = emotion.get("needs_calming", False)
    arousal = emotion.get("voice_arousal", None)

    bar = "█" * intensity + "░" * (10 - intensity)

    lines = [
        f"**Emocja:** {em}",
        f"**Intensywność:** [{bar}] {intensity}/10",
        f"**Pilność medyczna:** {urgency}/10",
        f"**Wiek (szacunek):** {age}",
        f"**Płeć (szacunek):** {gender}",
    ]

    if arousal is not None:
        arousal_bar = "█" * int(arousal * 10) + "░" * (10 - int(arousal * 10))
        lines.append(f"**Pobudzenie głosowe:** [{arousal_bar}] {arousal:.2f}")

    if needs_calming:
        lines.append("⚠️ **WYMAGA USPOKOJENIA**")

    stress = emotion.get("voice_stress", {})
    if stress:
        indicators = [k for k, v in stress.items() if v]
        if indicators:
            lines.append(f"**Wskaźniki stresu:** {', '.join(indicators)}")

    return "\n".join(lines)


def format_state_display(state: dict) -> str:
    """Format conversation state for display."""
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
        lines.append("_Oczekiwanie na dane pacjenta..._")

    return "\n".join(lines)


def process_audio_input(audio, chat_history):
    """Process audio input from Gradio microphone."""
    global agent

    if agent is None:
        initialize_agent()

    if audio is None:
        return chat_history, "", ""

    sr, audio_data = audio

    transcript = agent.asr.transcribe_array(audio_data, sr)

    if not transcript.strip():
        return chat_history, "Nie rozpoznano mowy — mów wyraźniej, min. 2 sekundy", ""

    # Process turn (audio for emotion analysis — resample to 16 kHz)
    audio_16k = agent.asr._prepare_audio(audio_data, sr)
    audio_int16 = (audio_16k * 32767).astype(np.int16)
    result = agent.process_text_turn(transcript, audio_int16)

    # Update chat
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": transcript})
    chat_history.append({"role": "assistant", "content": result["response"]})

    emotion_display = format_emotion_display(result.get("emotion", {}))
    state_display = format_state_display(result.get("state", {}))

    # Handle actions
    for action in result.get("actions", []):
        if action.get("action") == "appointment_booked":
            conf = action["result"].get("confirmation", "")
            if conf:
                chat_history.append({"role": "assistant", "content": f"✅ {conf}"})

    return chat_history, emotion_display, state_display


def process_text_input(text, chat_history):
    """Process text input (for testing without microphone)."""
    global agent

    if agent is None:
        initialize_agent()

    if not text or not text.strip():
        return chat_history, "", "", ""

    result = agent.process_text_turn(text.strip())

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": text.strip()})
    chat_history.append({"role": "assistant", "content": result["response"]})

    emotion_display = format_emotion_display(result.get("emotion", {}))
    state_display = format_state_display(result.get("state", {}))

    for action in result.get("actions", []):
        if action.get("action") == "appointment_booked":
            conf = action["result"].get("confirmation", "")
            if conf:
                chat_history.append({"role": "assistant", "content": f"✅ {conf}"})
        elif action.get("action") == "transfer_to_operator":
            chat_history.append({"role": "assistant", "content": "📞 Przekazuję do operatora..."})
        elif action.get("action") == "emergency":
            chat_history.append({"role": "assistant", "content": f"🚨 {action['message']}"})

    return chat_history, emotion_display, state_display, ""


def synthesize_last_response(chat_history):
    """Synthesize the last assistant response to audio."""
    global agent
    if not chat_history or agent is None:
        return None

    last_assistant = None
    for msg in reversed(chat_history):
        if msg["role"] == "assistant" and not msg["content"].startswith("✅") and not msg["content"].startswith("📞"):
            last_assistant = msg["content"]
            break

    if not last_assistant:
        return None

    try:
        audio, sr = agent.tts.synthesize(last_assistant)
        return (sr, audio)
    except Exception as e:
        print(f"[TTS Error] {e}")
        return None


def start_live_call():
    """Start hands-free voice conversation."""
    global live_session, _last_tts_version
    _last_tts_version = -1
    ag = initialize_agent()
    if live_session is None:
        live_session = LiveCallSession(ag)
    status = live_session.start()
    snap = live_session.get_snapshot()
    return (
        snap["chat"],
        snap["emotion"],
        snap["state"],
        status,
        snap["mic"],
        snap["tts"],
        gr.Timer(active=True),
    )


def stop_live_call():
    """Stop hands-free voice conversation."""
    global live_session
    if live_session:
        status = live_session.stop()
    else:
        status = LiveCallSession.STATUS_IDLE
    return status, gr.Timer(active=False)


def refresh_live_ui():
    """Poll live session state and update UI."""
    global _last_tts_version
    if live_session is None or not live_session.active:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    snap = live_session.get_snapshot()
    # Odtwarzaj TTS tylko raz — nie powtarzaj przy każdym ticku timera
    if snap["tts_version"] != _last_tts_version:
        _last_tts_version = snap["tts_version"]
        tts_out = snap["tts"]
    else:
        tts_out = gr.skip()

    return (
        snap["chat"],
        snap["emotion"],
        snap["state"],
        snap["status"],
        snap["mic"],
        tts_out,
    )


def reset_conversation():
    """Reset the conversation state."""
    global agent, live_session
    if live_session and live_session.active:
        live_session.stop()
    live_session = None
    if agent:
        agent.state = __import__("agent.pipeline", fromlist=["ConversationState"]).ConversationState()
        agent.llm.set_system_prompt(
            __import__("medical.prompts", fromlist=["get_system_prompt"]).get_system_prompt(
                agent.scheduler.get_available_dates_text()
            )
        )
    return [], "", "", None, "_Mikrofon nieaktywny_", LiveCallSession.STATUS_IDLE, gr.Timer(active=False)


def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(title="ABCMed - Asystent Głosowy") as demo:
        gr.Markdown("""
        # 🏥 ABCMed — Asystent Głosowy Call Center
        ### ASR (Whisper) + LLM (Ollama) + TTS (macOS) + Analiza emocji
        """)

        live_status = gr.Markdown(value=LiveCallSession.STATUS_IDLE, label="Status")
        mic_display = gr.Markdown(value="_Mikrofon nieaktywny — kliknij „Rozpocznij rozmowę”_", label="Mikrofon")

        with gr.Row():
            start_call_btn = gr.Button("📞 Rozpocznij rozmowę na żywo", variant="primary", scale=2)
            stop_call_btn = gr.Button("📴 Zakończ rozmowę", variant="stop", scale=1)
            reset_btn = gr.Button("🔄 Nowa rozmowa", variant="secondary", scale=1)

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": GREETING}],
                    height=500,
                    label="Rozmowa",
                )

                with gr.Accordion("Tryb tekstowy (opcjonalny)", open=False):
                    with gr.Row():
                        text_input = gr.Textbox(
                            placeholder="Wpisz tekst zamiast mówić...",
                            label="Tekst",
                            lines=1,
                        )
                        send_btn = gr.Button("Wyślij", variant="secondary")

                audio_output = gr.Audio(
                    label="🔊 Odpowiedź głosowa Ani",
                    type="numpy",
                    autoplay=True,
                    interactive=False,
                )

            with gr.Column(scale=1):
                gr.Markdown("## 📊 Monitor emocji")
                emotion_display = gr.Markdown(
                    value="Oczekiwanie na rozmowę...",
                    elem_classes=["emotion-panel"],
                )

                gr.Markdown("---")

                state_display = gr.Markdown(
                    value="## Stan rozmowy\n_Oczekiwanie..._",
                    elem_classes=["state-panel"],
                )

                gr.Markdown("---")
                gr.Markdown("""
                ### ℹ️ Jak używać
                1. Kliknij **Rozpocznij rozmowę na żywo**
                2. Mów swobodnie — system sam wykrywa koniec wypowiedzi
                3. Ania odpowiada głosem i tekst pojawia się w czacie
                4. Tekst poniżej to opcja pomocnicza
                """)

        live_timer = gr.Timer(0.5, active=False)

        # Event handlers
        start_call_btn.click(
            start_live_call,
            outputs=[chatbot, emotion_display, state_display, live_status, mic_display, audio_output, live_timer],
        )

        stop_call_btn.click(
            stop_live_call,
            outputs=[live_status, live_timer],
        )

        live_timer.tick(
            refresh_live_ui,
            outputs=[chatbot, emotion_display, state_display, live_status, mic_display, audio_output],
        )

        send_btn.click(
            process_text_input,
            inputs=[text_input, chatbot],
            outputs=[chatbot, emotion_display, state_display, text_input],
        )

        text_input.submit(
            process_text_input,
            inputs=[text_input, chatbot],
            outputs=[chatbot, emotion_display, state_display, text_input],
        )

        reset_btn.click(
            reset_conversation,
            outputs=[chatbot, emotion_display, state_display, audio_output, mic_display, live_status, live_timer],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    )
