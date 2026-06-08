"""Main entry point for Medical Voice Agent."""

import sys
from config import AppConfig
from agent.pipeline import MedicalVoiceAgent
from medical.prompts import GREETING


def run_text_demo():
    """Run text-only demo (no microphone needed, for quick testing)."""
    config = AppConfig()
    agent = MedicalVoiceAgent(config)

    print("=" * 60)
    print("  ABCMed - Asystent Głosowy Call Center (tryb tekstowy)")
    print("=" * 60)
    print()
    print("  Ładowanie modeli... (ASR + LLM + TTS)")
    print("  Uwaga: TTS Chatterbox ładuje się dłużej za pierwszym razem.")
    print()

    agent.asr.load_model()
    # TTS will be loaded on first use
    available_dates = agent.scheduler.get_available_dates_text()
    from medical.prompts import get_system_prompt
    system_prompt = get_system_prompt(available_dates)
    agent.llm.set_system_prompt(system_prompt)

    print(f"\n{'─' * 60}")
    print(f"🤖 Ania: {GREETING}")
    print(f"{'─' * 60}")
    print("\n  (Wpisz tekst jako pacjent. 'q' = wyjście, 'tts' = odczytaj ostatnią)\n")

    last_response = GREETING

    while True:
        try:
            user_input = input("🗣️  Pacjent: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[Koniec rozmowy]")
            break

        if not user_input:
            continue
        if user_input.lower() == "q":
            print("\n[Koniec rozmowy]")
            break
        if user_input.lower() == "tts":
            print(f"\n[Synteza mowy: '{last_response[:50]}...']")
            try:
                agent.tts.synthesize_to_file(last_response, "/tmp/abcmed_response.wav")
                print("[Zapisano: /tmp/abcmed_response.wav]")
                agent.audio_io.play_audio(*agent.tts.synthesize(last_response))
            except Exception as e:
                print(f"[Błąd TTS: {e}]")
            continue

        result = agent.process_text_turn(user_input)

        print(f"\n{'─' * 60}")
        print(f"🤖 Ania: {result['response']}")

        # Show emotion analysis
        em = result.get("emotion", {})
        if em:
            emotion = em.get("emotion", "?")
            intensity = em.get("emotion_intensity", 0)
            age = em.get("estimated_age_group", "?")
            gender = em.get("estimated_gender", "?")
            bar = "█" * intensity + "░" * (10 - intensity)
            print(f"   📊 Emocja: {emotion} [{bar}] | Wiek: {age} | Płeć: {gender}")
            if em.get("needs_calming"):
                print(f"   ⚠️  Pacjent wymaga uspokojenia!")

        # Show actions
        for action in result.get("actions", []):
            if action.get("action") == "appointment_booked":
                conf = action["result"].get("confirmation", "")
                print(f"   ✅ {conf}")
            elif action.get("action") == "transfer_to_operator":
                print(f"   📞 Przekazanie do operatora: {action.get('reason', '')}")
            elif action.get("action") == "emergency":
                print(f"   🚨 NAGŁY PRZYPADEK!")

        print(f"{'─' * 60}\n")
        last_response = result["response"]


def run_live_mode():
    """Run live voice mode with microphone."""
    config = AppConfig()
    agent = MedicalVoiceAgent(config)
    agent.start_live()


def run_web_ui():
    """Run Gradio web interface."""
    from app import create_ui
    import gradio as gr
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "live":
            run_live_mode()
        elif mode == "web":
            run_web_ui()
        elif mode == "text":
            run_text_demo()
        else:
            print(f"Nieznany tryb: {mode}")
            print("Użycie: python main.py [text|live|web]")
    else:
        print("ABCMed Voice Agent - Wybierz tryb:")
        print("  python main.py text  — tryb tekstowy (bez mikrofonu)")
        print("  python main.py live  — tryb głosowy (mikrofon)")
        print("  python main.py web   — interfejs Gradio (przeglądarka)")
        print()
        run_text_demo()
