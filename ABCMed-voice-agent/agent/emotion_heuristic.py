"""Szybka analiza emocji bez LLM — heurystyki na tekście polskim."""


def quick_emotion_from_text(text: str) -> dict:
    t = text.lower()

    angry = any(
        w in t
        for w in [
            "zdenerwow", "wściek", "wkurz", "tracę cierpliwość", "nikt nie",
            "po raz", "maszyn", "natychmiast", "w końcu",
        ]
    )
    scared = any(w in t for w in ["boję", "strach", "panik", "przeraż"])
    sad = any(w in t for w in ["smutn", "płacz", "beznadziej"])
    urgent = any(
        w in t
        for w in [
            "ból w klatce", "duszno", "utrata przytomności", "krwaw",
            "nagły", "pilne", "112", "pogotow",
        ]
    )

    if urgent:
        emotion, intensity, urgency = "strach", 8, 9
    elif angry:
        emotion, intensity, urgency = "złość", 7, 6
    elif scared:
        emotion, intensity, urgency = "strach", 7, 7
    elif sad:
        emotion, intensity, urgency = "smutek", 6, 5
    elif any(w in t for w in ["stres", "nerw", "niepokoj"]):
        emotion, intensity, urgency = "stres", 6, 5
    else:
        emotion, intensity, urgency = "spokój", 3, 4

    # Prosta heurystyka płci z form gramatycznych
    if any(w in t for w in ["am zdenerwowana", "byłam", "chciałabym", "mogłabym", "panią"]):
        gender = "kobieta"
    elif any(w in t for w in ["byłem", "chciałbym", "mogłbym", "pana "]):
        gender = "mężczyzna"
    else:
        gender = "nieokreślone"

    return {
        "emotion": emotion,
        "emotion_intensity": intensity,
        "estimated_age_group": "dorosły",
        "estimated_gender": gender,
        "urgency": urgency,
        "needs_calming": angry or scared or intensity >= 7,
    }
