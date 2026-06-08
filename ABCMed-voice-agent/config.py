from dataclasses import dataclass, field


@dataclass
class ASRConfig:
    model_size: str = "small"  # lepszy polski niż base
    language: str = "pl"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    vad_threshold: float = 0.5
    min_silence_duration_ms: int = 500


@dataclass
class LLMConfig:
    model: str = "llama3.1:8b"
    fallback_model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 300
    use_llm_emotion: bool = True
    num_ctx: int = 4096
    timeout_seconds: float = 45.0


@dataclass
class TTSConfig:
    # auto | macos | pyttsx3 | edge | chatterbox
    # auto: macOS say on Darwin, pyttsx3 on Windows/Linux, edge as fallback
    backend: str = "auto"
    # macos: "Zosia" | pyttsx3: system voice name | edge: "pl-PL-ZofiaNeural"
    voice: str = "Zosia"
    device: str = "cpu"
    sample_rate: int = 24000


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format_width: int = 2


@dataclass
class LiveConfig:
    silence_seconds: float = 0.85  # dłuższa pauza — nie ucina końcówki wypowiedzi
    min_speech_seconds: float = 0.35  # wymaga minimalnej długości mowy przed ASR
    speech_rms_threshold: int = 80  # wyższy próg — mniej fałszywych wykryć z szumu
    min_peak_rms_multiplier: float = 1.5  # szczyt RMS musi przekroczyć próg × ten mnożnik
    min_avg_logprob: float = -1.0  # odrzuć transkrypcję poniżej tej pewności Whisper
    silence_hallucination_words: tuple[str, ...] = (
        "cześć",
        "czesc",
        "dziękuję",
        "dziekuje",
        "hej",
        "halo",
        "tak",
        "nie",
        "proszę",
        "prosze",
    )
    silence_hallucination_max_duration_s: float = 1.5
    silence_hallucination_max_rms: int = 140
    status_log_interval_s: float = 3.0  # okresowy log RMS w konsoli
    max_utterance_seconds: float = 8.0  # wymuś koniec wypowiedzi mimo braku ciszy
    relative_silence_drop_ratio: float = 0.35  # spadek RMS od szczytu = koniec mowy
    relative_silence_seconds: float = 0.6  # ile trwa „względna cisza” zanim kończymy


@dataclass
class AppConfig:
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
