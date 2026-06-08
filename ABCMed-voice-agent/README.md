# ABCMed — Asystent Głosowy Call Center (Placówka Medyczna)

Demo systemu call center AI łączącego:
- **ASR streaming** (Faster-Whisper + VAD)
- **LLM** (Ollama — lokalne modele, cross-platform)
- **TTS** (auto: macOS say / pyttsx3 / edge-tts / opcjonalnie Chatterbox)
- **Analiza emocji** (tekst + cechy głosu via librosa)

Scenariusz: pacjent dzwoni do placówki medycznej. System automatycznie:
1. Rozpoznaje mowę w czasie rzeczywistym
2. Monitoruje emocje, szacuje wiek/płeć
3. Uspokaja zdenerwowanego pacjenta
4. Prowadzi rozmowę wstępną
5. Umawia wizytę u specjalisty lub przekazuje do operatora

**Wszystko działa lokalnie, offline (z wyjątkiem edge-tts), za darmo.**

---

## Wymagania systemowe

- **Windows 10/11**, **Linux** (Ubuntu/Debian), lub **macOS**
- Python 3.10+ (zalecany 3.11)
- ~8 GB RAM (dla modeli)
- ~10 GB miejsca na dysku (modele LLM + Whisper)

---

## Tryby pracy

| Tryb | Mikrofon | Platforma | Instalacja |
|------|----------|-----------|------------|
| **Lab / ewaluacja ASR** | Nie | Windows, Linux, macOS | `requirements-lab.txt` |
| **Tekstowy** (`main.py text`) | Nie | Wszystkie | `requirements-lab.txt` lub `requirements.txt` |
| **Web UI** (`main.py web`) | Opcjonalnie | Wszystkie | `requirements.txt` |
| **Live głosowy** (`main.py live`) | Tak | Wszystkie (wymaga PyAudio) | `requirements.txt` + PyAudio |

---

## Instalacja — macOS

### 1. Ollama (lokalny LLM)

```bash
brew install ollama
ollama serve          # osobny terminal
ollama pull llama3.1:8b
```

### 2. Python i zależności

```bash
cd ~/Projects/medical-voice-agent
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Pełna instalacja (live + web)
pip install -r requirements.txt
brew install portaudio   # dla PyAudio / mikrofonu

# Lub minimalna (lab, bez mikrofonu)
pip install -r requirements-lab.txt
```

### 3. TTS

Domyślnie `backend="auto"` wybiera **macOS say** (głos Zosia). Nie wymaga dodatkowych pakietów.

---

## Instalacja — Linux (Ubuntu/Debian)

### 1. Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull llama3.1:8b
```

### 2. Python i zależności

```bash
cd ~/Projects/medical-voice-agent
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Zależności systemowe
sudo apt update
sudo apt install -y python3-dev portaudio19-dev libsndfile1 ffmpeg

# Pełna instalacja
pip install -r requirements.txt
pip install pyaudio

# Lub minimalna (lab)
pip install -r requirements-lab.txt
```

### 3. TTS

`backend="auto"` wybiera **pyttsx3** (głosy systemowe). Fallback: **edge-tts** (wymaga internetu, dobra jakość polskiego).

---

## Instalacja — Windows

### 1. Ollama

Pobierz instalator z [https://ollama.ai/download](https://ollama.ai/download), uruchom `ollama serve` i:

```powershell
ollama pull llama3.1:8b
```

### 2. Python i zależności

```powershell
cd %USERPROFILE%\Projects\medical-voice-agent
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip

# Pełna instalacja
pip install -r requirements.txt
pip install pyaudio

# Lub minimalna (lab, bez mikrofonu)
pip install -r requirements-lab.txt
```

Jeśli `pip install pyaudio` się nie powiedzie, pobierz wheel z [https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

### 3. TTS

`backend="auto"` wybiera **pyttsx3** (głosy SAPI5). Fallback: **edge-tts**.

---

## Uruchomienie

### Tryb laboratoryjny (bez mikrofonu)

```bash
source venv/bin/activate   # Windows: venv\Scripts\activate
python labs/asr_eval.py --audio-dir labs/test_audio --references labs/test_audio/references.csv
```

Dodaj pliki WAV i `references.csv` według `labs/test_audio/README.md`.

### Tryb tekstowy (najszybszy start)

```bash
python main.py text
```

### Tryb Web UI (Gradio)

```bash
python main.py web
```

Otwórz http://localhost:7860

### Tryb głosowy live (mikrofon)

```bash
python main.py live
```

Wymaga PyAudio. Bez niego aplikacja wyświetli komunikat z instrukcją instalacji i zasugeruje tryb tekstowy lub `labs/asr_eval.py`.

---

## Konfiguracja TTS

Edytuj `config.py` → `TTSConfig`:

```python
backend: str = "auto"    # auto | macos | pyttsx3 | edge | chatterbox
voice: str = "Zosia"     # macos: Zosia | edge: pl-PL-ZofiaNeural
```

| Backend | Platforma | Uwagi |
|---------|-----------|-------|
| `auto` | Wszystkie | macOS say → pyttsx3 → edge |
| `macos` | macOS only | Najszybszy, offline |
| `pyttsx3` | Win/Linux/mac | Głosy systemowe, offline |
| `edge` | Wszystkie | Wymaga internetu, dobra jakość PL |
| `chatterbox` | Wszystkie | ~2 GB download, najwyższa jakość |

---

## Architektura

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Mikrofon   │───▶│ Faster-Whisper│───▶│   Ollama    │
│  (PyAudio)  │    │  (ASR + VAD) │    │   (LLM)     │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                              │
┌─────────────┐    ┌──────────────┐           │
│  Głośnik    │◀───│  TTS (auto)  │◀──────────┘
│  (PyAudio)  │    │ say/pyttsx3/ │
└─────────────┘    │    edge      │
                   └──────────────┘
```

## Struktura plików

```
medical-voice-agent/
├── main.py              # Entry point (text/live/web)
├── app.py               # Gradio Web UI
├── config.py            # Konfiguracja
├── requirements.txt     # Pełne zależności
├── requirements-lab.txt # Minimalne (lab, bez mikrofonu)
├── agent/
│   ├── pipeline.py      # Orkiestracja ASR→LLM→TTS
│   ├── asr.py           # Faster-Whisper + VAD
│   ├── llm.py           # Ollama + analiza emocji
│   ├── tts.py           # Cross-platform TTS
│   ├── live_session.py  # Sesja live z fallbackiem
│   └── audio_io.py      # Mikrofon / głośnik
├── labs/
│   ├── asr_eval.py      # CLI ewaluacji WER/CER
│   └── test_audio/      # Pliki WAV + references.csv
└── medical/
    ├── prompts.py
    ├── scheduler.py
    └── specialties.py
```

## Scenariusze demo

1. **Spokojny pacjent**: "Dzień dobry, chciałbym umówić się do internisty na badania kontrolne"
2. **Zdenerwowany pacjent**: "Halo?! Czekam już 3 godziny na połączenie!"
3. **Nagły przypadek**: "Proszę pomocy, mam silny ból w klatce piersiowej"
4. **Żądanie operatora**: "Chcę rozmawiać z człowiekiem, nie z maszyną"

---

## Troubleshooting

| Problem | Rozwiązanie |
|---------|-------------|
| `ollama: command not found` | Zainstaluj Ollama (patrz sekcja dla Twojego OS) |
| Ollama nie odpowiada | Uruchom `ollama serve` w osobnym terminalu |
| PyAudio error (live) | Zainstaluj portaudio + pyaudio (patrz README dla OS) |
| Brak TTS na Linux/Win | `pip install pyttsx3 edge-tts` lub ustaw `backend="edge"` |
| edge-tts timeout | Sprawdź połączenie internetowe |
| Whisper wolny | Zmień `model_size` na `tiny` w config.py |
| Brak polskiego w LLM | Użyj `llama3.1:8b` |
