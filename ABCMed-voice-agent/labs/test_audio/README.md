# Pliki testowe ASR

Ten katalog służy do ewaluacji rozpoznawania mowy **bez mikrofonu** — działa na Windows, Linux i macOS.

## Co dodać

1. Nagraj lub wygeneruj pliki WAV (16 kHz mono zalecane, ale inne formaty też działają).
2. Utwórz plik referencji `references.csv`:

```csv
file,reference
sample_01,Dzień dobry, chciałbym umówić wizytę u internisty.
sample_02,Mam silny ból głowy od trzech dni.
```

Alternatywnie możesz użyć formatu pipe-delimited (`references.txt`):

```
sample_01|Dzień dobry, chciałbym umówić wizytę u internisty.
sample_02|Mam silny ból głowy od trzech dni.
```

Nazwa w kolumnie `file` (bez rozszerzenia) musi odpowiadać plikowi `sample_01.wav` itd.

## Uruchomienie

```bash
# Minimalne środowisko lab (bez PyAudio / Chatterbox)
pip install -r requirements-lab.txt

# Ewaluacja
python labs/asr_eval.py --audio-dir labs/test_audio --references labs/test_audio/references.csv --model small --beam 5 --output labs/results/asr_eval_results.csv
```

Wynik: plik CSV z WER, CER i czasem transkrypcji dla każdego nagrania.
