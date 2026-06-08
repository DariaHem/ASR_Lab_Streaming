#!/usr/bin/env python3
"""Cross-platform ASR evaluation CLI — WAV folder vs references, no microphone needed."""

import argparse
import csv
import sys
import time
from pathlib import Path

import jiwer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig
from agent.asr import StreamingASR


def load_references(path: Path) -> dict[str, str]:
    """Load references from CSV (file,reference) or pipe-delimited (file_id|text)."""
    refs: dict[str, str] = {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return refs

    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_id = row.get("file") or row.get("file_id") or row.get("filename", "")
                reference = row.get("reference") or row.get("transcript") or row.get("text", "")
                if file_id and reference:
                    refs[file_id.strip()] = reference.strip()
        return refs

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            file_id, reference = line.split("|", 1)
            refs[file_id.strip()] = reference.strip()
    return refs


def resolve_audio_path(audio_dir: Path, file_id: str) -> Path | None:
    stem = Path(file_id).stem
    for ext in (".wav", ".WAV", ".flac", ".mp3"):
        candidate = audio_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        candidate = audio_dir / file_id
        if candidate.exists():
            return candidate
    return None


def evaluate(
    audio_dir: Path,
    references_path: Path,
    model_size: str,
    beam_size: int,
    output_path: Path,
) -> list[dict]:
    refs = load_references(references_path)
    if not refs:
        raise SystemExit(f"Brak referencji w pliku: {references_path}")

    config = AppConfig()
    config.asr.model_size = model_size
    config.asr.beam_size = beam_size
    asr = StreamingASR(config.asr, config.audio)
    asr.load_model()

    results: list[dict] = []
    for file_id, reference in sorted(refs.items()):
        audio_path = resolve_audio_path(audio_dir, file_id)
        if audio_path is None:
            print(f"[WARN] Brak pliku audio dla: {file_id}")
            continue

        t0 = time.perf_counter()
        hypothesis = asr.transcribe_file(str(audio_path))
        latency = time.perf_counter() - t0

        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        row = {
            "file": file_id,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "latency_s": round(latency, 3),
            "model": model_size,
            "beam_size": beam_size,
        }
        results.append(row)
        print(
            f"[{file_id}] WER={wer:.2%} CER={cer:.2%} "
            f"latency={latency:.2f}s — {hypothesis[:60]}..."
        )

    if not results:
        raise SystemExit("Brak wyników — sprawdź katalog audio i plik referencji.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    avg_wer = sum(r["wer"] for r in results) / len(results)
    avg_cer = sum(r["cer"] for r in results) / len(results)
    print(f"\nŚrednie: WER={avg_wer:.2%} CER={avg_cer:.2%} ({len(results)} plików)")
    print(f"Zapisano: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ewaluacja ASR: folder WAV vs referencje → WER/CER CSV"
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=PROJECT_ROOT / "labs" / "test_audio",
        help="Katalog z plikami WAV",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=PROJECT_ROOT / "labs" / "test_audio" / "references.csv",
        help="Plik referencji (CSV lub file_id|tekst)",
    )
    parser.add_argument("--model", default="small", help="Rozmiar modelu Whisper")
    parser.add_argument("--beam", type=int, default=5, help="beam_size")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "labs" / "results" / "asr_eval_results.csv",
        help="Ścieżka wyjściowa CSV",
    )
    args = parser.parse_args()

    evaluate(
        audio_dir=args.audio_dir,
        references_path=args.references,
        model_size=args.model,
        beam_size=args.beam,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
