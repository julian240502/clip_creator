"""Transcription mot-à-mot via faster-whisper (optionnel).

Le paquet `faster-whisper` n'est pas une dépendance de base : il s'installe avec
`pip install -r requirements-transcribe.txt`. Toutes les fonctions de ce module
sauf `transcribe()` fonctionnent sans lui ; `transcribe()` lève une erreur claire
s'il manque.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.paths import TRANSCRIPTIONS_DIR

DEFAULT_MODEL = "large-v3-turbo"
ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass(frozen=True)
class Transcript:
    language: str
    duration: float
    model: str
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def words(self) -> list[Word]:
        return [word for segment in self.segments for word in segment.words]

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments).strip()


def transcription_available() -> bool:
    """True si faster-whisper est installé."""
    return importlib.util.find_spec("faster_whisper") is not None


_model_lock = threading.Lock()
_models: dict[tuple[str, str, str], object] = {}


def _load_model(model: str, device: str, compute_type: str):
    """Garde le modèle en mémoire : le chargement (long) n'a lieu qu'une fois par session."""
    key = (model, device, compute_type)
    with _model_lock:
        if key not in _models:
            from faster_whisper import WhisperModel

            _models[key] = WhisperModel(model, device=device, compute_type=compute_type)
        return _models[key]


def prewarm_model(model: str = DEFAULT_MODEL) -> None:
    """Charge le modèle ET initialise les kernels CUDA (1re inférence).

    À lancer dans un thread : masque le délai pendant que l'utilisateur règle le style.
    """
    if not transcription_available():
        return
    try:
        whisper = _load_model(model, *_resolve_backend())
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "warmup.wav"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0.5",
                    "-c:a", "pcm_s16le", str(wav),
                ],
                capture_output=True, check=True,
            )
            segments, _ = whisper.transcribe(str(wav), vad_filter=False)
            list(segments)
    except Exception:  # noqa: BLE001 - le préchauffage est best-effort
        pass


def _resolve_backend() -> tuple[str, str]:
    """(device, compute_type) — CUDA si une carte est visible, sinon CPU."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:  # noqa: BLE001 - CUDA absent ou pilote cassé -> repli CPU
        pass
    return "cpu", "int8"


def _source_key(video_path: Path, model: str, language: str | None) -> str:
    stat = video_path.stat()
    raw = f"{video_path.name}|{stat.st_size}|{int(stat.st_mtime)}|{model}|{language or 'auto'}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _extract_audio(video_path: Path, out_wav: Path) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(out_wav),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Extraction audio impossible.")


def dump_transcript(transcript: Transcript, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": transcript.language,
        "duration": transcript.duration,
        "model": transcript.model,
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [
                    {"start": word.start, "end": word.end, "text": word.text}
                    for word in segment.words
                ],
            }
            for segment in transcript.segments
        ],
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def load_transcript(path: str | Path) -> Transcript:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Transcript(
        language=data["language"],
        duration=data["duration"],
        model=data.get("model", ""),
        segments=[
            TranscriptSegment(
                start=segment["start"],
                end=segment["end"],
                text=segment["text"],
                words=[Word(**word) for word in segment.get("words", [])],
            )
            for segment in data["segments"]
        ],
    )


def transcribe(
    video_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    cache_dir: str | Path | None = None,
    cache: bool = True,
    progress: ProgressCallback | None = None,
) -> Transcript:
    """Transcrit une vidéo en mots horodatés. Résultat mis en cache par source."""
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {source}")
    report = progress or (lambda _value, _message: None)
    cache_root = Path(cache_dir or TRANSCRIPTIONS_DIR)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"{_source_key(source, model, language)}.json"
    if cache and cache_file.is_file():
        report(1.0, "Transcription réutilisée depuis le cache.")
        return load_transcript(cache_file)

    if not transcription_available():
        raise RuntimeError(
            "faster-whisper n'est pas installé. "
            "Installez-le avec : pip install -r requirements-transcribe.txt"
        )

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        report(0.05, "Extraction de l'audio…")
        _extract_audio(source, wav)
        device, compute_type = _resolve_backend()
        report(0.15, f"Chargement du modèle {model} ({device})…")
        whisper = _load_model(model, device, compute_type)
        report(0.25, "Transcription en cours…")
        segment_iter, info = whisper.transcribe(
            str(wav), language=language, word_timestamps=True, vad_filter=True,
        )
        segments: list[TranscriptSegment] = []
        for segment in segment_iter:
            words = [
                Word(start=float(word.start), end=float(word.end), text=word.word)
                for word in (segment.words or [])
                if word.start is not None and word.end is not None
            ]
            segments.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    words=words,
                )
            )
            report(min(0.95, 0.25 + 0.7 * (segment.end / max(info.duration, 1e-6))), "Transcription en cours…")

    transcript = Transcript(
        language=info.language,
        duration=float(info.duration),
        model=model,
        segments=segments,
    )
    if cache:
        dump_transcript(transcript, cache_file)
    report(1.0, "Transcription terminée.")
    return transcript
