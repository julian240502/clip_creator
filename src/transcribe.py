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
import os
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.paths import TRANSCRIPTIONS_DIR

# Modèle Whisper. `large-v3-turbo` par défaut (rapide) ; `large-v3` complet est
# plus robuste sur audio bruité / timestamps — activable sans toucher au code.
DEFAULT_MODEL = os.environ.get("CLIP_CREATOR_WHISPER_MODEL", "").strip() or "large-v3-turbo"
ProgressCallback = Callable[[float, str], None]

# Réglages calibrés pour du long-form bruité (rediff de live : voix + son du jeu +
# alertes + musique). Voir README « Sous-titres incrustés ».
_VAD_PARAMETERS = {
    "threshold": 0.35,               # garde la parole plus faible sous le son du jeu
    "min_silence_duration_ms": 350,  # ne recolle pas par-dessus les vraies pauses
    "min_speech_duration_ms": 120,
    "speech_pad_ms": 200,
}
# Sur un mix voix + jeu la confiance du modèle baisse : sans assouplir ces seuils,
# des segments entiers sont jetés comme « pas de parole » (sous-titres manquants).
_DECODE_OPTIONS = {
    "word_timestamps": True,
    "vad_filter": True,
    "vad_parameters": _VAD_PARAMETERS,
    "no_speech_threshold": 0.4,
    "log_prob_threshold": -1.2,
    # Pas de conditionnement sur le texte précédent : sur une longue vidéo bruitée
    # une fenêtre ratée empoisonne toutes les suivantes (dérive + boucles).
    "condition_on_previous_text": False,
}
_BATCH_SIZE = max(1, int(os.environ.get("CLIP_CREATOR_WHISPER_BATCH", "8") or "8"))

# Pré-nettoyage de l'audio avant Whisper : coupe le grave, atténue le bruit
# stationnaire, égalise les niveaux. ~0 VRAM. CLIP_CREATOR_WHISPER_AUDIO_CLEAN=0
# pour désactiver.
_AUDIO_CLEAN_FILTER = "highpass=f=70,afftdn=nf=-20,dynaudnorm=f=150:g=12"

# Bump -> invalide les transcripts en cache produits avec d'anciens réglages.
_PIPELINE_VERSION = "2"


def _audio_clean_enabled() -> bool:
    return os.environ.get("CLIP_CREATOR_WHISPER_AUDIO_CLEAN", "1").strip().lower() not in {
        "0", "false", "off", "no",
    }


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


def unload_models() -> None:
    """Vide le cache des modèles Whisper -> libère la VRAM. À appeler avant une
    grosse étape LLM sur une carte serrée ; le modèle se rechargera au besoin."""
    import gc

    with _model_lock:
        _models.clear()
    gc.collect()


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


def _run_transcription(whisper, wav_path: str, language: str | None):
    """Transcrit `wav_path`. Pipeline **batché** d'abord : le VAD découpe en énoncés
    et chacun est décodé indépendamment → pas de dérive cumulée sur une longue
    vidéo. Repli sur le mode séquentiel si le batché échoue (indispo, VRAM…).
    """
    opts = {"language": language, **_DECODE_OPTIONS}
    try:
        from faster_whisper import BatchedInferencePipeline

        batched = BatchedInferencePipeline(model=whisper)
        return batched.transcribe(wav_path, batch_size=_BATCH_SIZE, **opts)
    except Exception:  # noqa: BLE001 - batché absent / OOM -> séquentiel, plus sobre
        return whisper.transcribe(wav_path, **opts)


def _source_key(
    video_path: Path, model: str, language: str | None,
    clip_range: tuple[float, float] | None = None,
) -> str:
    stat = video_path.stat()
    clean = "c" if _audio_clean_enabled() else "r"
    window = f"|{clip_range[0]:.1f}-{clip_range[1]:.1f}" if clip_range else ""
    raw = (
        f"{video_path.name}|{stat.st_size}|{int(stat.st_mtime)}|{model}|"
        f"{language or 'auto'}|v{_PIPELINE_VERSION}{clean}{window}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _extract_audio(
    video_path: Path, out_wav: Path, *, start: float | None = None, end: float | None = None,
) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None and start > 0:
        command += ["-ss", f"{start:.3f}"]           # seek rapide avant -i
    command += ["-i", str(video_path)]
    if end is not None and (start is None or end > start):
        command += ["-t", f"{end - (start or 0.0):.3f}"]
    command += ["-vn", "-ac", "1", "-ar", "16000"]
    if _audio_clean_enabled():
        command += ["-af", _AUDIO_CLEAN_FILTER]
    command += ["-c:a", "pcm_s16le", str(out_wav)]
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
    clip_range: tuple[float, float] | None = None,
    progress: ProgressCallback | None = None,
) -> Transcript:
    """Transcrit une vidéo en mots horodatés. Résultat mis en cache par source.

    `clip_range` (start, end) en secondes : ne transcrit **que** cette portion —
    indispensable sur une rediff de plusieurs heures dont on n'analyse qu'un
    extrait. Les horodatages renvoyés restent dans le temps absolu de la source.
    """
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {source}")
    if clip_range is not None and clip_range[1] <= clip_range[0]:
        clip_range = None
    report = progress or (lambda _value, _message: None)
    cache_root = Path(cache_dir or TRANSCRIPTIONS_DIR)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"{_source_key(source, model, language, clip_range)}.json"
    if cache and cache_file.is_file():
        report(1.0, "Transcription réutilisée depuis le cache.")
        return load_transcript(cache_file)

    if not transcription_available():
        raise RuntimeError(
            "faster-whisper n'est pas installé. "
            "Installez-le avec : pip install -r requirements-transcribe.txt"
        )

    offset = clip_range[0] if clip_range else 0.0
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        report(0.05, "Extraction de l'audio…")
        start = clip_range[0] if clip_range else None
        end = clip_range[1] if clip_range else None
        _extract_audio(source, wav, start=start, end=end)
        device, compute_type = _resolve_backend()
        report(0.15, f"Chargement du modèle {model} ({device})…")
        whisper = _load_model(model, device, compute_type)
        report(0.25, "Transcription en cours…")
        segment_iter, info = _run_transcription(whisper, str(wav), language)
        segments: list[TranscriptSegment] = []
        for segment in segment_iter:
            words = [
                Word(start=float(word.start) + offset, end=float(word.end) + offset, text=word.word)
                for word in (segment.words or [])
                if word.start is not None and word.end is not None
            ]
            segments.append(
                TranscriptSegment(
                    start=float(segment.start) + offset,
                    end=float(segment.end) + offset,
                    text=segment.text.strip(),
                    words=words,
                )
            )
            report(min(0.95, 0.25 + 0.7 * (segment.end / max(info.duration, 1e-6))), "Transcription en cours…")

    transcript = Transcript(
        language=info.language,
        duration=float(info.duration) + offset,
        model=model,
        segments=segments,
    )
    if cache:
        dump_transcript(transcript, cache_file)
    report(1.0, "Transcription terminée.")
    return transcript
