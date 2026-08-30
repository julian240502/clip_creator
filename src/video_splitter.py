from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path

from src.encoder import resolve_video_encoder, video_encoder_args

ClipCallback = Callable[[Path], None]


def get_video_duration(video_path: str | Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Impossible d'analyser la vidéo.")
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Durée vidéo illisible.") from exc


def get_video_resolution(video_path: str | Path) -> tuple[int, int]:
    """Retourne (largeur, hauteur) du premier flux vidéo."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(video_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Impossible d'analyser la vidéo.")
    try:
        stream = json.loads(result.stdout)["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Résolution vidéo illisible.") from exc


def resolve_source_window(
    full_duration: float,
    source_start: float | None,
    source_end: float | None,
) -> tuple[float, float]:
    """Borne la portion demandée à la durée réelle et valide qu'elle n'est pas vide."""
    start_at = max(0.0, source_start or 0.0)
    end_at = full_duration if source_end is None else min(float(source_end), full_duration)
    if end_at - start_at <= 0:
        raise ValueError("La portion sélectionnée est vide.")
    return start_at, end_at


def split_video(
    video_path: str | Path,
    clip_length: int,
    output_dir: str | Path,
    encoder: str = "auto",
    encoding_speed: str = "balanced",
    *,
    source_start: float | None = None,
    source_end: float | None = None,
    on_clip: ClipCallback | None = None,
) -> list[str]:
    """Découpe précisément une vidéo avec un encodage compatible éditeurs mobiles."""
    source, destination = Path(video_path), Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {source}")
    if clip_length <= 0:
        raise ValueError("La durée d'un clip doit être positive.")
    destination.mkdir(parents=True, exist_ok=True)
    start_at, end_at = resolve_source_window(
        get_video_duration(source), source_start, source_end
    )
    span = end_at - start_at
    resolved_encoder = resolve_video_encoder(encoder)
    clips: list[str] = []
    for index in range(math.ceil(span / clip_length)):
        start = start_at + index * clip_length
        length = min(clip_length, end_at - start)
        output_path = destination / f"clip_{index + 1:03d}.mp4"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
            "-i", str(source), "-t", str(length),
            *video_encoder_args(resolved_encoder, encoding_speed),
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"Échec du clip {index + 1}.")
        clips.append(str(output_path))
        if on_clip is not None:
            on_clip(output_path)
    return clips
