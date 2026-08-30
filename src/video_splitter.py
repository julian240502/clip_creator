from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path


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


def split_video(video_path: str | Path, clip_length: int, output_dir: str | Path) -> list[str]:
    """Découpe précisément une vidéo avec un encodage compatible éditeurs mobiles."""
    source, destination = Path(video_path), Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {source}")
    if clip_length <= 0:
        raise ValueError("La durée d'un clip doit être positive.")
    destination.mkdir(parents=True, exist_ok=True)
    duration = get_video_duration(source)
    clips = []
    for index in range(math.ceil(duration / clip_length)):
        start, length = index * clip_length, min(clip_length, duration - index * clip_length)
        output_path = destination / f"clip_{index + 1:03d}.mp4"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
            "-i", str(source), "-t", str(length), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"Échec du clip {index + 1}.")
        clips.append(str(output_path))
    return clips
