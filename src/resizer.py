from __future__ import annotations

import subprocess
from pathlib import Path


def resize_clip_for_vertical(input_path: str | Path, output_path: str | Path, mode: str = "crop") -> Path:
    """Convertit un clip en 9:16 sans déformer l'image."""
    source, destination = Path(input_path), Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    filters = {
        "crop": "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "fit": "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
    }
    if mode not in filters:
        raise ValueError("Le mode doit être 'crop' ou 'fit'.")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-vf", filters[mode], "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Échec du redimensionnement FFmpeg.")
    return destination


def resize_clip_for_tiktok(input_path: str | Path, output_path: str | Path) -> Path:
    return resize_clip_for_vertical(input_path, output_path, mode="crop")
