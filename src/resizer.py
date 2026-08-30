from __future__ import annotations

import subprocess
from pathlib import Path

from src.encoder import resolve_video_encoder, video_encoder_args


def resize_clip_for_vertical(
    input_path: str | Path,
    output_path: str | Path,
    mode: str = "crop",
    *,
    encoder: str = "auto",
    start: float | None = None,
    duration: float | None = None,
) -> Path:
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
    if start is not None and start < 0:
        raise ValueError("Le début du clip ne peut pas être négatif.")
    if duration is not None and duration <= 0:
        raise ValueError("La durée du clip doit être positive.")
    resolved_encoder = resolve_video_encoder(encoder)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        command.extend(["-ss", str(start)])
    command.extend(["-i", str(source)])
    if duration is not None:
        command.extend(["-t", str(duration)])
    command.extend([
        "-vf", filters[mode], *video_encoder_args(resolved_encoder), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(destination),
    ])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Échec du redimensionnement FFmpeg.")
    return destination


def resize_clip_for_tiktok(input_path: str | Path, output_path: str | Path) -> Path:
    return resize_clip_for_vertical(input_path, output_path, mode="crop")
