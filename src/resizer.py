from __future__ import annotations

import subprocess
from pathlib import Path

from src.encoder import (
    cuda_scaling_available,
    resolve_video_encoder,
    video_encoder_args,
)
from src.quality import get_quality_preset


def _black_background_filter(width: int, height: int, *, cuda: bool) -> str:
    if cuda:
        return (
            "format=nv12,hwupload_cuda,"
            f"scale_cuda=w={width}:h={height}:force_original_aspect_ratio=decrease:"
            "force_divisible_by=2:interp_algo=bicubic:format=nv12,"
            "hwdownload,format=nv12,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
    )


def _blur_background_filter(width: int, height: int) -> str:
    preview_width = max(180, width // 4)
    preview_height = max(320, height // 4)
    # Un flou modéré garde le fond identifiable sans concurrencer le premier plan.
    blur_radius = max(6, preview_width // 28)
    return (
        "[0:v]split=2[background_source][foreground_source];"
        f"[background_source]scale={preview_width}:{preview_height}:"
        "force_original_aspect_ratio=increase,"
        f"crop={preview_width}:{preview_height},boxblur={blur_radius}:2,"
        f"scale={width}:{height}[background];"
        f"[foreground_source]scale={width}:{height}:"
        "force_original_aspect_ratio=decrease[foreground];"
        "[background][foreground]overlay=(W-w)/2:(H-h)/2,format=yuv420p[vout]"
    )


def resize_clip_for_vertical(
    input_path: str | Path,
    output_path: str | Path,
    *,
    encoder: str = "auto",
    encoding_speed: str = "balanced",
    quality: str = "1080p",
    background: str = "blur",
    start: float | None = None,
    duration: float | None = None,
) -> Path:
    """Place une vidéo paysage entière dans un cadre 9:16, sans recadrage."""
    source, destination = Path(input_path), Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    preset = get_quality_preset(quality)
    width, height = preset.width, preset.height
    if background not in {"blur", "black"}:
        raise ValueError("Le fond doit être 'blur' ou 'black'.")
    if start is not None and start < 0:
        raise ValueError("Le début du clip ne peut pas être négatif.")
    if duration is not None and duration <= 0:
        raise ValueError("La durée du clip doit être positive.")
    resolved_encoder = resolve_video_encoder(encoder)
    # overlay_cuda produit des zones vertes avec certains builds Windows.
    # Le fond flouté reste en filtres logiciels, tout en conservant NVENC.
    use_cuda = (
        background == "black"
        and resolved_encoder == "h264_nvenc"
        and cuda_scaling_available()
    )

    def build_command(cuda: bool) -> list[str]:
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        if start is not None:
            command.extend(["-ss", str(start)])
        command.extend(["-i", str(source)])
        if duration is not None:
            command.extend(["-t", str(duration)])
        if background == "blur":
            video_filter = _blur_background_filter(width, height)
            command.extend([
                "-filter_complex", video_filter,
                "-map", "[vout]", "-map", "0:a?",
            ])
        else:
            video_filter = _black_background_filter(width, height, cuda=cuda)
            command.extend(["-vf", video_filter])
        command.extend(video_encoder_args(resolved_encoder, encoding_speed))
        if not cuda:
            command.extend(["-pix_fmt", "yuv420p"])
        command.extend([
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(destination),
        ])
        return command

    result = subprocess.run(build_command(use_cuda), capture_output=True, text=True)
    if result.returncode != 0 and use_cuda:
        # Certains formats d'entrée ne sont pas acceptés par la chaîne CUDA.
        # L'encodage NVENC est conservé, seul le filtre repasse sur le CPU.
        result = subprocess.run(build_command(False), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Échec du redimensionnement FFmpeg.")
    return destination


def resize_clip_for_tiktok(input_path: str | Path, output_path: str | Path) -> Path:
    return resize_clip_for_vertical(input_path, output_path)
