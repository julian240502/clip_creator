from __future__ import annotations

import subprocess
from pathlib import Path

from src.encoder import cuda_scaling_available, resolve_video_encoder, video_encoder_args
from src.quality import get_quality_preset


def resize_clip_for_vertical(
    input_path: str | Path,
    output_path: str | Path,
    mode: str = "crop",
    *,
    encoder: str = "auto",
    encoding_speed: str = "balanced",
    quality: str = "1080p",
    start: float | None = None,
    duration: float | None = None,
) -> Path:
    """Convertit un clip en 9:16 sans déformer l'image."""
    source, destination = Path(input_path), Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    preset = get_quality_preset(quality)
    width, height = preset.width, preset.height
    software_filters = {
        "crop": f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
        "fit": f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
    }
    if mode not in software_filters:
        raise ValueError("Le mode doit être 'crop' ou 'fit'.")
    if start is not None and start < 0:
        raise ValueError("Le début du clip ne peut pas être négatif.")
    if duration is not None and duration <= 0:
        raise ValueError("La durée du clip doit être positive.")
    resolved_encoder = resolve_video_encoder(encoder)
    use_cuda = (
        mode == "crop"
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
        if cuda:
            crop = (
                "crop=w=if(gte(a\\,9/16)\\,trunc(ih*9/16/2)*2\\,iw)"
                ":h=if(gte(a\\,9/16)\\,ih\\,trunc(iw*16/9/2)*2)"
            )
            video_filter = (
                f"{crop},format=nv12,hwupload_cuda,"
                f"scale_cuda=w={width}:h={height}:interp_algo=bicubic:format=nv12"
            )
        else:
            video_filter = software_filters[mode]
        command.extend([
            "-vf", video_filter,
            *video_encoder_args(resolved_encoder, encoding_speed),
        ])
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
    return resize_clip_for_vertical(input_path, output_path, mode="crop")
