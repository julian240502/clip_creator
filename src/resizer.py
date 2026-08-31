from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from src.encoder import (
    cuda_scaling_available,
    resolve_video_encoder,
    video_encoder_args,
)
from src.quality import frame_size
from src.reframe import crop_box
from src.video_splitter import get_video_resolution

ClipCallback = Callable[[Path], None]
BACKGROUNDS = {"blur", "black", "reframe"}


def _reframe_filter(source: Path, width: int, height: int, x_expr: str) -> str:
    src_w, src_h = get_video_resolution(source)
    crop_w, crop_h = crop_box(src_w, src_h, width, height)
    y0 = (src_h - crop_h) // 2
    # x est une expression du temps évaluée à chaque image -> mouvement continu.
    return (
        f"crop={crop_w}:{crop_h}:x='{x_expr}':y={y0},"
        f"scale={width}:{height},setsar=1,format=yuv420p"
    )


def _prepare_sidecar(sidecar: str | Path, run_dir: Path) -> str:
    path = Path(sidecar)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    if path.parent.resolve() != run_dir.resolve():
        shutil.copyfile(path, run_dir / path.name)
    return path.name


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
    aspect: str = "9:16",
    background: str = "blur",
    start: float | None = None,
    duration: float | None = None,
    captions_file: str | Path | None = None,
    reframe_x_expr: str | None = None,
) -> Path:
    """Recadre la vidéo source dans le cadre au ratio choisi (fond flou / bandes / visage)."""
    # FFmpeg tourne depuis le dossier de sortie : le chemin source doit être absolu.
    source, destination = Path(input_path).resolve(), Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {source}")
    if background not in BACKGROUNDS:
        raise ValueError("Le fond doit être 'blur', 'black' ou 'reframe'.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_dir = destination.parent
    captions_name = _prepare_sidecar(captions_file, run_dir) if captions_file is not None else None
    if background == "reframe" and not reframe_x_expr:
        raise ValueError("Le recadrage visage exige une expression x(t).")
    width, height = frame_size(quality, aspect)
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
            if captions_name:
                video_filter = video_filter.replace("[vout]", f",ass={captions_name}[vout]")
            command.extend([
                "-filter_complex", video_filter,
                "-map", "[vout]", "-map", "0:a?",
            ])
        elif background == "reframe":
            video_filter = _reframe_filter(source, width, height, reframe_x_expr)
            if captions_name:
                video_filter += f",ass={captions_name}"
            command.extend(["-vf", video_filter])
        else:
            video_filter = _black_background_filter(width, height, cuda=cuda)
            if captions_name:
                video_filter += f",format=yuv420p,ass={captions_name}"
            command.extend(["-vf", video_filter])
        command.extend(video_encoder_args(resolved_encoder, encoding_speed))
        if not cuda:
            command.extend(["-pix_fmt", "yuv420p"])
        command.extend([
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(destination),
        ])
        return command

    result = subprocess.run(
        build_command(use_cuda), capture_output=True, text=True, cwd=str(run_dir),
    )
    if result.returncode != 0 and use_cuda:
        # Certains formats d'entrée ne sont pas acceptés par la chaîne CUDA.
        # L'encodage NVENC est conservé, seul le filtre repasse sur le CPU.
        result = subprocess.run(
            build_command(False), capture_output=True, text=True, cwd=str(run_dir),
        )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Échec du redimensionnement FFmpeg.")
    return destination


def resize_clip_for_tiktok(input_path: str | Path, output_path: str | Path) -> Path:
    return resize_clip_for_vertical(input_path, output_path)


def segment_vertical(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    clip_length: int,
    window_start: float,
    window_end: float,
    encoder: str = "auto",
    encoding_speed: str = "fast",
    quality: str = "1080p",
    aspect: str = "9:16",
    background: str = "blur",
    captions_file: str | Path | None = None,
    reframe_x_expr: str | None = None,
    on_clip: ClipCallback | None = None,
) -> list[Path]:
    """Découpe + reformat en une seule passe FFmpeg (segment muxer).

    La source n'est décodée qu'une fois, le graphe de filtres et l'encodeur ne
    sont initialisés qu'une fois. `on_clip` est appelé au fil des segments écrits.
    """
    source, out = Path(input_path).resolve(), Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {source}")
    if background not in BACKGROUNDS:
        raise ValueError("Le fond doit être 'blur', 'black' ou 'reframe'.")
    span = window_end - window_start
    if span <= 0:
        raise ValueError("La fenêtre sélectionnée est vide.")
    if clip_length <= 0:
        raise ValueError("La durée d'un clip doit être positive.")
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("clip_*.mp4"):
        stale.unlink()
    width, height = frame_size(quality, aspect)
    resolved_encoder = resolve_video_encoder(encoder)

    captions_name = _prepare_sidecar(captions_file, out) if captions_file is not None else None
    if background == "reframe" and not reframe_x_expr:
        raise ValueError("Le recadrage visage exige une expression x(t).")

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-ss", str(window_start), "-i", str(source), "-t", str(span)]
    if background == "blur":
        video_filter = _blur_background_filter(width, height)
        if captions_name:
            video_filter = video_filter.replace("[vout]", f",ass={captions_name}[vout]")
        command += ["-filter_complex", video_filter, "-map", "[vout]", "-map", "0:a?"]
    elif background == "reframe":
        video_filter = _reframe_filter(source, width, height, reframe_x_expr)
        if captions_name:
            video_filter += f",ass={captions_name}"
        command += ["-vf", video_filter]
    else:
        video_filter = _black_background_filter(width, height, cuda=False)
        if captions_name:
            video_filter += f",format=yuv420p,ass={captions_name}"
        command += ["-vf", video_filter]
    # IDR forcé à chaque coupe + petite tolérance pour que le muxer tranche
    # exactement sur ces keyframes (sinon il rate la 1re coupe quand il y a de l'audio).
    command += ["-force_key_frames", f"expr:gte(t,n_forced*{clip_length})"]
    command += video_encoder_args(resolved_encoder, encoding_speed)
    command += [
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-f", "segment", "-segment_time", str(clip_length), "-segment_time_delta", "0.1",
        "-reset_timestamps", "1", "-segment_start_number", "1",
        "-segment_format", "mp4", "-segment_format_options", "movflags=+faststart",
        "clip_%03d.mp4",
    ]

    proc = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, cwd=str(out),
    )
    emitted: set[str] = set()
    clips: list[Path] = []
    while True:
        finished = proc.poll() is not None
        present = sorted(out.glob("clip_*.mp4"))
        ready = present if finished else present[:-1]
        for path in ready:
            if path.name not in emitted:
                emitted.add(path.name)
                clips.append(path)
                if on_clip is not None:
                    on_clip(path)
        if finished:
            break
        time.sleep(0.4)
    stderr = proc.communicate()[1]
    if proc.returncode != 0:
        raise RuntimeError(stderr.strip() or "Échec du découpage vertical FFmpeg.")
    return clips
