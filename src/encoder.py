from __future__ import annotations

import os
import subprocess
from functools import lru_cache


ENCODERS = {
    "cpu": "libx264",
    "nvidia": "h264_nvenc",
    "intel": "h264_qsv",
    "amd": "h264_amf",
}

ENCODER_LABELS = {
    "libx264": "CPU · x264",
    "h264_nvenc": "NVIDIA · NVENC",
    "h264_qsv": "Intel · Quick Sync",
    "h264_amf": "AMD · AMF",
}

ENCODING_SPEEDS = ("fast", "balanced", "quality")


@lru_cache(maxsize=None)
def _encoder_works(encoder: str) -> bool:
    """Vérifie que l'encodeur est compilé et utilisable avec le pilote présent."""
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=black:size=64x64:rate=1:duration=0.1",
        "-frames:v", "1", "-pix_fmt", "yuv420p", "-c:v", encoder,
        "-f", "null", os.devnull,
    ]
    try:
        return subprocess.run(command, capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@lru_cache(maxsize=1)
def available_hardware_encoders() -> tuple[str, ...]:
    return tuple(
        encoder for encoder in ("h264_nvenc", "h264_qsv", "h264_amf")
        if _encoder_works(encoder)
    )


@lru_cache(maxsize=1)
def cuda_scaling_available() -> bool:
    """Vérifie que FFmpeg peut transférer, redimensionner et encoder via CUDA."""
    if not _encoder_works("h264_nvenc"):
        return False
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=black:size=64x64:rate=1:duration=0.1",
        "-frames:v", "1",
        "-vf", "format=nv12,hwupload_cuda,scale_cuda=w=64:h=64:format=nv12",
        "-c:v", "h264_nvenc", "-preset", "p1", "-f", "null", os.devnull,
    ]
    try:
        return subprocess.run(command, capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@lru_cache(maxsize=1)
def cuda_blur_compositing_available() -> bool:
    """Vérifie la présence de l'overlay CUDA utilisé par le fond flouté."""
    if not cuda_scaling_available():
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "overlay_cuda" in result.stdout


def resolve_video_encoder(preference: str = "auto") -> str:
    if preference == "auto":
        available = available_hardware_encoders()
        return available[0] if available else "libx264"
    if preference not in ENCODERS:
        raise ValueError("Encodeur inconnu. Utilisez auto, cpu, nvidia, intel ou amd.")
    encoder = ENCODERS[preference]
    if encoder != "libx264" and not _encoder_works(encoder):
        raise RuntimeError(f"{ENCODER_LABELS[encoder]} n'est pas utilisable avec FFmpeg et le pilote installés.")
    return encoder


def video_encoder_args(encoder: str, speed: str = "balanced") -> list[str]:
    if speed not in ENCODING_SPEEDS:
        raise ValueError("Vitesse inconnue. Utilisez fast, balanced ou quality.")
    if encoder == "h264_nvenc":
        settings = {
            "fast": ("p1", "23"),
            "balanced": ("p4", "21"),
            "quality": ("p6", "19"),
        }
        preset, cq = settings[speed]
        return ["-c:v", encoder, "-preset", preset, "-tune", "hq", "-rc", "vbr", "-cq", cq, "-b:v", "0"]
    if encoder == "h264_qsv":
        settings = {
            "fast": ("veryfast", "23"),
            "balanced": ("faster", "21"),
            "quality": ("medium", "19"),
        }
        preset, quality = settings[speed]
        return ["-c:v", encoder, "-preset", preset, "-global_quality", quality]
    if encoder == "h264_amf":
        settings = {
            "fast": ("speed", "23", "25", "27"),
            "balanced": ("balanced", "20", "22", "24"),
            "quality": ("quality", "18", "20", "22"),
        }
        quality, qp_i, qp_p, qp_b = settings[speed]
        return ["-c:v", encoder, "-quality", quality, "-rc", "cqp", "-qp_i", qp_i, "-qp_p", qp_p, "-qp_b", qp_b]
    if encoder == "libx264":
        settings = {
            "fast": ("ultrafast", "23"),
            "balanced": ("veryfast", "20"),
            "quality": ("medium", "18"),
        }
        preset, crf = settings[speed]
        return ["-c:v", encoder, "-preset", preset, "-crf", crf]
    raise ValueError(f"Encodeur FFmpeg non pris en charge : {encoder}")


def encoder_label(encoder: str) -> str:
    return ENCODER_LABELS.get(encoder, encoder)
