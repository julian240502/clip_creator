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


def video_encoder_args(encoder: str) -> list[str]:
    if encoder == "h264_nvenc":
        return ["-c:v", encoder, "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "21", "-b:v", "0"]
    if encoder == "h264_qsv":
        return ["-c:v", encoder, "-preset", "faster", "-global_quality", "21"]
    if encoder == "h264_amf":
        return ["-c:v", encoder, "-quality", "speed", "-rc", "cqp", "-qp_i", "20", "-qp_p", "22", "-qp_b", "24"]
    if encoder == "libx264":
        return ["-c:v", encoder, "-preset", "veryfast", "-crf", "20"]
    raise ValueError(f"Encodeur FFmpeg non pris en charge : {encoder}")


def encoder_label(encoder: str) -> str:
    return ENCODER_LABELS.get(encoder, encoder)
