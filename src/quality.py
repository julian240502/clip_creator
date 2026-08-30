from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityPreset:
    key: str
    label: str
    width: int
    height: int
    source_max_height: int


QUALITY_PRESETS = {
    "720p": QualityPreset("720p", "HD · 720p", 720, 1280, 720),
    "1080p": QualityPreset("1080p", "Full HD · 1080p", 1080, 1920, 1080),
    "4k": QualityPreset("4k", "4K · 2160p", 2160, 3840, 2160),
}


def get_quality_preset(quality: str) -> QualityPreset:
    try:
        return QUALITY_PRESETS[quality]
    except KeyError as exc:
        choices = ", ".join(QUALITY_PRESETS)
        raise ValueError(f"Qualité inconnue. Choisissez parmi : {choices}.") from exc
