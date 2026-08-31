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


# Formats d'export : ratio largeur:hauteur + usage principal.
ASPECTS: dict[str, tuple[int, int]] = {
    "9:16": (9, 16),
    "4:5": (4, 5),
    "1:1": (1, 1),
    "16:9": (16, 9),
}
ASPECT_USAGE = {
    "9:16": "TikTok, Reels, Shorts",
    "4:5": "Fil Instagram / Facebook",
    "1:1": "Post carré, LinkedIn",
    "16:9": "YouTube, paysage",
}


def _even(value: float) -> int:
    number = int(round(value))
    return number - (number % 2)


def frame_size(quality: str, aspect: str) -> tuple[int, int]:
    """Dimensions du cadre d'export pour une qualité et un ratio donnés."""
    short = get_quality_preset(quality).source_max_height
    ratio_w, ratio_h = ASPECTS.get(aspect, ASPECTS["9:16"])
    if ratio_w <= ratio_h:  # portrait ou carré : la largeur est le petit côté
        return _even(short), _even(short * ratio_h / ratio_w)
    return _even(short * ratio_w / ratio_h), _even(short)  # paysage
