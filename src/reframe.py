"""Recadrage dynamique qui suit le visage (jalon D).

`mediapipe` + `opencv` sont des dépendances optionnelles
(`pip install -r requirements-reframe.txt`). Sans elles, on retombe sur un
recadrage centré. Seul `detect_face_track()` a besoin de ces paquets ; le reste
(maths de crop, génération du script sendcmd) est en bibliothèque standard.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_FPS = 2.0        # fréquence d'analyse des visages
KEYFRAME_STEP = 0.2     # pas des commandes sendcmd (s)
EMA_ALPHA = 0.18        # lissage (plus petit = plus doux)
DEADZONE = 0.012        # ignore les micro-déplacements (fraction de la largeur)


@dataclass(frozen=True)
class CropPath:
    crop_w: int
    crop_h: int
    events: list[tuple[float, int, int]] = field(default_factory=list)  # (t, x, y) coin haut-gauche

    @property
    def x0(self) -> int:
        return self.events[0][1] if self.events else 0

    @property
    def y0(self) -> int:
        return self.events[0][2] if self.events else 0


def reframe_available() -> bool:
    return importlib.util.find_spec("cv2") is not None


def _even(value: float) -> int:
    number = int(round(value))
    return number - (number % 2)


def crop_box(source_w: int, source_h: int, target_w: int, target_h: int) -> tuple[int, int]:
    """Plus grand rectangle au ratio cible qui tient dans la source."""
    target_ratio = target_w / target_h
    if source_w / source_h > target_ratio:  # source plus large : on borne la largeur
        crop_h = source_h
        crop_w = source_h * target_ratio
    else:
        crop_w = source_w
        crop_h = source_w / target_ratio
    return min(_even(crop_w), source_w - source_w % 2), min(_even(crop_h), source_h - source_h % 2)


def detect_face_track(
    video_path: str | Path, *, sample_fps: float = SAMPLE_FPS, downscale: int = 400,
) -> list[tuple[float, float | None]]:
    """[(t, cx_normalisé | None)] — position horizontale du plus grand visage frontal.

    Les images échantillonnées sont extraites via FFmpeg (rapide), pas frame par
    frame en OpenCV (qui décoderait toute la vidéo).
    """
    import cv2

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    track: list[tuple[float, float | None]] = []
    with tempfile.TemporaryDirectory(prefix="reframe-") as tmp:
        pattern = str(Path(tmp) / "f_%06d.jpg")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
                "-vf", f"fps={sample_fps},scale={downscale}:-2", "-q:v", "5", pattern,
            ],
            capture_output=True, check=True,
        )
        for order, frame_path in enumerate(sorted(Path(tmp).glob("f_*.jpg"))):
            stamp = order / sample_fps
            image = cv2.imread(str(frame_path))
            if image is None:
                track.append((stamp, None))
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
            if len(faces):
                fx, _fy, fw, _fh = max(faces, key=lambda box: box[2] * box[3])
                track.append((stamp, min(max((fx + fw / 2) / image.shape[1], 0.0), 1.0)))
            else:
                track.append((stamp, None))
    return track


def _fill_gaps(track: list[tuple[float, float | None]]) -> list[tuple[float, float]]:
    known = [value for _t, value in track if value is not None]
    if not known:
        return [(t, 0.5) for t, _v in track]
    last = known[0]
    filled: list[tuple[float, float]] = []
    for stamp, value in track:
        if value is not None:
            last = value
        filled.append((stamp, last))
    return filled


def crop_path(
    track: list[tuple[float, float | None]],
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
    *,
    t_start: float = 0.0,
    t_end: float | None = None,
) -> CropPath:
    """Trajectoire de crop lissée, rebasée à t=0, coordonnées en pixels source."""
    top = (source_h - crop_h) // 2
    span = [
        (t, cx) for t, cx in _fill_gaps(track)
        if t >= t_start - 1.0 and (t_end is None or t <= t_end + 1.0)
    ] or [(t_start, 0.5)]
    max_x = source_w - crop_w
    smoothed = span[0][1]
    events: list[tuple[float, int, int]] = []
    next_emit = t_start
    for stamp, target in span:
        if abs(target - smoothed) > DEADZONE:
            smoothed += EMA_ALPHA * (target - smoothed)
        if stamp + 1e-6 >= next_emit:
            x = int(round(min(max(smoothed * source_w - crop_w / 2, 0), max_x)))
            events.append((round(max(stamp - t_start, 0.0), 3), x, top))
            next_emit += KEYFRAME_STEP
    if not events or events[0][0] > 0.0:
        first_x = int(round(min(max(span[0][1] * source_w - crop_w / 2, 0), max_x)))
        events.insert(0, (0.0, first_x, top))
    return CropPath(crop_w, crop_h, events)


def centred_crop_path(source_w: int, source_h: int, crop_w: int, crop_h: int) -> CropPath:
    return CropPath(crop_w, crop_h, [(0.0, (source_w - crop_w) // 2, (source_h - crop_h) // 2)])


def write_sendcmd(path: str | Path, crop: CropPath) -> Path:
    lines = [f"{t:.3f} crop x {x}, crop y {y};" for t, x, y in crop.events]
    destination = Path(path)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
