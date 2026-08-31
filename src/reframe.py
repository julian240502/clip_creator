"""Recadrage dynamique qui suit le visage (jalon D).

`opencv` est une dépendance optionnelle (`pip install -r requirements-reframe.txt`).
Sans elle, on retombe sur un recadrage centré. Seul `detect_face_track()` en a
besoin ; le reste (maths de crop, expression FFmpeg) est en bibliothèque standard.

Le mouvement est piloté par une expression `x(t)` linéaire par morceaux passée
au filtre `crop` (évaluée à chaque image) plutôt que par des commandes `sendcmd`
posées toutes les 0,2 s — ce qui supprime l'effet d'escalier.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

_PTS_RE = re.compile(r"pts_time:\s*([0-9.]+)")

SAMPLE_FPS = 1.0          # fréquence d'analyse des visages
DOWNSCALE = 320           # largeur d'analyse (px)
RESAMPLE_HZ = 10.0        # rééchantillonnage de la trajectoire avant lissage
SMOOTH_WINDOW_S = 1.2     # fenêtre du lissage symétrique (double passe)
MAX_VEL_FRAC = 0.10       # vitesse max du cadre : fraction de largeur / seconde


@dataclass(frozen=True)
class CropPath:
    crop_w: int
    crop_h: int
    y0: int
    x_expr: str                                        # expression FFmpeg pour `crop:x`
    points: list[tuple[float, int]] = field(default_factory=list)  # (t rebasé, x) de contrôle

    @property
    def x0(self) -> int:
        return self.points[0][1] if self.points else 0


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


def _extract_frames(video_path, width, height, sample_fps):
    """Rend (frames_gris, timestamps). D'abord les images-clés seules (décodage
    ~5-10× plus rapide), puis un échantillonnage régulier si trop peu d'images-clés."""
    import numpy as np

    frame_bytes = width * height

    def run(vfilter: str) -> tuple[list, list[float]]:
        log = Path(tempfile.gettempdir()) / f"reframe-{os.getpid()}-{abs(hash(vfilter)) % 9999}.log"
        with log.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "info", *pre, "-i", str(video_path),
                    "-an", "-vf", vfilter, "-vsync", "0", "-pix_fmt", "gray", "-f", "rawvideo", "-",
                ],
                stdout=subprocess.PIPE, stderr=handle,
            )
            raw = proc.stdout.read()
            proc.stdout.close()
            proc.wait()
        times = [float(x) for x in _PTS_RE.findall(log.read_text("utf-8", errors="ignore"))]
        log.unlink(missing_ok=True)
        count = len(raw) // frame_bytes
        frames = [
            np.frombuffer(raw[i * frame_bytes:(i + 1) * frame_bytes], np.uint8).reshape(height, width)
            for i in range(count)
        ]
        if len(times) < count:
            times = [i / sample_fps for i in range(count)]
        return frames, times[:count]

    pre = ["-skip_frame", "nokey"]
    frames, times = run(f"scale={width}:{height},showinfo")
    if len(frames) < 12:  # images-clés trop rares -> échantillonnage régulier
        pre = []
        frames, times = run(f"fps={sample_fps},scale={width}:{height},showinfo")
    return frames, times


def detect_face_track(
    video_path: str | Path, *, sample_fps: float = SAMPLE_FPS, downscale: int = DOWNSCALE,
) -> list[tuple[float, float | None]]:
    """[(t, cx_normalisé | None)] — position horizontale du plus grand visage frontal."""
    import cv2

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    src_w, src_h = (int(value) for value in probe.stdout.strip().split(","))
    width = downscale - downscale % 2
    height = _even(src_h * width / src_w)
    min_face = max(28, width // 12)

    frames, times = _extract_frames(video_path, width, height, sample_fps)

    def _face_cx(gray) -> float | None:
        faces = cascade.detectMultiScale(gray, 1.3, 4, minSize=(min_face, min_face))
        if not len(faces):
            return None
        fx, _fy, fw, _fh = max(faces, key=lambda box: box[2] * box[3])
        return min(max((fx + fw / 2) / width, 0.0), 1.0)

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        results = list(pool.map(_face_cx, frames))
    return list(zip(times, results, strict=True))


def _fill_gaps(track: list[tuple[float, float | None]]) -> list[tuple[float, float]]:
    if not any(value is not None for _t, value in track):
        return [(t, 0.5) for t, _v in track]
    first = next(value for _t, value in track if value is not None)
    last = first
    filled: list[tuple[float, float]] = []
    for stamp, value in track:
        if value is not None:
            last = value
        filled.append((stamp, last))
    return filled


def _resample(filled: list[tuple[float, float]], hz: float, lo: float, hi: float) -> list[float]:
    count = max(2, int((hi - lo) * hz) + 1)
    grid: list[float] = []
    cursor = 0
    for i in range(count):
        t = lo + i / hz
        while cursor + 1 < len(filled) and filled[cursor + 1][0] <= t:
            cursor += 1
        if cursor + 1 < len(filled):
            (ta, ca), (tb, cb) = filled[cursor], filled[cursor + 1]
            frac = 0.0 if tb == ta else min(max((t - ta) / (tb - ta), 0.0), 1.0)
            grid.append(ca + (cb - ca) * frac)
        else:
            grid.append(filled[-1][1])
    return grid


def _smooth(values: list[float], half: int) -> list[float]:
    out = list(values)
    for _ in range(2):  # double passe ≈ noyau gaussien
        src = list(out)
        for i in range(len(src)):
            window = src[max(0, i - half): i + half + 1]
            out[i] = sum(window) / len(window)
    return out


def _x_expression(points: list[tuple[float, int]], max_x: int) -> str:
    if len(points) == 1:
        return str(points[0][1])
    expr = str(points[-1][1])
    for (t0, x0), (t1, x1) in zip(reversed(points[:-1]), reversed(points[1:]), strict=True):
        gap = t1 - t0
        segment = str(x1) if gap < 1e-3 else f"({x0}+({x1 - x0})*(t-{t0:.3f})/{gap:.3f})"
        expr = f"if(lt(t,{t1:.3f}),{segment},{expr})"
    expr = f"if(lt(t,{points[0][0]:.3f}),{points[0][1]},{expr})"
    return f"clip({expr},0,{max_x})"


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
    """Trajectoire de crop lissée sous forme d'expression FFmpeg, rebasée à t=0."""
    y0 = (source_h - crop_h) // 2
    max_x = source_w - crop_w
    if max_x <= 0 or not any(value is not None for _t, value in track):
        return centred_crop_path(source_w, source_h, crop_w, crop_h)
    filled = _fill_gaps(track)

    end = t_end if t_end is not None else filled[-1][0]
    lo = max(t_start - 0.5, filled[0][0])
    hi = max(lo + 0.2, end + 0.5)
    smooth = _smooth(
        _resample(filled, RESAMPLE_HZ, lo, hi),
        max(1, int(SMOOTH_WINDOW_S * RESAMPLE_HZ / 2)),
    )
    pixels = [min(max(cx * source_w - crop_w / 2, 0.0), max_x) for cx in smooth]

    max_step = MAX_VEL_FRAC * source_w / RESAMPLE_HZ
    for i in range(1, len(pixels)):
        delta = max(-max_step, min(max_step, pixels[i] - pixels[i - 1]))
        pixels[i] = pixels[i - 1] + delta

    samples = [
        (round(lo + i / RESAMPLE_HZ - t_start, 3), int(round(x)))
        for i, x in enumerate(pixels)
        if lo + i / RESAMPLE_HZ - t_start >= -0.05
    ] or [(0.0, int(round(pixels[0])))]

    points = [(max(samples[0][0], 0.0), samples[0][1])]
    for t, x in samples[1:]:
        if abs(x - points[-1][1]) >= 2:
            points.append((t, x))
    if points[-1] != samples[-1]:
        points.append(samples[-1])
    if len(points) > 160:
        stride = len(points) // 160 + 1
        points = points[::stride] + [points[-1]]

    return CropPath(crop_w, crop_h, y0, _x_expression(points, max_x), points)


def centred_crop_path(source_w: int, source_h: int, crop_w: int, crop_h: int) -> CropPath:
    x0 = (source_w - crop_w) // 2
    y0 = (source_h - crop_h) // 2
    return CropPath(crop_w, crop_h, y0, str(x0), [(0.0, x0)])
