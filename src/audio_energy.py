"""Enveloppe de volume d'une piste audio — signal « il se passe un truc ».

Sert à la sélection intelligente : rires, cris, hype se traduisent par des pics
de volume et des transitions silence→explosion. On ne cherche pas à *nommer*
l'événement (rire vs cri), juste à repérer l'intensité. ~0 dépendance : ffmpeg
pour décoder + numpy pour le RMS (déjà présent avec faster-whisper). Si l'un
manque, les fonctions renvoient un signal neutre et la notation reste
purement textuelle.
"""

from __future__ import annotations

import subprocess

_SR = 8000  # sous-échantillonnage : une enveloppe n'a pas besoin de la HD


def loudness_curve(media_path: str, *, hop: float = 0.5) -> tuple[list[float], float] | None:
    """RMS par tranche de `hop` secondes, normalisé 0..1. `(valeurs, hop)` ou None."""
    try:
        import numpy as np
    except ImportError:
        return None
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(media_path), "-vn", "-ac", "1", "-ar", str(_SR),
        "-f", "s16le", "-",
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True, timeout=600).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if not raw:
        return None
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    step = max(1, int(_SR * hop))
    n = len(samples) // step
    if n == 0:
        return None
    frames = samples[: n * step].reshape(n, step)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-9)
    # échelle perceptuelle (dB) ramenée dans [0, 1]
    db = 20.0 * np.log10(np.maximum(rms, 1e-5))
    lo, hi = np.percentile(db, 5), np.percentile(db, 95)
    if hi - lo < 1e-3:
        return [0.0] * n, hop
    curve = np.clip((db - lo) / (hi - lo), 0.0, 1.0)
    return [float(x) for x in curve], hop


def excitement(values: list[float], hop: float, start: float, end: float) -> float:
    """Intensité 0..1 d'une fenêtre `[start, end]` (secondes, base de la courbe).

    Combine : niveau max vs médiane globale, proportion de tranches « fortes »,
    et la plus grosse montée soudaine (silence -> explosion) dans la fenêtre.
    """
    if not values or end <= start:
        return 0.0
    i0 = max(0, int(start / hop))
    i1 = min(len(values), int(end / hop) + 1)
    seg = values[i0:i1]
    if not seg:
        return 0.0
    ref = sorted(values)[len(values) // 2]  # médiane globale
    peak = max(seg)
    loud_frac = sum(1 for v in seg if v >= max(0.55, ref + 0.2)) / len(seg)
    jump = max((b - a for a, b in zip(seg, seg[1:], strict=False)), default=0.0)
    raw = 0.5 * max(0.0, peak - ref) + 0.3 * loud_frac + 0.6 * max(0.0, jump - 0.15)
    return max(0.0, min(1.0, raw))
