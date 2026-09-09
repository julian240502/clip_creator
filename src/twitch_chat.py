"""Chat d'une rediff Twitch : les pics de messages = moments potentiellement viraux.

Le meilleur signal pour repérer un highlight sur un live : quand le chat
explose (surtout en emotes de rire), il s'est passé quelque chose. On télécharge
le chat du VOD (`chat-downloader`, optionnel), on repère les tranches où le débit
de messages dépasse nettement sa base locale — pondéré par la densité d'emotes
de rire — et on renvoie des instants (déjà corrigés du délai de réaction du chat).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

# Le chat d'un très gros stream (Kai Cenat & co) = centaines de milliers de
# messages, et `chat-downloader` pagine (rate-limit Twitch) — voire se bloque en
# boucle de retry. Ce signal est **désactivé par défaut** (CLIP_CREATOR_ENABLE_CHAT=1
# pour l'activer) et l'appelant l'exécute dans un thread borné. Ici on borne aussi :
# échantillon représentatif suffit. CLIP_CREATOR_CHAT_MAX_SECONDS pour aller plus loin.
_CHAT_MAX_SECONDS = float(os.environ.get("CLIP_CREATOR_CHAT_MAX_SECONDS", "75") or 75)
_CHAT_MAX_MESSAGES = int(os.environ.get("CLIP_CREATOR_CHAT_MAX_MESSAGES", "150000") or 150000)

# Emotes / expressions de rire les plus courantes sur Twitch (+ « clip it »).
_LAUGH_RE = re.compile(
    r"KEKW|OMEGALUL|LULW?|LMAOO*|LMFAO|PepeLaugh|Pepega|ICANT|KEKLEO|OMEGAROFL|"
    r"4Head|xD+|\bMDR+\b|\bPTDR+\b|😂|🤣|💀|\bclip(?:\s?it|ped|\sthat)\b|\+2\b",
    re.IGNORECASE,
)


def is_twitch_vod(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host.endswith("twitch.tv"):
        return False
    return "/videos/" in url or bool(re.search(r"twitch\.tv/videos/\d+", url))


def _cache_file(
    url: str, cache_dir: str | Path, start: float | None, end: float | None,
) -> Path:
    vid = re.search(r"/videos/(\d+)", url)
    name = vid.group(1) if vid else re.sub(r"\W+", "_", url)[-40:]
    span = f"_{int(start or 0)}-{int(end)}" if end is not None else ""
    return Path(cache_dir) / f"chat_{name}{span}.json"


def download_chat(
    url: str, cache_dir: str | Path, *, start: float | None = None, end: float | None = None,
) -> list[tuple[float, str]] | None:
    """`(offset_secondes_dans_le_VOD, message)`. `start`/`end` limitent le
    téléchargement à cette fenêtre (les temps restent absolus). None si indispo."""
    cache = _cache_file(url, cache_dir, start, end)
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            return [(float(t), str(m)) for t, m in data]
        except (OSError, ValueError):
            pass
    try:
        from chat_downloader import ChatDownloader
    except ImportError:
        return None
    deadline = time.monotonic() + _CHAT_MAX_SECONDS
    truncated = False
    out: list[tuple[float, str]] = []
    try:
        # `timeout` (inactivité) + `max_attempts` : `chat-downloader` doit finir
        # par rendre la main même si Twitch rate-limite (sinon boucle de retry
        # infinie). L'appelant l'exécute en plus dans un thread borné.
        chat = ChatDownloader().get_chat(
            url, message_types=["text_message"], start_time=start, end_time=end,
            max_messages=_CHAT_MAX_MESSAGES, timeout=20, max_attempts=2,
            retry_timeout=15,
        )
        for msg in chat:
            t = msg.get("time_in_seconds")
            text = msg.get("message") or ""
            if t is not None and text:
                out.append((float(t), str(text)))
            if time.monotonic() > deadline:
                truncated = True
                break
    except Exception:  # noqa: BLE001 - réseau / VOD sans chat / API changée
        if not out:
            return None
        truncated = True
    if not out:
        return None
    out.sort(key=lambda item: item[0])
    if not truncated:  # un échantillon partiel n'est pas remis en cache
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return out


def chat_spikes(
    messages: list[tuple[float, str]],
    *,
    bucket: float = 5.0,
    baseline_radius: int = 60,
    threshold: float = 2.4,
    lag: float = 3.5,
    min_gap: float = 20.0,
) -> list[tuple[float, float]]:
    """Instants où le chat s'emballe : `[(temps, intensité), ...]`, triés.

    `temps` est déjà reculé de `lag` (le chat réagit après coup). `intensité` ≈
    combien de fois la normale (≥ `threshold`), bonifiée si c'est une salve de
    rires.
    """
    if not messages:
        return []
    span_end = messages[-1][0]
    n_buckets = int(span_end // bucket) + 1
    counts = [0] * n_buckets
    laughs = [0] * n_buckets
    for t, text in messages:
        b = int(t // bucket)
        if 0 <= b < n_buckets:
            counts[b] += 1
            if _LAUGH_RE.search(text):
                laughs[b] += 1

    scores: list[float] = []
    for b in range(n_buckets):
        if counts[b] < 3:
            scores.append(0.0)
            continue
        lo = max(0, b - baseline_radius)
        hi = min(n_buckets, b + baseline_radius + 1)
        active = sorted(c for c in counts[lo:hi] if c > 0)
        if len(active) < 5:  # pas assez de contexte (bord de fenêtre) -> on ignore
            scores.append(0.0)
            continue
        base = active[len(active) // 2] or 1
        ratio = counts[b] / base
        laugh_frac = laughs[b] / counts[b]
        scores.append(ratio + 1.5 * laugh_frac)

    spikes: list[tuple[float, float]] = []
    for b in range(n_buckets):
        s = scores[b]
        if s < threshold:
            continue
        if b > 0 and scores[b - 1] > s:
            continue  # garder le sommet local
        if b + 1 < n_buckets and scores[b + 1] > s:
            continue
        t = max(0.0, b * bucket - lag)
        if spikes and t - spikes[-1][0] < min_gap:
            if s > spikes[-1][1]:
                spikes[-1] = (t, round(s, 2))
            continue
        spikes.append((t, round(s, 2)))
    return spikes
