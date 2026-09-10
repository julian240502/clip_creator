"""Chat d'une rediff Twitch : les pics de messages = moments potentiellement viraux.

Le meilleur signal pour repérer un highlight sur un live : quand le chat
explose (surtout en emotes de rire), il s'est passé quelque chose. On télécharge
les commentaires du VOD via l'API GraphQL web de Twitch (aucune dépendance, pas
d'authentification), on repère les tranches où le débit de messages dépasse
nettement sa base locale — pondéré par la densité d'emotes de rire — et on
renvoie des instants (déjà corrigés du délai de réaction du chat).

Historique : on passait par `chat-downloader`, aujourd'hui cassé
(`PersistedQueryNotFound`) et non maintenu. La pagination **par cursor** de
l'API Twitch déclenche désormais un `IntegrityCheckFailed` (protection anti-bot).
La pagination **par offset** (`contentOffsetSeconds`) passe sans jeton : c'est
celle qu'on utilise ici.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

# Client-ID web public de Twitch (le même pour tous les navigateurs, non secret).
_TWITCH_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
_GQL_ENDPOINT = "https://gql.twitch.tv/gql"
# Hash de la requête persistée « VideoCommentsByOffsetOrCursor » de l'API web.
_VOD_COMMENTS_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"
_GQL_TIMEOUT = 20.0

# Le chat d'un très gros stream (Kai Cenat & co) = centaines de milliers de
# messages : chaque page GQL rend ~60-200 messages, donc une heure de chat dense
# = beaucoup de requêtes. On borne : un échantillon représentatif suffit à la
# détection de pics. CLIP_CREATOR_CHAT_MAX_SECONDS / _MESSAGES pour élargir.
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


def _vod_id(url: str) -> str | None:
    match = re.search(r"/videos/(\d+)", url)
    return match.group(1) if match else None


def _cache_file(
    url: str, cache_dir: str | Path, start: float | None, end: float | None,
) -> Path:
    vid = re.search(r"/videos/(\d+)", url)
    name = vid.group(1) if vid else re.sub(r"\W+", "_", url)[-40:]
    span = f"_{int(start or 0)}-{int(end)}" if end is not None else ""
    return Path(cache_dir) / f"chat_{name}{span}.json"


def _gql(body: list, *, timeout: float = _GQL_TIMEOUT) -> list | None:
    """POST une requête GraphQL Twitch. `None` si réseau / réponse illisible."""
    request = urllib.request.Request(
        _GQL_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Client-ID": _TWITCH_CLIENT_ID, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _fetch_vod_comments(
    vod_id: str, start: float, end: float,
) -> tuple[list[tuple[float, str]], bool]:
    """Commentaires du VOD sur `[start, end]` (offsets secondes), triés.

    Pagination **par offset uniquement** : on redemande la requête avec
    `contentOffsetSeconds` = offset du dernier message reçu (le champ `cursor`
    déclenche un `IntegrityCheckFailed`). Renvoie `(messages, tronqué)` ;
    `tronqué` = un cap (temps / nombre) ou une erreur API a coupé la collecte.
    """
    deadline = time.monotonic() + _CHAT_MAX_SECONDS
    seen: set[tuple[float, str]] = set()
    out: list[tuple[float, str]] = []
    offset = max(0.0, start)
    truncated = False
    stalls = 0

    while offset < end:
        if time.monotonic() > deadline or len(out) >= _CHAT_MAX_MESSAGES:
            truncated = True
            break
        body = [{
            "operationName": "VideoCommentsByOffsetOrCursor",
            "variables": {"videoID": vod_id, "contentOffsetSeconds": int(offset)},
            "extensions": {
                "persistedQuery": {"version": 1, "sha256Hash": _VOD_COMMENTS_HASH},
            },
        }]
        data = _gql(body)
        if not data or not isinstance(data, list) or data[0].get("errors"):
            # PersistedQueryNotFound / IntegrityCheckFailed / réseau : on garde
            # ce qu'on a (un échantillon partiel nourrit quand même la détection).
            truncated = truncated or bool(out)
            break
        comments = ((data[0].get("data") or {}).get("video") or {}).get("comments")
        if not comments or not comments.get("edges"):
            break

        edges = comments["edges"]
        last_offset = float(edges[-1]["node"]["contentOffsetSeconds"])
        for edge in edges:
            node = edge.get("node") or {}
            t = float(node.get("contentOffsetSeconds", 0.0))
            if t < start or t > end:
                continue
            text = "".join(
                frag.get("text", "")
                for frag in (node.get("message") or {}).get("fragments", [])
            ).strip()
            key = (t, text)
            if text and key not in seen:
                seen.add(key)
                out.append(key)

        if not comments.get("pageInfo", {}).get("hasNextPage") or last_offset > end:
            break
        if last_offset <= offset:  # la page n'a pas fait avancer l'offset
            stalls += 1
            if stalls > 20:
                break
            offset += 5.0
        else:
            stalls = 0
            offset = last_offset + 0.001

    out.sort(key=lambda item: item[0])
    return out, truncated


def download_chat(
    url: str, cache_dir: str | Path, *, start: float | None = None, end: float | None = None,
) -> list[tuple[float, str]] | None:
    """`(offset_secondes_dans_le_VOD, message)`. `start`/`end` limitent la
    collecte à cette fenêtre (les temps restent absolus). `None` si l'URL n'est
    pas un VOD Twitch ou si l'API ne renvoie rien."""
    if not is_twitch_vod(url):
        return None
    vod_id = _vod_id(url)
    if not vod_id:
        return None

    cache = _cache_file(url, cache_dir, start, end)
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            return [(float(t), str(m)) for t, m in data]
        except (OSError, ValueError):
            pass

    lo = 0.0 if start is None else max(0.0, float(start))
    hi = float("inf") if end is None else float(end)
    messages, truncated = _fetch_vod_comments(vod_id, lo, hi)
    if not messages:
        return None
    if not truncated:  # un échantillon partiel n'est pas remis en cache
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return messages


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
