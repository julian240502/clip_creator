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

Sur un mega-stream (xQc, Kai Cenat…) le chat tourne parfois à plusieurs dizaines
de messages/seconde. Marcher séquentiellement depuis le début d'une fenêtre de
plusieurs heures n'atteint alors jamais la fin dans le budget temps imparti : on
épuise tout sur les toutes premières minutes et le reste de la fenêtre n'est
jamais vu (0 pic détecté, pas parce qu'il n'y en a pas, mais parce qu'on n'a
jamais regardé). `find_chat_spikes` mesure le débit réel avant de choisir : la
marche séquentielle habituelle si elle peut couvrir toute la fenêtre dans le
budget, sinon des **sondes réparties sur toute la fenêtre** (voir
`_sample_chat_activity`).
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

# Choix marche séquentielle / sondes réparties (voir find_chat_spikes) : coût
# prudent d'une requête GQL, et part du budget qu'on autorise à la marche
# séquentielle avant de basculer en échantillonnage.
_REQUEST_COST_S = 0.5
_SEQUENTIAL_BUDGET_FRAC = 0.9

# Échantillonnage réparti (fenêtre trop longue / chat trop dense) : nombre de
# sondes (~1 toutes les 10 min, borné), pages consécutives lues par sonde, et
# seuil (ratio au débit médian des sondes) pour signaler un pic.
_PROBE_MIN = 10
_PROBE_MAX = 50
_PROBE_PAGES = 2
_PROBE_THRESHOLD = 2.2

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


def _fetch_page(
    vod_id: str, offset: float,
) -> tuple[list[tuple[float, str]], float, bool] | None:
    """Une page de commentaires à cet offset : `(messages, offset_du_dernier,
    hasNextPage)`. `None` si erreur API (PersistedQueryNotFound,
    IntegrityCheckFailed…) ou réseau."""
    body = [{
        "operationName": "VideoCommentsByOffsetOrCursor",
        "variables": {"videoID": vod_id, "contentOffsetSeconds": int(offset)},
        "extensions": {"persistedQuery": {"version": 1, "sha256Hash": _VOD_COMMENTS_HASH}},
    }]
    data = _gql(body)
    if not data or not isinstance(data, list) or data[0].get("errors"):
        return None
    comments = ((data[0].get("data") or {}).get("video") or {}).get("comments")
    if not comments or not comments.get("edges"):
        return [], offset, False

    edges = comments["edges"]
    msgs: list[tuple[float, str]] = []
    for edge in edges:
        node = edge.get("node") or {}
        t = float(node.get("contentOffsetSeconds", 0.0))
        text = "".join(
            frag.get("text", "") for frag in (node.get("message") or {}).get("fragments", [])
        ).strip()
        if text:
            msgs.append((t, text))
    last_offset = float(edges[-1]["node"]["contentOffsetSeconds"])
    return msgs, last_offset, bool(comments.get("pageInfo", {}).get("hasNextPage"))


def _fetch_vod_comments(
    vod_id: str, start: float, end: float, *, deadline: float | None = None,
) -> tuple[list[tuple[float, str]], bool]:
    """Commentaires du VOD sur `[start, end]` (offsets secondes), triés.

    Marche **séquentielle par offset** depuis `start` : on redemande la requête
    avec `contentOffsetSeconds` = offset du dernier message reçu (le champ
    `cursor` déclenche un `IntegrityCheckFailed`). Couverture continue et
    précise, mais consomme le budget temps dans l'ordre chronologique — sur un
    chat très dense et une longue fenêtre, ça n'atteint jamais la fin (voir
    `find_chat_spikes`, qui décide quand basculer sur des sondes réparties).
    Renvoie `(messages, tronqué)` ; `tronqué` = un cap ou une erreur API a
    coupé la collecte avant `end`.
    """
    deadline = deadline if deadline is not None else time.monotonic() + _CHAT_MAX_SECONDS
    seen: set[tuple[float, str]] = set()
    out: list[tuple[float, str]] = []
    offset = max(0.0, start)
    truncated = False
    stalls = 0

    while offset < end:
        if time.monotonic() > deadline or len(out) >= _CHAT_MAX_MESSAGES:
            truncated = True
            break
        page = _fetch_page(vod_id, offset)
        if page is None:
            # PersistedQueryNotFound / IntegrityCheckFailed / réseau : on garde
            # ce qu'on a (un échantillon partiel nourrit quand même la détection).
            truncated = truncated or bool(out)
            break
        msgs, last_offset, has_next = page
        for t, text in msgs:
            if start <= t <= end:
                key = (t, text)
                if key not in seen:
                    seen.add(key)
                    out.append(key)

        if not has_next or last_offset > end:
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


def _sample_chat_activity(
    vod_id: str, start: float, end: float, deadline: float,
) -> list[dict]:
    """Chat trop dense pour couvrir toute la fenêtre en marchant depuis `start` :
    des **sondes espacées régulièrement sur toute la fenêtre**, chacune
    `_PROBE_PAGES` pages consécutives, pour répartir le budget au lieu de le
    consommer sur les premières minutes. Renvoie `[{"t", "rate", "laugh_frac"}]`.
    """
    span = max(1.0, end - start)
    n_probes = int(min(_PROBE_MAX, max(_PROBE_MIN, span / 600)))  # ~1 sonde / 10 min
    stride = span / n_probes
    probes: list[dict] = []
    for i in range(n_probes):
        if time.monotonic() > deadline:
            break
        cursor = start + i * stride
        msgs: list[tuple[float, str]] = []
        for _ in range(_PROBE_PAGES):
            page = _fetch_page(vod_id, cursor)
            if page is None:
                break
            page_msgs, last_offset, has_next = page
            msgs.extend(page_msgs)
            if not has_next or last_offset <= cursor:
                break
            cursor = last_offset + 0.001
        if len(msgs) < 3:
            continue
        t0, t1 = msgs[0][0], msgs[-1][0]
        duration = max(1.0, t1 - t0)
        laughs = sum(1 for _, text in msgs if _LAUGH_RE.search(text))
        probes.append({
            "t": (t0 + t1) / 2, "rate": len(msgs) / duration, "laugh_frac": laughs / len(msgs),
        })
    return probes


def _spikes_from_probes(
    probes: list[dict], *, threshold: float = _PROBE_THRESHOLD, lag: float = 3.5,
) -> list[tuple[float, float]]:
    """Repère les sondes largement au-dessus du débit **médian de l'échantillon**
    (pas d'une fenêtre glissante locale : les sondes sont trop espacées pour ça).
    """
    if len(probes) < 5:
        return []
    rates = sorted(p["rate"] for p in probes)
    base = rates[len(rates) // 2] or 0.01
    spikes = [
        (max(0.0, p["t"] - lag), round(p["rate"] / base + 1.5 * p["laugh_frac"], 2))
        for p in probes
        if p["rate"] / base + 1.5 * p["laugh_frac"] >= threshold
    ]
    spikes.sort(key=lambda item: item[0])
    return spikes


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


def find_chat_spikes(
    url: str, cache_dir: str | Path, *, start: float | None = None, end: float | None = None,
) -> list[tuple[float, float]] | None:
    """Point d'entrée recommandé pour la sélection intelligente : fetch + pics
    en une fois, en choisissant automatiquement comment collecter.

    Une petite sonde initiale mesure le débit réel du chat. Si couvrir toute la
    fenêtre en marchant depuis `start` (comme `download_chat` + `chat_spikes`,
    couverture continue, meilleure résolution) tient dans le budget temps, on
    fait ça. Sinon — mega-streamer, fenêtre de plusieurs heures — marcher
    séquentiellement n'atteindrait jamais la fin : le budget est réparti en
    **sondes sur toute la fenêtre** (`_sample_chat_activity`) plutôt que
    consommé sur les premières minutes.

    `None` si l'URL n'est pas un VOD Twitch ou si l'API ne répond rien.
    """
    if not is_twitch_vod(url):
        return None
    vod_id = _vod_id(url)
    if not vod_id:
        return None
    lo = 0.0 if start is None else max(0.0, float(start))
    hi = float("inf") if end is None else float(end)

    cache = _cache_file(url, cache_dir, start, end)
    spike_cache = cache.with_name(f"spikes_{cache.name}")
    if spike_cache.is_file():
        try:
            data = json.loads(spike_cache.read_text(encoding="utf-8"))
            return [(float(t), float(i)) for t, i in data]
        except (OSError, ValueError):
            pass

    deadline = time.monotonic() + _CHAT_MAX_SECONDS
    probe_t0 = time.monotonic()
    probe = _fetch_page(vod_id, lo)
    if probe is None:
        return None
    probe_msgs, _probe_last, _probe_has_next = probe
    probe_cost = max(time.monotonic() - probe_t0, 0.05)

    span = None if hi == float("inf") else max(0.0, hi - lo)
    page_span = probe_msgs[-1][0] - probe_msgs[0][0] if len(probe_msgs) > 1 else 0.0

    sequential_ok = span is None or page_span <= 0.0
    if not sequential_ok:
        estimated_requests = span / max(page_span, 1.0)
        estimated_wall = estimated_requests * max(probe_cost, _REQUEST_COST_S)
        sequential_ok = estimated_wall <= _CHAT_MAX_SECONDS * _SEQUENTIAL_BUDGET_FRAC

    if sequential_ok:
        messages, _truncated = _fetch_vod_comments(vod_id, lo, hi, deadline=deadline)
        spikes = chat_spikes(messages) if messages else []
    else:
        probes = _sample_chat_activity(vod_id, lo, hi, deadline)
        spikes = _spikes_from_probes(probes) if len(probes) >= 5 else []

    try:
        spike_cache.parent.mkdir(parents=True, exist_ok=True)
        spike_cache.write_text(json.dumps(spikes, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return spikes


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
