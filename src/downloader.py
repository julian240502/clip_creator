from __future__ import annotations

import hashlib
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

from yt_dlp import YoutubeDL

from src.paths import RAW_VIDEOS_DIR
from src.twitch_chat import is_twitch_vod


def _client_opts() -> dict:
    """Options yt-dlp partagées : contourne le « Sign in to confirm you're not a bot ».

    YouTube exige de plus en plus une authentification. Renseigner l'une de ces
    variables d'environnement avant de lancer l'app :

      CLIP_CREATOR_YTDLP_COOKIES_BROWSER=chrome   (ou firefox, edge, brave… ;
                                                   « chrome:Profile 1 » pour un profil)
      CLIP_CREATOR_YTDLP_COOKIES_FILE=C:\\chemin\\cookies.txt
      CLIP_CREATOR_YTDLP_PLAYER_CLIENT=android,web   (optionnel, dépannage)
    """
    opts: dict = {}
    browser = os.environ.get("CLIP_CREATOR_YTDLP_COOKIES_BROWSER", "").strip()
    if browser:
        name, _, profile = browser.partition(":")
        opts["cookiesfrombrowser"] = (name.strip().lower(), profile.strip() or None, None, None)
    cookie_file = os.environ.get("CLIP_CREATOR_YTDLP_COOKIES_FILE", "").strip()
    if cookie_file:
        opts["cookiefile"] = cookie_file
    clients = os.environ.get("CLIP_CREATOR_YTDLP_PLAYER_CLIENT", "").strip()
    if clients:
        opts["extractor_args"] = {
            "youtube": {"player_client": [c.strip() for c in clients.split(",") if c.strip()]}
        }
    return opts


def _format_selector(max_height: int) -> str:
    if max_height <= 0:
        raise ValueError("La hauteur maximale doit être positive.")
    return f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b"


def _validate_url(url: str) -> str:
    value = url.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Saisissez une URL vidéo http(s) valide.")
    return value


def probe_url(video_url: str) -> dict:
    """Lit les métadonnées d'une URL sans télécharger la vidéo."""
    url = _validate_url(video_url)
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        **_client_opts(),
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise RuntimeError("Impossible de lire les informations de cette URL.")
    duration = info.get("duration")
    return {
        "title": info.get("title") or url,
        "duration": float(duration) if duration else None,
        "width": info.get("width"),
        "height": info.get("height"),
        "thumbnail": info.get("thumbnail"),
        "uploader": info.get("uploader"),
        "webpage_url": info.get("webpage_url") or url,
    }


def download_video(
    video_url: str,
    output_dir: str | Path | None = None,
    max_height: int = 1080,
) -> str:
    """Télécharge une vidéo et retourne avec certitude le fichier vidéo final."""
    url = _validate_url(video_url)
    destination = Path(output_dir or RAW_VIDEOS_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    options = {
        "format": _format_selector(max_height),
        "merge_output_format": "mp4",
        # Nom de fichier = id de la vidéo (ASCII, court, stable). Le titre peut
        # contenir des emoji / être très long : tronqué en octets il coupe un
        # caractère multi-octets et yt-dlp ne retrouve plus ses fragments
        # `.part-FragNNNN` sous Windows.
        "outtmpl": str(destination / "%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "windowsfilenames": True,
        "noplaylist": True,
        "overwrites": False,
        "quiet": True,
        "no_warnings": True,
        **_client_opts(),
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError("Aucune information vidéo reçue.")
    return _newest_media(destination)


def download_source(video_url: str, cache_root: str | Path, max_height: int = 1080) -> str:
    """Télécharge la vidéo, mise en cache par URL + qualité. Renvoie le fichier."""
    url = _validate_url(video_url)
    key = hashlib.sha1(f"{url}|{max_height}".encode()).hexdigest()[:16]
    bucket = Path(cache_root) / key
    bucket.mkdir(parents=True, exist_ok=True)
    cached = [
        path for path in bucket.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
    ]
    if cached:
        return str(max(cached, key=lambda path: path.stat().st_size).resolve())
    _clear_partial_downloads(bucket)
    return download_video(url, bucket, max_height=max_height)


def _clear_partial_downloads(bucket: Path) -> None:
    """Purge les restes d'un téléchargement interrompu (`.part`, `.part-FragNNNN`,
    `.ytdl`) pour repartir proprement."""
    for stale in bucket.iterdir():
        if stale.is_file() and (".part" in stale.name or stale.name.endswith(".ytdl")):
            stale.unlink(missing_ok=True)


# Une fenêtre dont la fin tombe pile sur (ou après) la toute fin réelle du VOD
# ne renvoie presque rien (un conteneur quasi vide, quelques centaines
# d'octets) : yt-dlp ne signale aucune erreur, et ffprobe lit le fichier sans
# erreur non plus mais sans `format.duration` -> "Durée vidéo illisible" bien
# plus tard, pour une cause qui n'a rien à voir avec ce message. Le duration
# annoncé par la plateforme peut être légèrement optimiste par rapport à ce
# qui est réellement récupérable en bout de rediff.
_MIN_VALID_BYTES = 100_000  # 100 Ko : bien sous un vrai segment, bien au-dessus d'un flux vide
_END_PULLBACK_S = 8.0
_END_PULLBACK_ATTEMPTS = 4


def _is_valid_media(path: Path) -> bool:
    """Taille plausible **et** conteneur lisible. Un fichier tronqué (process
    tué par un timeout en plein écriture, par ex.) peut largement dépasser le
    seuil de taille tout en étant structurellement invalide (`moov atom not
    found`) — un contrôle sur la seule taille le laisserait passer pour
    « valide », y compris en cache, et ferait planter tout ce qui l'utilise
    ensuite (`get_video_duration`…) bien plus tard et sans lien apparent."""
    try:
        if not path.is_file() or path.stat().st_size < _MIN_VALID_BYTES:
            return False
    except OSError:
        return False
    from src.video_splitter import get_video_duration

    try:
        return get_video_duration(path) > 0
    except Exception:  # noqa: BLE001 - conteneur illisible -> pas valide
        return False


def _range_bucket(cache_root: str | Path, url: str, max_height: int, start: float, end: float) -> Path:
    key = hashlib.sha1(f"{url}|{max_height}|{start:.1f}|{end:.1f}".encode()).hexdigest()[:16]
    bucket = Path(cache_root) / f"range_{key}"
    bucket.mkdir(parents=True, exist_ok=True)
    return bucket


def _cached_media(bucket: Path) -> Path | None:
    valid = [
        path for path in bucket.iterdir()
        if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"} and _is_valid_media(path)
    ]
    return max(valid, key=lambda path: path.stat().st_size).resolve() if valid else None


def _fetch_range_via_ffmpeg(
    url: str, bucket: Path, start: float, end: float, max_height: int,
) -> Path | None:
    """Un essai de `download_ranges` (ffmpeg, copie de flux). `None` (pas
    d'exception) si le résultat est quasi vide — laisse l'appelant décider de
    la suite plutôt que de faire remonter une erreur à chaque tentative."""
    stale = [
        path for path in bucket.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
    ]
    for path in stale:  # résidu quasi vide d'un essai précédent -> sinon yt-dlp
        path.unlink(missing_ok=True)  # ne l'écraserait pas (overwrites=False)
    _clear_partial_downloads(bucket)
    options = {
        "format": _format_selector(max_height),
        "merge_output_format": "mp4",
        "outtmpl": str(bucket / "%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "windowsfilenames": True,
        "noplaylist": True,
        "overwrites": False,
        "quiet": True,
        "no_warnings": True,
        "download_ranges": lambda _info, _ydl, _s=start, _e=end: [
            {"start_time": float(_s), "end_time": float(_e)}
        ],
        # PAS de force_keyframes_at_cuts : il force un ré-encodage CPU de toute la
        # section (silencieux, ~20-40 min pour 25 min de 1080p60). La copie de
        # flux suffit ici — la transcription et le rendu recadrent au besoin.
        **_client_opts(),
    }
    with YoutubeDL(options) as ydl:
        if not ydl.extract_info(url, download=True):
            raise RuntimeError("Aucune information vidéo reçue.")
    media = Path(_newest_media(bucket))
    return media if _is_valid_media(media) else None


# Découpe HLS « au segment » ---------------------------------------------
#
# Sur certains VOD Twitch (encodage fMP4/CMAF récent — repérable au tag
# #EXT-X-MAP dans la playlist, indépendant du fait que le VOD soit « muted »
# ou pas), le `-ss` distant de ffmpeg dans `_fetch_range_via_ffmpeg` échoue
# pour TOUT instant non nul : il lit ~1 Go de données mais démuxe 0 paquet.
# Seul `start = 0` fonctionne. Contourner ça en téléchargeant `[0, end]` marche
# mais coûte des heures de flux inutiles sur une longue rediff.
#
# Le vrai contournement : construire une playlist HLS **locale** qui ne liste
# QUE les segments (+ le segment d'init fMP4 s'il y en a un) qui couvrent la
# fenêtre demandée, et pointer ffmpeg dessus — aucun seek n'est alors
# nécessaire (la liste est déjà la bonne), donc le bug ne s'applique jamais,
# et seuls les segments voulus transitent sur le réseau (vérifié : ~13 Mo
# pour 80 s sur une rediff de 3h48, quel que soit l'endroit visé).
_EXTINF_RE = re.compile(r"#EXTINF:([0-9.]+)")
_MAP_RE = re.compile(r'#EXT-X-MAP:URI="([^"]+)"')
_SEGMENT_START_PAD = 12.0  # marge avant `start` : ~1 segment Twitch (10 s) de sécurité
# Sans #EXT-X-MAP (segments .ts classiques), le seek direct (download_ranges)
# marche nativement et est plus rapide qu'assembler nous-mêmes des dizaines de
# segments — au-delà de cette taille de fenêtre, autant lui laisser la main
# plutôt que risquer le timeout de la concaténation pour rien.
_MAX_SEGMENTS_WITHOUT_MAP = 60  # ~10 min à 10 s/segment
_SEGMENT_FETCH_TIMEOUT = 600.0  # généreux : seul recours qui marche sur un manifeste fMP4


def _fetch_text(url: str, timeout: float = 20.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _media_playlist_url(url: str, max_height: int) -> str:
    """URL de la playlist HLS (segments) pour la meilleure qualité <= max_height."""
    with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, **_client_opts()}) as ydl:
        info = ydl.extract_info(url, download=False)
    formats = [f for f in (info or {}).get("formats") or [] if f.get("url")]
    candidates = [f for f in formats if f.get("height") and f["height"] <= max_height]
    if not candidates:
        candidates = formats
    if not candidates:
        raise RuntimeError("Aucun flux vidéo disponible.")
    return max(candidates, key=lambda f: f.get("height") or 0)["url"]


def _parse_hls_segments(text: str, base_url: str) -> tuple[str | None, list[tuple[float, str]]]:
    """`(uri_segment_init_ou_None, [(durée, uri_absolue), ...])`."""
    map_match = _MAP_RE.search(text)
    map_uri = urljoin(base_url, map_match.group(1)) if map_match else None
    segments: list[tuple[float, str]] = []
    pending_duration: float | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("#EXTINF:"):
            match = _EXTINF_RE.match(line)
            pending_duration = float(match.group(1)) if match else 0.0
        elif line and not line.startswith("#"):
            segments.append((pending_duration or 0.0, urljoin(base_url, line)))
            pending_duration = None
    return map_uri, segments


def _select_segments(
    segments: list[tuple[float, str]], start: float, end: float, pad: float,
) -> tuple[list[tuple[float, str]], float]:
    """`([(durée, uri), ...], début_absolu_du_1er_segment_choisi)` couvrant
    `[start - pad, end]` — marge seulement au début (pour le seek arrière
    façon image-clé) ; à la fin on inclut juste le segment qui contient `end`,
    sans marge en plus (même profil d'imprécision que le `-ss/-t` direct, sur
    lequel l'appelant se cale déjà pour recaler l'offset)."""
    lo = max(0.0, start - pad)
    chosen: list[tuple[float, str]] = []
    first_start = 0.0
    cursor = 0.0
    for duration, uri in segments:
        seg_start, seg_end = cursor, cursor + duration
        if seg_end >= lo and seg_start <= end:
            if not chosen:
                first_start = seg_start
            chosen.append((duration, uri))
        cursor = seg_end
        if seg_start > end:
            break
    return chosen, first_start


def _fetch_range_via_segments(
    url: str, bucket: Path, start: float, end: float, max_height: int,
    output_name: str = "v.mp4",
) -> tuple[Path, float] | None:
    """Playlist locale ne listant que les segments couvrant `[start, end]`,
    lue par ffmpeg sans seek. `None` (pas d'exception) si la playlist n'a pas
    pu être construite ou lue — l'appelant retombe alors sur une autre méthode.
    Renvoie `(fichier, début_absolu_du_1er_segment_choisi)` — le fichier
    démarre à cet instant, pas nécessairement `start` (granularité des
    segments) ; l'appelant recale s'il a besoin d'un début exact."""
    try:
        playlist_url = _media_playlist_url(url, max_height)
        text = _fetch_text(playlist_url)
    except Exception:  # noqa: BLE001 - repli sur une autre méthode
        return None
    base = playlist_url.rsplit("/", 1)[0] + "/"
    map_uri, segments = _parse_hls_segments(text, base)
    chosen, first_start = _select_segments(segments, start, end, _SEGMENT_START_PAD)
    if not chosen:
        return None
    if map_uri is None and len(chosen) > _MAX_SEGMENTS_WITHOUT_MAP:
        return None  # pas de bug de seek à contourner ici -> autant laisser la main au direct

    lines = ["#EXTM3U", "#EXT-X-VERSION:7", f"#EXT-X-TARGETDURATION:{int(max(d for d, _ in chosen)) + 1}"]
    if map_uri:
        lines.append(f'#EXT-X-MAP:URI="{map_uri}"')
    for duration, uri in chosen:
        lines.extend([f"#EXTINF:{duration:.3f},", uri])
    lines.append("#EXT-X-ENDLIST")

    local_playlist = bucket / "local.m3u8"
    local_playlist.write_text("\n".join(lines), encoding="utf-8")
    output = bucket / output_name
    try:
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    # Nécessaire pour qu'une playlist LOCALE puisse pointer vers des
                    # segments distants (http/https) : ffmpeg les refuse sinon.
                    "-protocol_whitelist", "file,http,https,tcp,tls,crypto,data",
                    "-i", str(local_playlist), "-c", "copy", str(output),
                ],
                capture_output=True, text=True, timeout=_SEGMENT_FETCH_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            # Le process tué en plein écriture laisse un fichier tronqué (taille
            # plausible, conteneur invalide) — sans ce nettoyage, un appel
            # suivant sur la même fenêtre le prendrait pour un résultat en
            # cache valide (`_is_valid_media` avant son propre durcissement ne
            # regardait que la taille) et plantait plus tard, loin de la cause.
            output.unlink(missing_ok=True)
            raise
    finally:
        local_playlist.unlink(missing_ok=True)
    if result.returncode != 0 or not _is_valid_media(output):
        output.unlink(missing_ok=True)  # sinon un résidu invalide gênerait le repli suivant
        return None
    return output, first_start


def download_source_range(
    video_url: str, cache_root: str | Path, start: float, end: float, max_height: int = 1080,
) -> str:
    """Télécharge **seulement** `[start, end]` de la source (VOD de plusieurs heures
    dont on n'analyse qu'un extrait) et met en cache par URL + qualité + fenêtre.

    Le fichier renvoyé démarre à ~0 : l'appelant décale ses horodatages de `start`.
    Coupe en **copie de flux** (rapide) : le début réel peut reculer jusqu'à
    l'image-clé précédente (quelques secondes de marge), pas de ré-encodage.

    Essaie d'abord une **playlist HLS locale** ne listant que les segments
    voulus (`_fetch_range_via_segments`) : ne transfère jamais plus que la
    fenêtre demandée (± une poignée de secondes), quel que soit l'endroit visé
    dans une rediff de plusieurs heures — et contourne au passage un bug de
    seek ffmpeg sur les VOD encodés en fMP4 (voir plus haut).

    Si cette playlist ne peut pas être construite ou lue (manifeste
    inhabituel), replis successifs :

    1. `download_ranges` (ffmpeg, `-ss`/`-t` distant), avec retrait de `end`
       par `_END_PULLBACK_S` si elle déborde de la toute fin réelle du flux ;
    2. téléchargement de `[0, end]` puis découpe locale — fonctionne toujours,
       mais coûteux si `start` est loin dans une longue rediff.
    """
    url = _validate_url(video_url)
    if end <= start:
        raise ValueError("La fin de la fenêtre doit être après le début.")

    bucket = _range_bucket(cache_root, url, max_height, start, end)
    cached = _cached_media(bucket)
    if cached is not None:
        return str(cached)
    fetched = None
    if is_twitch_vod(url):
        # La playlist locale n'a d'intérêt que pour le bug de seek fMP4
        # spécifique aux VOD Twitch (voir plus haut) : sur les autres sources
        # (YouTube...), `_media_playlist_url` choisit un format par hauteur
        # sans vérifier la présence d'une piste audio, alors que YouTube sert
        # souvent le flux vidéo et le flux audio séparément (DASH) — un aperçu
        # sans son en résulterait. `_fetch_range_via_ffmpeg`/`download_ranges`
        # gèrent déjà correctement l'appariement audio+vidéo pour ces sources.
        try:
            fetched = _fetch_range_via_segments(url, bucket, start, end, max_height, output_name="raw.mp4")
        except Exception:  # noqa: BLE001 - repli sur download_ranges ci-dessous
            fetched = None
    if fetched is not None:
        raw_media, raw_start = fetched
        # La playlist locale rend des segments ENTIERS (jusqu'à ~1 de plus à
        # chaque bord) : sans découpe, `get_video_duration` peut dépasser la
        # fenêtre demandée de bien plus que les quelques secondes que les
        # appelants tolèrent (leur calcul de recalage `lead`/`media_t0`,
        # pensé pour l'imprécision "image-clé" du seek direct, ignore alors la
        # correction — décalage audio/sous-titres dans le clip final). On
        # découpe donc localement à `[start, end]`, comme le fait déjà
        # `download_clip` pour les aperçus.
        trimmed = bucket / "v.mp4"
        ok = _trim_local(raw_media, start - raw_start, end - start, trimmed)
        raw_media.unlink(missing_ok=True)
        if ok:
            return str(trimmed)

    last_error: Exception | None = None
    for attempt in range(_END_PULLBACK_ATTEMPTS):
        window_end = end - attempt * _END_PULLBACK_S
        if window_end <= start:
            break
        bucket = _range_bucket(cache_root, url, max_height, start, window_end)
        cached = _cached_media(bucket)
        if cached is not None:
            return str(cached)
        try:
            media = _fetch_range_via_ffmpeg(url, bucket, start, window_end, max_height)
        except Exception as exc:  # noqa: BLE001 - on retente avec une fenêtre plus courte
            last_error = exc
            continue
        if media is not None:
            return str(media)
        last_error = RuntimeError(f"téléchargement quasi vide pour [{start:.0f}, {window_end:.0f}]")

    if start > 0:
        try:
            return _download_range_from_start_and_trim(url, cache_root, start, end, max_height)
        except Exception as exc:  # noqa: BLE001 - message final unique plus bas
            last_error = exc

    raise RuntimeError(
        "Impossible de télécharger cette fenêtre — sa fin dépasse probablement "
        "la fin réelle du flux disponible, ou le manifeste distant refuse tout "
        "accès direct à un instant non nul et la playlist locale n'a pas pu "
        "être construite non plus."
    ) from last_error


def _trim_local(source: Path, offset: float, duration: float, output: Path) -> bool:
    """Découpe `[offset, offset+duration]` d'un fichier LOCAL avec ffmpeg (copie
    de flux) — aucune limite de seek ici, contrairement à un flux distant."""
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(max(0.0, offset)), "-i", str(source), "-t", str(duration),
            "-c", "copy", str(output),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not _is_valid_media(output):
        output.unlink(missing_ok=True)
        return False
    return True


def _download_range_from_start_and_trim(
    url: str, cache_root: str | Path, start: float, end: float, max_height: int,
) -> str:
    """Dernier repli (voir `download_source_range`) quand ni la playlist locale
    ni `download_ranges` distant n'ont marché : télécharge `[0, end]` — `0`
    fonctionne toujours, même sur les flux qui refusent tout `start` non nul —
    puis découpe `[start, end]` localement avec ffmpeg (copie de flux, aucune
    limite de seek sur un fichier local)."""
    full_bucket = _range_bucket(cache_root, url, max_height, 0.0, end)
    full_media = _cached_media(full_bucket)
    if full_media is None:
        full_media = _fetch_range_via_ffmpeg(url, full_bucket, 0.0, end, max_height)
        if full_media is None:
            raise RuntimeError(f"téléchargement quasi vide pour [0, {end:.0f}]")

    trimmed_bucket = _range_bucket(cache_root, url, max_height, start, end)
    cached_trim = _cached_media(trimmed_bucket)
    if cached_trim is not None:
        return str(cached_trim)
    output = trimmed_bucket / f"{full_media.stem}_trim{full_media.suffix}"
    if not _trim_local(full_media, start, end - start, output):
        raise RuntimeError("Découpage local impossible.")
    return str(output)


def download_clip(
    video_url: str,
    output_dir: str | Path,
    start: float,
    end: float,
    max_height: int = 480,
) -> str:
    """Télécharge seulement l'intervalle [start, end] en basse résolution (pour l'aperçu)."""
    url = _validate_url(video_url)
    if end <= start:
        raise ValueError("La fin de l'extrait doit être après le début.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("preview_source*"):
        stale.unlink()

    # Même bug de seek fMP4 que download_source_range (voir plus haut) : sur un
    # VOD concerné, un aperçu loin dans la vidéo revenait vide -> plus aucune
    # piste audio pour Whisper ("Output file does not contain any stream").
    # Même parade : playlist HLS locale (juste les segments utiles) puis
    # découpe locale précise (l'aperçu a besoin d'un extrait exact, pas
    # juste "à la granularité du segment près").
    fetched = None
    if is_twitch_vod(url):
        # Idem download_source_range : ce contournement est spécifique au bug
        # de seek fMP4 des VOD Twitch. Sur YouTube, `_media_playlist_url`
        # choisirait un format sans piste audio (flux DASH séparés) et
        # produirait un aperçu muet -> échec de `_extract_audio` ("Output
        # file does not contain any stream"). Le repli `download_ranges`
        # apparie déjà correctement audio+vidéo via `_format_selector`.
        try:
            fetched = _fetch_range_via_segments(
                url, destination, start, end, max_height, output_name="preview_source_raw.mp4",
            )
        except Exception:  # noqa: BLE001 - repli sur download_ranges ci-dessous
            fetched = None
    if fetched is not None:
        raw_media, raw_start = fetched
        trimmed = destination / "preview_source.mp4"
        ok = _trim_local(raw_media, start - raw_start, end - start, trimmed)
        raw_media.unlink(missing_ok=True)
        if ok:
            return str(trimmed)

    options = {
        "format": _format_selector(max_height),
        "merge_output_format": "mp4",
        "outtmpl": str(destination / "preview_source.%(ext)s"),
        "noplaylist": True,
        "overwrites": True,
        "quiet": True,
        "no_warnings": True,
        "download_ranges": lambda _info, _ydl: [{"start_time": float(start), "end_time": float(end)}],
        "force_keyframes_at_cuts": True,
        **_client_opts(),
    }
    with YoutubeDL(options) as ydl:
        if not ydl.extract_info(url, download=True):
            raise RuntimeError("Aucune information vidéo reçue.")
    matches = sorted(
        (
            path for path in destination.glob("preview_source.*")
            if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
        ),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not matches:
        raise RuntimeError("Extrait téléchargé introuvable.")
    return str(matches[0].resolve())


def _newest_media(destination: Path) -> str:
    candidates = [
        path for path in destination.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
    ]
    if not candidates:
        raise RuntimeError("Téléchargement terminé, mais aucun fichier vidéo n'a été trouvé.")
    # Le MP4 fusionné est prioritaire, puis le plus gros média disponible.
    candidates.sort(key=lambda path: (path.suffix.lower() == ".mp4", path.stat().st_size), reverse=True)
    return str(candidates[0].resolve())
