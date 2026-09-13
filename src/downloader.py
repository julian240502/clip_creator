from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from yt_dlp import YoutubeDL

from src.paths import RAW_VIDEOS_DIR


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
    try:
        return path.is_file() and path.stat().st_size >= _MIN_VALID_BYTES
    except OSError:
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


def download_source_range(
    video_url: str, cache_root: str | Path, start: float, end: float, max_height: int = 1080,
) -> str:
    """Télécharge **seulement** `[start, end]` de la source (VOD de plusieurs heures
    dont on n'analyse qu'un extrait) et met en cache par URL + qualité + fenêtre.

    Le fichier renvoyé démarre à ~0 : l'appelant décale ses horodatages de `start`.
    Coupe en **copie de flux** (rapide) : le début réel peut reculer jusqu'à
    l'image-clé précédente (quelques secondes de marge), pas de ré-encodage.

    Deux causes distinctes peuvent rendre `[start, end]` inaccessible en direct :

    1. `end` déborde de la toute fin réelle du flux (le duration annoncé par
       la plateforme peut être légèrement optimiste) -> on retente en reculant
       `end` de `_END_PULLBACK_S` par cran, jusqu'à `_END_PULLBACK_ATTEMPTS` fois.
    2. Le VOD tourne sur une playlist Twitch **« muted »** (musique sous droits
       coupée sur tout ou partie de la rediff) : ffmpeg n'arrive alors à sauter
       (`-ss`) qu'à `start = 0`, jamais à un instant non nul — quel que soit cet
       instant. Repli : télécharger `[0, end]` (ça marche toujours) puis
       découper `[start, end]` **localement** (le seek sur un fichier local
       n'a pas cette limite). Plus lent — voire coûteux si `start` est loin
       dans une longue rediff — mais ça marche là où le direct échoue net.
    """
    url = _validate_url(video_url)
    if end <= start:
        raise ValueError("La fin de la fenêtre doit être après le début.")

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
        "la fin réelle du flux disponible, ou ce VOD (musique sous droits "
        "\"muted\") empêche tout accès direct à un instant non nul."
    ) from last_error


def _download_range_from_start_and_trim(
    url: str, cache_root: str | Path, start: float, end: float, max_height: int,
) -> str:
    """Repli pour les VOD « muted » (voir `download_source_range`) : télécharge
    `[0, end]` — un `start` non nul est ce qui échoue, `0` fonctionne toujours
    sur ces flux — puis découpe `[start, end]` localement avec ffmpeg (copie de
    flux, aucune limite de seek sur un fichier local)."""
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
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(max(0.0, start)), "-i", str(full_media), "-t", str(end - start),
            "-c", "copy", str(output),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not _is_valid_media(output):
        raise RuntimeError(result.stderr.strip() or "Découpage local impossible.")
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
    for stale in destination.glob("preview_source.*"):
        stale.unlink()
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
