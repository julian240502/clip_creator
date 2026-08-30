from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from yt_dlp import YoutubeDL

from src.paths import RAW_VIDEOS_DIR


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
        "outtmpl": str(destination / "%(title).120B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "overwrites": False,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError("Aucune information vidéo reçue.")
    return _newest_media(destination)


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
