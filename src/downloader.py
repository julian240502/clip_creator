from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from yt_dlp import YoutubeDL

from src.paths import RAW_VIDEOS_DIR


def _validate_url(url: str) -> str:
    value = url.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Saisissez une URL vidéo http(s) valide.")
    return value


def download_video(video_url: str, output_dir: str | Path | None = None) -> str:
    """Télécharge une vidéo et retourne avec certitude le fichier vidéo final."""
    url = _validate_url(video_url)
    destination = Path(output_dir or RAW_VIDEOS_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bv*+ba/b",
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
    candidates = [
        path for path in destination.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
    ]
    if candidates:
        # Le MP4 fusionné est prioritaire, puis le plus gros média disponible.
        candidates.sort(key=lambda path: (path.suffix.lower() == ".mp4", path.stat().st_size), reverse=True)
        return str(candidates[0].resolve())
    raise RuntimeError("Téléchargement terminé, mais aucun fichier vidéo n'a été trouvé.")
