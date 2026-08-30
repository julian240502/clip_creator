from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.downloader import download_video
from src.paths import DATA_DIR
from src.resizer import resize_clip_for_vertical
from src.video_splitter import split_video

ProgressCallback = Callable[[float, str], None]


def _project_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    return slug[:60] or "video"


def process_video(
    *, url: str | None = None, uploaded_path: str | Path | None = None,
    clip_length: int = 30, vertical: bool = True, vertical_mode: str = "crop",
    progress: ProgressCallback | None = None,
) -> tuple[Path, list[Path]]:
    """Exécute le pipeline et renvoie le dossier projet et les exports finaux."""
    report = progress or (lambda _value, _message: None)
    if bool(url) == bool(uploaded_path):
        raise ValueError("Fournissez une URL ou un fichier, mais pas les deux.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    hint = Path(uploaded_path).stem if uploaded_path else "download"
    project_dir = Path(DATA_DIR) / "projects" / f"{stamp}-{_project_name(hint)}"
    source_dir = project_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=False)
    if url:
        report(0.08, "Téléchargement de la vidéo…")
        source = Path(download_video(url, source_dir))
    else:
        report(0.08, "Préparation du fichier…")
        incoming = Path(uploaded_path)
        if not incoming.is_file():
            raise FileNotFoundError(f"Fichier introuvable : {incoming}")
        source = source_dir / incoming.name
        shutil.copy2(incoming, source)
    report(0.25, "Découpage précis des clips…")
    landscape = [Path(item) for item in split_video(source, clip_length, project_dir / "clips")]
    if not vertical:
        report(1.0, "Exports terminés")
        return project_dir, landscape
    vertical_dir, exports = project_dir / "vertical", []
    for index, clip in enumerate(landscape, start=1):
        report(0.25 + 0.7 * ((index - 1) / max(len(landscape), 1)), f"Format vertical {index}/{len(landscape)}…")
        exports.append(resize_clip_for_vertical(clip, vertical_dir / clip.name, vertical_mode))
    report(1.0, "Exports terminés")
    return project_dir, exports
