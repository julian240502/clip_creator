from __future__ import annotations

import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.downloader import download_video
from src.encoder import encoder_label, resolve_video_encoder
from src.paths import DATA_DIR
from src.quality import get_quality_preset
from src.resizer import resize_clip_for_vertical
from src.video_splitter import get_video_duration, split_video

ProgressCallback = Callable[[float, str], None]


def _project_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    return slug[:60] or "video"


def process_video(
    *, url: str | None = None, uploaded_path: str | Path | None = None,
    clip_length: int = 30, vertical: bool = True, vertical_mode: str = "crop",
    encoder: str = "auto", export_quality: str = "1080p",
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
    quality = get_quality_preset(export_quality)
    if url:
        report(0.08, f"Téléchargement de la vidéo · maximum {quality.label}…")
        source = Path(download_video(url, source_dir, max_height=quality.source_max_height))
    else:
        report(0.08, "Préparation du fichier…")
        incoming = Path(uploaded_path)
        if not incoming.is_file():
            raise FileNotFoundError(f"Fichier introuvable : {incoming}")
        source = source_dir / incoming.name
        shutil.copy2(incoming, source)
    resolved_encoder = resolve_video_encoder(encoder)
    if not vertical:
        report(0.25, f"Découpage via {encoder_label(resolved_encoder)}…")
        landscape = [
            Path(item) for item in split_video(
                source, clip_length, project_dir / "clips", encoder=encoder
            )
        ]
        report(1.0, "Exports terminés")
        return project_dir, landscape
    # Découpage et conversion verticale sont réalisés ensemble : chaque image
    # n'est encodée qu'une fois au lieu de subir deux passes successives.
    duration = get_video_duration(source)
    clip_count = math.ceil(duration / clip_length)
    vertical_dir, exports = project_dir / "vertical", []
    report(0.25, f"Découpage + format vertical via {encoder_label(resolved_encoder)}…")
    for offset in range(clip_count):
        start = offset * clip_length
        length = min(clip_length, duration - start)
        report(
            0.25 + 0.7 * (offset / max(clip_count, 1)),
            f"Export {offset + 1}/{clip_count} · {encoder_label(resolved_encoder)}…",
        )
        output = vertical_dir / f"clip_{offset + 1:03d}.mp4"
        exports.append(
            resize_clip_for_vertical(
                source, output, vertical_mode, encoder=encoder,
                quality=quality.key, start=start, duration=length,
            )
        )
    report(1.0, "Exports terminés")
    return project_dir, exports
