from __future__ import annotations

import math
import re
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.downloader import download_source
from src.encoder import encoder_label, resolve_video_encoder
from src.paths import DATA_DIR, SOURCE_CACHE_DIR
from src.quality import get_quality_preset
from src.resizer import segment_vertical
from src.transcribe import DEFAULT_MODEL
from src.video_splitter import (
    get_video_duration,
    resolve_source_window,
    split_video,
)

if TYPE_CHECKING:
    from src.captions import CaptionStyle

ProgressCallback = Callable[[float, str], None]
ClipCallback = Callable[[Path], None]


def _project_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    return slug[:60] or "video"


def process_video(
    *, url: str | None = None, uploaded_path: str | Path | None = None,
    clip_length: int = 30, vertical: bool = True,
    encoder: str = "auto", export_quality: str = "1080p",
    encoding_speed: str = "balanced",
    vertical_background: str = "blur",
    source_start: float | None = None,
    source_end: float | None = None,
    transcribe: bool = False,
    transcribe_model: str = DEFAULT_MODEL,
    captions_style: CaptionStyle | None = None,
    on_clip: ClipCallback | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Path, list[Path]]:
    """Exécute le pipeline et renvoie le dossier projet et les exports finaux."""
    report = progress or (lambda _value, _message: None)
    notify_clip = on_clip or (lambda _path: None)
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
        source = Path(
            download_source(url, SOURCE_CACHE_DIR, max_height=quality.source_max_height)
        )
    else:
        report(0.08, "Préparation du fichier…")
        incoming = Path(uploaded_path)
        if not incoming.is_file():
            raise FileNotFoundError(f"Fichier introuvable : {incoming}")
        source = source_dir / incoming.name
        shutil.copy2(incoming, source)
    resolved_encoder = resolve_video_encoder(encoder)
    window_start, window_end = resolve_source_window(
        get_video_duration(source), source_start, source_end
    )
    transcript = None
    if transcribe or captions_style is not None:
        from src.transcribe import dump_transcript
        from src.transcribe import transcribe as run_transcription

        report(0.15, "Transcription de la vidéo…")
        transcript = run_transcription(
            source, model=transcribe_model,
            progress=lambda value, message: report(0.15 + 0.08 * value, message),
        )
        dump_transcript(transcript, project_dir / "transcript.json")
        if captions_style is not None and not transcript.words:
            # Aucune parole détectée : on garde les clips mais sans sous-titres.
            report(0.24, "Aucune parole détectée — clips générés sans sous-titres.")
            captions_style = None
    if not vertical:
        report(0.25, f"Découpage via {encoder_label(resolved_encoder)}…")
        landscape = [
            Path(item) for item in split_video(
                source, clip_length, project_dir / "clips", encoder=encoder,
                encoding_speed=encoding_speed,
                source_start=window_start, source_end=window_end,
                on_clip=notify_clip,
            )
        ]
        report(1.0, "Exports terminés")
        return project_dir, landscape
    # Découpage + format vertical en UNE passe : la source n'est décodée qu'une
    # fois, l'encodeur et libass ne sont initialisés qu'une fois.
    clip_count = math.ceil((window_end - window_start) / clip_length)
    report(0.25, f"Découpage + format vertical via {encoder_label(resolved_encoder)}…")
    captions_file = None
    if captions_style is not None and transcript is not None:
        from src.captions import write_clip_captions

        captions_file = write_clip_captions(
            transcript, project_dir / "vertical" / "captions.ass",
            clip_start=window_start, clip_end=window_end,
            width=quality.width, height=quality.height, style=captions_style,
        )
    done = {"n": 0}

    def _on_segment(path: Path) -> None:
        done["n"] += 1
        report(
            0.25 + 0.7 * (done["n"] / max(clip_count, 1)),
            f"Clip {done['n']}/{clip_count}…",
        )
        notify_clip(path)

    exports = segment_vertical(
        source, project_dir / "vertical",
        clip_length=clip_length, window_start=window_start, window_end=window_end,
        encoder=encoder, encoding_speed=encoding_speed, quality=quality.key,
        background=vertical_background, captions_file=captions_file,
        on_clip=_on_segment,
    )
    report(1.0, "Exports terminés")
    return project_dir, exports
