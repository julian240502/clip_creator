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
from src.quality import frame_size, get_quality_preset
from src.resizer import resize_clip_for_vertical, segment_vertical
from src.transcribe import DEFAULT_MODEL
from src.video_splitter import (
    get_video_duration,
    get_video_resolution,
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


def _safe_folder(name: str) -> str:
    """Nom de dossier lisible (garde espaces/casse) mais sans caractère interdit."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", (name or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned[:80] or "Clips"


def _publish_to_folder(
    exports: list[Path], export_dir: str | Path, label: str, session: str,
) -> Path:
    """Copie les clips et leurs `.txt` dans `<export_dir>/<créateur>/<session>/`.

    Deux sous-dossiers parallèles : `clips/` (vidéos) et `textes/` (titre +
    description + hashtags), le `.txt` gardant le nom du clip → on retrouve
    facilement la paire. Pensé pour un dossier Google Drive synchronisé : tout
    se retrouve sur le téléphone, prêt à poster.
    """
    root = Path(export_dir).expanduser() / _safe_folder(label) / session
    clips_dir = root / "clips"
    texts_dir = root / "textes"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for clip in exports:
        clip = Path(clip)
        shutil.copy2(clip, clips_dir / clip.name)
        sidecar = clip.with_suffix(".txt")
        if sidecar.is_file():
            texts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sidecar, texts_dir / sidecar.name)
    return root


def _window_text(transcript, start: float, end: float) -> str:
    return " ".join(
        word.text.strip() for word in transcript.words
        if word.end > start and word.start < end
    ).strip()


def _make_crop_cmd(track, source_w, source_h, frame_w, frame_h, out_path, *, win_start, win_end):
    from src.reframe import centred_crop_path, crop_box, crop_path, write_sendcmd

    crop_w, crop_h = crop_box(source_w, source_h, frame_w, frame_h)
    if track:
        path = crop_path(track, source_w, source_h, crop_w, crop_h, t_start=win_start, t_end=win_end)
    else:
        path = centred_crop_path(source_w, source_h, crop_w, crop_h)
    return write_sendcmd(out_path, path)


def _write_metadata_files(exports, windows, transcript, model, hints, video_title="") -> None:
    from src.metadata import generate_metadata, write_metadata

    video_context = transcript.text
    for index, clip_path in enumerate(exports):
        clip_start, clip_end = windows[index]
        hint_title, hint_summary = hints[index] if hints and index < len(hints) else ("", "")
        meta = generate_metadata(
            _window_text(transcript, clip_start, clip_end),
            hint_title=hint_title, hint_summary=hint_summary, model=model,
            language=transcript.language,
            video_title=video_title, video_context=video_context,
        )
        write_metadata(meta, Path(clip_path).with_suffix(".txt"))


def process_video(
    *, url: str | None = None, uploaded_path: str | Path | None = None,
    clip_length: int = 30, vertical: bool = True,
    export_format: str = "9:16",
    encoder: str = "auto", export_quality: str = "1080p",
    encoding_speed: str = "balanced",
    vertical_background: str = "blur",
    source_start: float | None = None,
    source_end: float | None = None,
    clips_windows: list[tuple[float, float]] | None = None,
    transcribe: bool = False,
    transcribe_model: str = DEFAULT_MODEL,
    captions_style: CaptionStyle | None = None,
    generate_meta: bool = False,
    meta_model: str | None = None,
    clips_hints: list[tuple[str, str]] | None = None,
    video_title: str | None = None,
    export_dir: str | Path | None = None,
    export_label: str | None = None,
    on_clip: ClipCallback | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Path, list[Path]]:
    """Exécute le pipeline et renvoie le dossier projet et les exports finaux."""
    report = progress or (lambda _value, _message: None)
    notify_clip = on_clip or (lambda _path: None)
    if bool(url) == bool(uploaded_path):
        raise ValueError("Fournissez une URL ou un fichier, mais pas les deux.")
    if export_dir:
        # Échoue tôt (avant le rendu) si le dossier d'export est inaccessible.
        try:
            Path(export_dir).expanduser().mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"Dossier d'export inaccessible : {exc}") from exc
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    hint = Path(uploaded_path).stem if uploaded_path else "download"
    project_dir = Path(DATA_DIR) / "projects" / f"{stamp}-{_project_name(hint)}"
    export_session = "-".join(project_dir.name.split("-")[:2])  # AAAAMMJJ-HHMMSS

    def _maybe_publish(items: list[Path]) -> None:
        if not export_dir:
            return
        dest = _publish_to_folder(items, export_dir, export_label or "", export_session)
        report(0.99, f"Clips copiés vers {dest}")

    source_dir = project_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=False)
    quality = get_quality_preset(export_quality)
    frame_w, frame_h = frame_size(export_quality, export_format)
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
    reframe = vertical and vertical_background == "reframe"
    face_track: list | None = None
    source_dims = (0, 0)
    if reframe:
        from src.reframe import detect_face_track, reframe_available

        source_dims = get_video_resolution(source)
        if reframe_available():
            report(0.24, "Analyse des visages…")
            face_track = detect_face_track(source)
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
        _maybe_publish(landscape)
        report(1.0, "Exports terminés")
        return project_dir, landscape
    if clips_windows:
        # Mode « sélection intelligente » : on rend exactement les fenêtres choisies.
        windows = sorted((float(s), float(e)) for s, e in clips_windows if e > s)
        vertical_dir, exports = project_dir / "vertical", []
        report(0.25, f"Rendu de {len(windows)} clip(s) via {encoder_label(resolved_encoder)}…")
        for index, (clip_start, clip_end) in enumerate(windows):
            report(
                0.25 + 0.7 * index / max(len(windows), 1),
                f"Clip {index + 1}/{len(windows)}…",
            )
            output = vertical_dir / f"clip_{index + 1:03d}.mp4"
            clip_captions = None
            if captions_style is not None and transcript is not None:
                from src.captions import write_clip_captions

                clip_captions = write_clip_captions(
                    transcript, output.with_suffix(".ass"),
                    clip_start=clip_start, clip_end=clip_end,
                    width=frame_w, height=frame_h, style=captions_style,
                )
            crop_cmd = None
            if reframe:
                crop_cmd = _make_crop_cmd(
                    face_track, *source_dims, frame_w, frame_h,
                    output.with_suffix(".cmd"), win_start=clip_start, win_end=clip_end,
                )
            clip_path = resize_clip_for_vertical(
                source, output, encoder=encoder, encoding_speed=encoding_speed,
                quality=quality.key, aspect=export_format, background=vertical_background,
                start=clip_start, duration=clip_end - clip_start,
                captions_file=clip_captions, crop_cmd_file=crop_cmd,
            )
            exports.append(clip_path)
            notify_clip(clip_path)
        if generate_meta and transcript is not None and transcript.words:
            report(0.96, "Titres & hashtags…")
            _write_metadata_files(
                exports, windows, transcript, meta_model, clips_hints, video_title or "",
            )
        _maybe_publish(exports)
        report(1.0, "Exports terminés")
        return project_dir, exports
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
            width=frame_w, height=frame_h, style=captions_style,
        )
    done = {"n": 0}

    def _on_segment(path: Path) -> None:
        done["n"] += 1
        report(
            0.25 + 0.7 * (done["n"] / max(clip_count, 1)),
            f"Clip {done['n']}/{clip_count}…",
        )
        notify_clip(path)

    crop_cmd = None
    if reframe:
        crop_cmd = _make_crop_cmd(
            face_track, *source_dims, frame_w, frame_h,
            project_dir / "vertical" / "reframe.cmd",
            win_start=window_start, win_end=window_end,
        )
    exports = segment_vertical(
        source, project_dir / "vertical",
        clip_length=clip_length, window_start=window_start, window_end=window_end,
        encoder=encoder, encoding_speed=encoding_speed, quality=quality.key,
        aspect=export_format, background=vertical_background, captions_file=captions_file,
        crop_cmd_file=crop_cmd, on_clip=_on_segment,
    )
    if generate_meta and transcript is not None and transcript.words:
        report(0.96, "Titres & hashtags…")
        seg_windows = [
            (window_start + k * clip_length,
             min(window_start + (k + 1) * clip_length, window_end))
            for k in range(len(exports))
        ]
        _write_metadata_files(exports, seg_windows, transcript, meta_model, None, video_title or "")
    _maybe_publish(exports)
    report(1.0, "Exports terminés")
    return project_dir, exports
