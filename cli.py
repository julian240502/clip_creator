"""Clip Creator en ligne de commande — même moteur que l'interface (`src/`).

Pensé pour être piloté par un script, une tâche planifiée ou un agent IA :
les messages de progression vont sur stderr, le résultat final est un JSON
sur stdout.

    python cli.py info    <url>
    python cli.py analyze <url|fichier> [--start 10:00 --end 40:00] [--count 8]
    python cli.py frames  <url|fichier> --at 12:30,13:05
    python cli.py render  <url|fichier> --window 12:30-13:05 [--captions active]
    python cli.py render  --analysis data/analyses/<…>/analysis.json --pick 1,3
    python cli.py render  --analysis …/analysis.json --clips choix.json

Voir `python cli.py <commande> --help` et la section « Ligne de commande »
du README.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from src.paths import DATA_DIR, SOURCE_CACHE_DIR

ANALYSIS_MAX_HEIGHT = 480  # analyse = audio + vignettes, inutile de télécharger en HD
ANALYSES_DIR = Path(DATA_DIR) / "analyses"

CAPTION_TEMPLATES = {  # slug CLI -> nom du template dans src.captions.TEMPLATES
    "active": "Mot actif",
    "karaoke": "Karaoké",
    "fade": "Fondu",
    "headline": "Gros titres",
}
BACKGROUNDS = ("blur", "black", "reframe", "split")
FORMATS = ("9:16", "4:5", "1:1", "16:9", "source")


# --------------------------------------------------------------------------- utilitaires


class CliError(Exception):
    """Erreur d'usage : message clair, code de sortie 2."""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def progress(value: float, message: str) -> None:
    log(f"[{round(max(0.0, min(value, 1.0)) * 100):3d}%] {message}")


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_time(text: str | float | int) -> float:
    """`95`, `95.5`, `1:35`, `01:01:35.2` -> secondes."""
    if isinstance(text, (int, float)):
        return float(text)
    value = str(text).strip()
    if not value:
        raise CliError("Horodatage vide.")
    try:
        parts = [float(part) for part in value.split(":")]
    except ValueError as exc:
        raise CliError(f"Horodatage invalide : {value!r} (attendu 95, 1:35 ou 1:01:35).") from exc
    if len(parts) > 3:
        raise CliError(f"Horodatage invalide : {value!r}.")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def parse_window(text: str) -> tuple[float, float]:
    """`12:30-13:05` -> (750.0, 785.0)."""
    match = re.fullmatch(r"\s*([\d:.]+)\s*-\s*([\d:.]+)\s*", text)
    if not match:
        raise CliError(f"Fenêtre invalide : {text!r} (attendu DEBUT-FIN, ex. 12:30-13:05).")
    start, end = parse_time(match.group(1)), parse_time(match.group(2))
    if end <= start:
        raise CliError(f"Fenêtre invalide : {text!r} (la fin doit suivre le début).")
    return start, end


def timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    tenths = int(round((seconds - int(seconds)) * 10)) % 10
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}.{tenths}"
    return f"{minutes}:{secs:02d}.{tenths}"


def slug(text: str, limit: int = 40) -> str:
    cleaned = re.sub(r"[^\w-]+", "-", text, flags=re.UNICODE).strip("-_")
    return (cleaned or "video")[:limit]


def is_url(ref: str) -> bool:
    return bool(re.match(r"^https?://", ref.strip(), re.IGNORECASE))


def resolve_source(ref: str) -> dict:
    """`{"kind": "url"|"file", "ref": …}` + titre / durée si on les obtient."""
    if is_url(ref):
        from src.downloader import probe_url

        try:
            meta = probe_url(ref)
        except Exception as exc:  # noqa: BLE001 - message yt-dlp remonté tel quel
            raise CliError(f"Impossible de lire l'URL : {exc}") from exc
        return {
            "kind": "url", "ref": meta.get("webpage_url") or ref, "title": meta.get("title"),
            "duration": meta.get("duration"), "uploader": meta.get("uploader"),
        }
    path = Path(ref).expanduser().resolve()
    if not path.is_file():
        raise CliError(f"Fichier introuvable : {path}")
    from src.video_splitter import get_video_duration

    return {
        "kind": "file", "ref": str(path), "title": path.stem,
        "duration": get_video_duration(path), "uploader": None,
    }


def resolve_range(source: dict, start: str | None, end: str | None) -> tuple[float, float] | None:
    """Portion demandée, bornée à la durée de la source. None = vidéo entière."""
    if start is None and end is None:
        return None
    duration = source.get("duration")
    lo = parse_time(start) if start is not None else 0.0
    hi = parse_time(end) if end is not None else (duration or 0.0)
    if duration:
        hi = min(hi, float(duration))
    if hi <= lo:
        raise CliError("Portion vide : --end doit être après --start (et dans la vidéo).")
    if lo <= 0.5 and duration and hi >= float(duration) - 0.5:
        return None
    return lo, hi


def fetch_media(source: dict, window: tuple[float, float] | None, max_height: int) -> tuple[Path, float]:
    """Média local + décalage (temps source de la 1re image du fichier)."""
    if source["kind"] == "file":
        return Path(source["ref"]), 0.0
    from src.video_splitter import get_video_duration

    if window:
        from src.downloader import download_source_range

        media = Path(download_source_range(
            source["ref"], SOURCE_CACHE_DIR, window[0], window[1], max_height=max_height,
        ))
        # La coupe en copie de flux recule le début à l'image-clé précédente.
        lead = get_video_duration(media) - (window[1] - window[0])
        return media, window[0] - (lead if 0.0 < lead < 30.0 else 0.0)
    from src.downloader import download_source

    return Path(download_source(source["ref"], SOURCE_CACHE_DIR, max_height=max_height)), 0.0


def shift_transcript(transcript, delta: float):
    """Recale un transcript 0-basé (extrait téléchargé) dans le temps absolu de la source."""
    if not delta:
        return transcript
    from src.transcribe import Transcript, TranscriptSegment, Word

    return Transcript(
        language=transcript.language, duration=transcript.duration + delta, model=transcript.model,
        segments=[
            TranscriptSegment(
                start=seg.start + delta, end=seg.end + delta, text=seg.text,
                words=[Word(w.start + delta, w.end + delta, w.text) for w in seg.words],
            )
            for seg in transcript.segments
        ],
    )


def write_transcript_txt(transcript, path: Path) -> Path:
    """Transcript lisible ligne à ligne : `[12:30.4 → 12:34.1] texte`."""
    lines = [
        f"[{timecode(seg.start)} → {timecode(seg.end)}] {seg.text.strip()}"
        for seg in transcript.segments if seg.text.strip()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def loud_moments(curve, offset: float, count: int = 15, spacing: float = 10.0) -> list[dict]:
    """Pics de volume (rires, cris, hype) : les `count` plus forts, espacés de `spacing` s."""
    if not curve:
        return []
    values, hop = curve
    ranked = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    picked: list[int] = []
    for index in ranked:
        if values[index] <= 0:
            break
        if all(abs(index - other) * hop >= spacing for other in picked):
            picked.append(index)
        if len(picked) >= count:
            break
    return [
        {"t": round(offset + index * hop, 1), "at": timecode(offset + index * hop),
         "level": round(float(values[index]), 3)}
        for index in sorted(picked)
    ]


def grab_frame(media: Path, at: float, out: Path, width: int = 640) -> str | None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(at, 0.0):.2f}",
        "-i", str(media), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(out),
    ]
    if subprocess.run(command, capture_output=True).returncode == 0 and out.is_file():
        return str(out)
    return None


# --------------------------------------------------------------------------- info


def cmd_info(args: argparse.Namespace) -> dict:
    return resolve_source(args.source)


# --------------------------------------------------------------------------- analyze


def collect_chat(url: str, cache_dir: Path, window, timeout: float = 90.0):
    from src.twitch_chat import find_chat_spikes, is_twitch_vod

    if not is_twitch_vod(url):
        log("Chat : ignoré (pas un VOD Twitch).")
        return None
    box: dict = {}
    start, end = (window or (None, None))
    thread = threading.Thread(
        target=lambda: box.setdefault("spikes", find_chat_spikes(url, cache_dir, start=start, end=end)),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        log("Chat Twitch : trop lent, abandonné — on continue sans.")
    return box.get("spikes") or None


def cmd_analyze(args: argparse.Namespace) -> dict:
    from src.audio_energy import loudness_curve
    from src.highlights import find_highlights
    from src.transcribe import DEFAULT_MODEL, dump_transcript, transcribe, transcription_available, unload_models

    if not transcription_available():
        raise CliError("faster-whisper manquant : pip install -r requirements-transcribe.txt")
    source = resolve_source(args.source)
    window = resolve_range(source, args.start, args.end)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else ANALYSES_DIR / f"{stamp}-{slug(source.get('title') or 'video')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    progress(0.05, "Téléchargement (basse déf)…" if source["kind"] == "url" else "Lecture du fichier…")
    media, offset = fetch_media(source, window, ANALYSIS_MAX_HEIGHT)

    progress(0.25, "Transcription…")
    model = args.whisper_model or DEFAULT_MODEL
    if source["kind"] == "url":
        transcript = shift_transcript(transcribe(media, model=model), offset)
    else:
        transcript = transcribe(media, model=model, clip_range=window)
    unload_models()  # libère la VRAM avant Ollama
    transcript_json = dump_transcript(transcript, out_dir / "transcript.json")
    transcript_txt = write_transcript_txt(transcript, out_dir / "transcript.txt")

    progress(0.55, "Enveloppe audio…")
    curve = loudness_curve(str(media))
    audio_curve = (curve[0], curve[1], offset) if curve else None

    chat_spikes = None
    if args.chat and source["kind"] == "url":
        progress(0.6, "Chat Twitch…")
        chat_spikes = collect_chat(source["ref"], out_dir, window)

    rating_model = None
    if not args.no_llm:
        from src.llm import ollama_available, pick_rating_model

        rating_model = args.rating_model or (pick_rating_model() if ollama_available() else None)
    progress(0.7, f"Notation des extraits ({rating_model or 'heuristique'})…")
    found = find_highlights(
        transcript, target_count=args.count, min_duration=args.min, max_duration=args.max,
        model=rating_model, source_window=window, audio_curve=audio_curve,
        chat_spikes=chat_spikes, content_profile=args.profile,
    )

    thumbs = out_dir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    highlights = []
    for index, item in enumerate(found, start=1):
        data = asdict(item)
        data = {"id": index, **data, "duration": round(item.end - item.start, 2),
                "range": f"{timecode(item.start)}-{timecode(item.end)}"}
        middle = (item.start + item.end) / 2 - offset
        data["thumb"] = grab_frame(media, middle, thumbs / f"hl_{index:02d}.jpg", width=480)
        highlights.append(data)

    result = {
        "analysis_dir": str(out_dir),
        "source": source,
        "window": list(window) if window else None,
        "language": transcript.language,
        "transcript_json": str(transcript_json),
        "transcript_txt": str(transcript_txt),
        "rating_model": rating_model,
        "content_profile": args.profile,
        "highlights": highlights,
        "loud_moments": loud_moments(curve, offset),
        "chat_spikes": [
            {"t": round(t, 1), "at": timecode(t), "intensity": round(v, 2)} for t, v in (chat_spikes or [])
        ],
    }
    analysis_path = out_dir / "analysis.json"
    analysis_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(1.0, f"Analyse écrite dans {analysis_path}")
    return {"analysis": str(analysis_path), **result}


# --------------------------------------------------------------------------- frames


def cmd_frames(args: argparse.Namespace) -> dict:
    source = resolve_source(args.source)
    times: list[float] = []
    if args.at:
        times = [parse_time(part) for part in args.at.split(",") if part.strip()]
    if args.window:
        start, end = parse_window(args.window)
        count = max(1, args.count)
        step = (end - start) / count
        times += [start + step * (k + 0.5) for k in range(count)]
    if not times:
        raise CliError("Indique --at T1,T2… ou --window DEBUT-FIN.")
    out_dir = Path(args.out) if args.out else Path(DATA_DIR) / "frames" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in sorted(times):
        if source["kind"] == "url":
            media, offset = fetch_media(source, (max(0.0, t - 2.0), t + 2.0), args.height)
        else:
            media, offset = Path(source["ref"]), 0.0
        path = grab_frame(media, t - offset, out_dir / f"frame_{int(t * 10):07d}.jpg", width=args.width)
        frames.append({"t": round(t, 2), "at": timecode(t), "path": path})
    return {"frames": frames, "dir": str(out_dir)}


# --------------------------------------------------------------------------- render


def load_clip_specs(args: argparse.Namespace, analysis: dict | None) -> list[dict]:
    """Fenêtres à rendre, avec titre / description / hashtags optionnels."""
    specs: list[dict] = []
    if args.pick:
        if not analysis:
            raise CliError("--pick nécessite --analysis.")
        by_id = {item["id"]: item for item in analysis.get("highlights", [])}
        for raw in args.pick.split(","):
            raw = raw.strip()
            if not raw:
                continue
            if not raw.isdigit() or int(raw) not in by_id:
                raise CliError(f"--pick : extrait {raw!r} absent de l'analyse (ids : {sorted(by_id)}).")
            item = by_id[int(raw)]
            specs.append({"start": item["start"], "end": item["end"],
                          "hint_title": item.get("title", ""), "hint_summary": item.get("summary", "")})
    if args.clips:
        data = json.loads(Path(args.clips).read_text(encoding="utf-8"))
        items = data.get("clips", []) if isinstance(data, dict) else data
        for item in items:
            start, end = parse_time(item["start"]), parse_time(item["end"])
            if end <= start:
                raise CliError(f"--clips : fenêtre invalide {item!r}.")
            tags = item.get("hashtags") or []
            if isinstance(tags, str):
                tags = tags.split()
            specs.append({
                "start": start, "end": end,
                "title": item.get("title"), "description": item.get("description"),
                "hashtags": [tag if tag.startswith("#") else f"#{tag}" for tag in map(str, tags)],
            })
    for window in args.window or []:
        start, end = parse_window(window)
        specs.append({"start": start, "end": end})
    # process_video trie les fenêtres : on trie pareil pour garder la correspondance.
    specs.sort(key=lambda spec: (float(spec["start"]), float(spec["end"])))
    return specs


def build_caption_style(args: argparse.Namespace):
    if not args.captions:
        return None
    from src.captions import TEMPLATES

    style = TEMPLATES[CAPTION_TEMPLATES[args.captions]]
    changes = {}
    if args.caption_position:
        changes["position"] = args.caption_position
    if args.caption_size:
        changes["font_size"] = args.caption_size
    if args.caption_font:
        changes["font"] = args.caption_font
    if args.caption_color:
        changes["highlight_color"] = args.caption_color
    if args.uppercase:
        changes["uppercase"] = True
    return replace(style, **changes) if changes else style


def build_split_layout(args: argparse.Namespace):
    if args.background != "split":
        return None
    if not args.facecam:
        raise CliError("--background split nécessite --facecam X,Y,L,H (fractions 0-1 de la source).")
    try:
        box = tuple(float(part) for part in args.facecam.split(","))
    except ValueError as exc:
        raise CliError("--facecam : 4 nombres séparés par des virgules, ex. 0.75,0.05,0.22,0.3") from exc
    if len(box) != 4 or not all(0.0 <= value <= 1.0 for value in box):
        raise CliError("--facecam : 4 fractions entre 0 et 1 (X,Y,largeur,hauteur).")
    from src.resizer import SplitLayout

    return SplitLayout(top=box, top_frac=args.facecam_height)


def cmd_render(args: argparse.Namespace) -> dict:
    from src.pipeline import _publish_to_folder, process_video
    from src.transcribe import Transcript, load_transcript

    analysis = None
    if args.analysis:
        analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    ref = args.source or (analysis or {}).get("source", {}).get("ref")
    if not ref:
        raise CliError("Indique une source (URL ou fichier) ou --analysis.")
    if analysis and args.source is None:
        source = analysis["source"]
    else:
        source = resolve_source(ref)

    specs = load_clip_specs(args, analysis)
    vertical = args.format != "source"
    if specs and not vertical:
        raise CliError("Les fenêtres (--window/--pick/--clips) demandent un format vertical/carré/16:9, pas 'source'.")

    captions_style = build_caption_style(args)
    split_layout = build_split_layout(args)

    transcript = None
    if analysis and analysis.get("transcript_json") and Path(analysis["transcript_json"]).is_file():
        transcript = load_transcript(analysis["transcript_json"])

    windows = [(float(spec["start"]), float(spec["end"])) for spec in specs] or None
    hints = [(spec.get("title") or spec.get("hint_title") or "",
              spec.get("description") or spec.get("hint_summary") or "") for spec in specs] or None

    source_start = source_end = None
    pretranscript = None
    if source["kind"] == "file":
        pretranscript = transcript  # fichier local : même référentiel de temps que l'analyse
    elif windows:
        if transcript is not None:
            pretranscript = transcript  # chaque fenêtre est téléchargée seule
        elif captions_style is None and not args.meta:
            pretranscript = Transcript(language="", duration=0.0, model="")  # rien à sous-titrer
        else:
            # Pas d'analyse : on télécharge la portion qui couvre les fenêtres et on la transcrit.
            source_start, source_end = min(w[0] for w in windows), max(w[1] for w in windows)
            if source_end - source_start > 1800:
                log("Attention : fenêtres très espacées sans --analysis — gros téléchargement. "
                    "Lance plutôt `analyze` d'abord.")
    if not windows:
        window = resolve_range(source, args.start, args.end)
        if window:
            source_start, source_end = window

    meta_model = None
    if args.meta:
        from src.llm import ollama_available, pick_model

        meta_model = pick_model() if ollama_available() else None
        if meta_model is None:
            log("Ollama absent : titres & hashtags en mode heuristique.")

    project_dir, exports = process_video(
        url=source["ref"] if source["kind"] == "url" else None,
        uploaded_path=source["ref"] if source["kind"] == "file" else None,
        clip_length=args.length, vertical=vertical, export_format=args.format,
        encoder=args.encoder, export_quality=args.quality, encoding_speed=args.speed,
        vertical_background=args.background, split_layout=split_layout,
        source_start=source_start, source_end=source_end, source_duration=source.get("duration"),
        clips_windows=windows, pretranscript=pretranscript,
        captions_style=captions_style, caption_lang=args.caption_lang,
        generate_meta=bool(args.meta), meta_model=meta_model, clips_hints=hints,
        video_title=source.get("title"), progress=progress,
    )

    # Textes fournis par l'appelant (ex. écrits par Claude) : prioritaires sur Ollama.
    from src.metadata import ClipMeta, write_metadata

    clips = []
    for index, path in enumerate(exports):
        path = Path(path)
        spec = specs[index] if index < len(specs) else {}
        if spec.get("title") or spec.get("description") or spec.get("hashtags"):
            write_metadata(
                ClipMeta(title=spec.get("title") or "", description=spec.get("description") or "",
                         hashtags=spec.get("hashtags") or []),
                path.with_suffix(".txt"),
            )
        sidecar = path.with_suffix(".txt")
        clips.append({
            "path": str(path),
            "start": spec.get("start"), "end": spec.get("end"),
            "text_file": str(sidecar) if sidecar.is_file() else None,
        })

    published = None
    if args.publish:
        lang = args.caption_lang or (analysis or {}).get("language") or (transcript.language if transcript else "")
        session = datetime.now().strftime("%Y-%m-%d %Hh%M")
        published = str(_publish_to_folder(
            exports, args.publish, lang or "XX", args.label or source.get("uploader") or "Clips", session,
        ))
        log(f"Clips copiés dans {published}")

    return {"project_dir": str(project_dir), "clips": clips, "published_to": published}


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py", description="Clip Creator en ligne de commande (résultat JSON sur stdout).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="Titre, durée, chaîne d'une URL (sans télécharger).")
    info.add_argument("source", help="URL ou fichier vidéo")
    info.set_defaults(func=cmd_info)

    analyze = sub.add_parser("analyze", help="Transcrit et repère les meilleurs moments.")
    analyze.add_argument("source", help="URL ou fichier vidéo")
    analyze.add_argument("--start", help="début de la portion à analyser (95, 1:35, 1:01:35)")
    analyze.add_argument("--end", help="fin de la portion à analyser")
    analyze.add_argument("--count", type=int, default=8, help="nombre d'extraits visés (défaut 8)")
    analyze.add_argument("--min", type=float, default=20.0, help="durée min d'un extrait, s (défaut 20)")
    analyze.add_argument("--max", type=float, default=75.0, help="durée max d'un extrait, s (défaut 75)")
    analyze.add_argument("--profile", choices=("gaming", "podcast", "balanced"), default="gaming",
                         help="pondération du score (défaut gaming)")
    analyze.add_argument("--chat", action="store_true", help="utiliser le chat (VOD Twitch uniquement)")
    analyze.add_argument("--no-llm", action="store_true", help="ne pas appeler Ollama (notation heuristique)")
    analyze.add_argument("--rating-model", help="modèle Ollama de notation (défaut : auto)")
    analyze.add_argument("--whisper-model", help="modèle Whisper (défaut : celui de l'app)")
    analyze.add_argument("--out", help="dossier de sortie (défaut data/analyses/<date>-<titre>)")
    analyze.set_defaults(func=cmd_analyze)

    frames = sub.add_parser("frames", help="Extrait des images pour vérifier visuellement un passage.")
    frames.add_argument("source", help="URL ou fichier vidéo")
    frames.add_argument("--at", help="instants séparés par des virgules (ex. 12:30,13:05)")
    frames.add_argument("--window", help="DEBUT-FIN : images réparties sur la fenêtre")
    frames.add_argument("--count", type=int, default=6, help="images sur --window (défaut 6)")
    frames.add_argument("--width", type=int, default=640, help="largeur des images (défaut 640)")
    frames.add_argument("--height", type=int, default=720, help="définition max téléchargée (URL)")
    frames.add_argument("--out", help="dossier de sortie")
    frames.set_defaults(func=cmd_frames)

    render = sub.add_parser("render", help="Génère les clips.")
    render.add_argument("source", nargs="?", help="URL ou fichier (facultatif avec --analysis)")
    render.add_argument("--analysis", help="analysis.json produit par `analyze` (réutilise le transcript)")
    render.add_argument("--pick", help="ids d'extraits de l'analyse, ex. 1,3,4")
    render.add_argument("--clips", help="JSON : [{start, end, title?, description?, hashtags?}, …]")
    render.add_argument("--window", action="append", help="DEBUT-FIN, répétable (ex. --window 12:30-13:05)")
    render.add_argument("--start", help="mode régulier : début de la portion")
    render.add_argument("--end", help="mode régulier : fin de la portion")
    render.add_argument("--length", type=int, default=30, help="mode régulier : durée d'un clip, s (défaut 30)")
    render.add_argument("--format", choices=FORMATS, default="9:16")
    render.add_argument("--quality", choices=("720p", "1080p", "4k"), default="1080p")
    render.add_argument("--encoder", choices=("auto", "cpu", "nvidia", "intel", "amd"), default="auto")
    render.add_argument("--speed", choices=("fast", "balanced", "quality"), default="balanced")
    render.add_argument("--background", choices=BACKGROUNDS, default="blur",
                        help="blur (défaut), black, reframe (suivi visage), split (facecam haut / jeu bas)")
    render.add_argument("--facecam", help="split : cadre du facecam X,Y,L,H en fractions (ex. 0.75,0.05,0.22,0.3)")
    render.add_argument("--facecam-height", type=float, default=0.40, help="split : part du panneau haut (0.4)")
    render.add_argument("--captions", choices=tuple(CAPTION_TEMPLATES), help="sous-titres incrustés")
    render.add_argument("--caption-lang", choices=("fr", "en", "zh", "ko"), help="traduire les sous-titres")
    render.add_argument("--caption-position", choices=("top", "middle", "bottom"))
    render.add_argument("--caption-size", type=int, help="taille de police (px)")
    render.add_argument("--caption-font", help="police (ex. Arial Black, Impact)")
    render.add_argument("--caption-color", help="couleur du mot actif (#RRGGBB)")
    render.add_argument("--uppercase", action="store_true", help="sous-titres en majuscules")
    render.add_argument("--meta", action="store_true",
                        help="titres & hashtags via Ollama (inutile si --clips fournit les textes)")
    render.add_argument("--publish", help="copier les clips dans ce dossier (ex. Google Drive synchronisé)")
    render.add_argument("--label", help="sous-dossier créateur / streamer pour --publish")
    render.set_defaults(func=cmd_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        emit(args.func(args))
    except CliError as exc:
        log(f"Erreur : {exc}")
        return 2
    except KeyboardInterrupt:
        log("Interrompu.")
        return 130
    except Exception as exc:  # noqa: BLE001 - message lisible + code d'erreur
        log(f"Échec : {exc}")
        if os.environ.get("CLIP_CREATOR_DEBUG"):
            raise
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
