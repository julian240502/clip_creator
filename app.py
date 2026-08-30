from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import threading
import uuid
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import streamlit as st

from src.captions import (
    CAPTION_FONTS,
    CAPTION_MODES,
    CAPTION_POSITIONS,
    TEMPLATES,
    write_clip_captions,
)
from src.downloader import download_clip, probe_url
from src.pipeline import process_video
from src.quality import get_quality_preset
from src.resizer import resize_clip_for_vertical
from src.transcribe import (
    DEFAULT_MODEL,
    load_transcript,
    prewarm_model,
    transcribe,
    transcription_available,
)
from src.video_splitter import get_video_duration, get_video_resolution

st.set_page_config(page_title="Clip Creator", page_icon="✦", layout="wide")
st.markdown(
    """
<style>
html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.stApp {background:radial-gradient(circle at 15% 0%,#20234b 0,#0d0e18 38%,#090a10 100%);color:#f4f5fb}
.hero {padding:2rem 0 1.2rem}
.eyebrow {color:#b7baff;font-weight:700;letter-spacing:.13em;text-transform:uppercase;font-size:.78rem}
.hero h1 {font-size:3rem;line-height:1;margin:.4rem 0;background:linear-gradient(90deg,#fff,#b8bcff);-webkit-background-clip:text;color:transparent}
.hero p {font-size:1.05rem;color:#c6c8d6;max-width:640px}

/* Lisibilité du texte secondaire sur le fond sombre */
button[data-baseweb="tab"] {color:#cbcddc}
button[data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {color:inherit;font-weight:600}
button[data-baseweb="tab"][aria-selected="true"] {color:#fff}
label p, [data-testid="stWidgetLabel"] p {color:#e2e3f0 !important}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:#bcbfd0}
[data-testid="stExpander"] summary p {color:#e2e3f0}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {color:#bcbfd0}

/* Estimation mise en avant */
[data-testid="stMetric"] {background:#181a26;border:1px solid #2c2f42;border-left:3px solid #8b87ff;padding:1rem 1.1rem;border-radius:14px}
[data-testid="stMetricValue"] {color:#fff;font-weight:700}
[data-testid="stMetricLabel"] p {color:#cbcddc !important;text-transform:uppercase;letter-spacing:.08em;font-size:.72rem}
[data-testid="stMetricDelta"] {color:#bcbfd0}

.stButton>button,.stDownloadButton>button {border-radius:12px;font-weight:700;border:0;background:linear-gradient(90deg,#7774ff,#a855f7);color:#fff}
.clip-card {background:#171924;border:1px solid #292c3d;border-radius:16px;padding:.9rem;margin:.5rem 0}
</style>
""",
    unsafe_allow_html=True,
)

WORK_ROOT = Path(tempfile.gettempdir()) / "clip-creator"
QUALITY_CHOICES = {
    "Full HD · 1080p — recommandé": "1080p",
    "HD · 720p — plus rapide": "720p",
    "4K · 2160p — meilleure qualité": "4k",
}
BACKGROUND_CHOICES = {"Fond vidéo flouté — recommandé": "blur", "Bandes noires": "black"}
PREVIEW_QUALITY = "720p"  # l'aperçu reste léger quelle que soit la qualité d'export


def session_dir() -> Path:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]
    path = WORK_ROOT / st.session_state.session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_source() -> None:
    for key in (
        "source", "clips", "project_dir", "captions_skipped",
        "style_preview", "style_preview_sig", "preview_at",
    ):
        st.session_state.pop(key, None)
    shutil.rmtree(session_dir(), ignore_errors=True)


def timecode(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def render_clip_card(clip: Path, key: str) -> None:
    st.video(str(clip))
    st.caption(clip.name)
    with clip.open("rb") as handle:
        st.download_button(
            "Télécharger", handle, clip.name, "video/mp4",
            key=key, use_container_width=True,
        )


def _preview_source(source: dict, at: float, seconds: float = 4.0) -> Path:
    """Extrait court et local pour l'aperçu ; réutilisé tant que la portion ne change pas."""
    preview_dir = session_dir() / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    existing = next(preview_dir.glob("preview_source.*"), None)
    if existing is not None and st.session_state.get("preview_at") == round(at, 2):
        return existing
    if source["kind"] == "url":
        clip = Path(download_clip(source["ref"], preview_dir, at, at + seconds, max_height=480))
    else:
        clip = preview_dir / "preview_source.mp4"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(at), "-i", source["path"], "-t", str(seconds),
            "-c:v", "libx264", "-c:a", "aac", str(clip),
        ]
        subprocess.run(command, capture_output=True, check=True)
    st.session_state["preview_at"] = round(at, 2)
    return clip


def _transcript_has_words(project_dir: str) -> bool:
    path = Path(project_dir) / "transcript.json"
    if not path.is_file():
        return False
    try:
        return bool(load_transcript(path).words)
    except Exception:  # noqa: BLE001 - transcription illisible = pas de sous-titres
        return False


def render_style_preview(source: dict, at: float, style, background: str) -> Path:
    short = _preview_source(source, at)
    duration = min(4.0, get_video_duration(short))
    quality = get_quality_preset(PREVIEW_QUALITY)
    # Même modèle que le rendu final pour que le texte de l'aperçu soit fidèle.
    transcript = transcribe(short, model=DEFAULT_MODEL, cache_dir=session_dir())
    if not transcript.words:
        raise RuntimeError(
            "Aucune parole détectée dans cet extrait — impossible de générer des sous-titres."
        )
    ass = write_clip_captions(
        transcript, session_dir() / "preview" / "preview.ass",
        clip_start=0.0, clip_end=duration,
        width=quality.width, height=quality.height, style=style,
    )
    output = session_dir() / "preview" / "preview.mp4"
    resize_clip_for_vertical(
        short, output, quality=PREVIEW_QUALITY, background=background,
        start=0.0, duration=duration, encoding_speed="fast", captions_file=ass,
    )
    return output


# --- Phase 1 : charger une vidéo -------------------------------------------------
if "source" not in st.session_state:
    st.markdown(
        """<section class="hero"><div class="eyebrow">Studio vidéo local</div>
        <h1>Clip Creator ✦</h1>
        <p>Chargez une vidéo, prévisualisez-la, choisissez la portion à traiter,
        puis générez des clips verticaux prêts à finaliser dans CapCut ou Camtasia.</p></section>""",
        unsafe_allow_html=True,
    )
    tab_link, tab_file = st.tabs(["Lien vidéo", "Fichier local"])
    with tab_link:
        url = st.text_input("URL de la vidéo", placeholder="https://www.youtube.com/watch?v=…")
        if st.button("Charger la vidéo", key="load_url", use_container_width=True):
            if not url.strip():
                st.warning("Ajoutez une URL vidéo.")
            else:
                with st.spinner("Lecture des informations…"):
                    try:
                        meta = probe_url(url.strip())
                        st.session_state.source = {"kind": "url", "ref": meta["webpage_url"], **meta}
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001 - message affiché tel quel
                        st.error(f"Impossible de charger cette URL : {exc}")
    with tab_file:
        uploaded = st.file_uploader("Déposer une vidéo", type=["mp4", "mov", "mkv", "webm"])
        if uploaded is not None and st.button("Charger la vidéo", key="load_file", use_container_width=True):
            destination = session_dir() / Path(uploaded.name).name
            destination.write_bytes(uploaded.getbuffer())
            try:
                duration = get_video_duration(destination)
                width, height = get_video_resolution(destination)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Fichier vidéo illisible : {exc}")
                st.stop()
            st.session_state.source = {
                "kind": "file", "ref": str(destination), "path": str(destination),
                "title": destination.stem, "duration": duration,
                "width": width, "height": height, "thumbnail": None,
            }
            st.rerun()
    st.stop()

source = st.session_state.source

# --- Phase 3 : clips générés --------------------------------------------------
if st.session_state.get("clips"):
    clips = [Path(item) for item in st.session_state.clips]
    st.markdown('<section class="hero"><h1>Clips prêts ✦</h1></section>', unsafe_allow_html=True)
    st.caption(f"{len(clips)} clip(s) · {source['title']}")
    if st.session_state.get("captions_skipped"):
        st.warning("Aucune parole détectée dans la vidéo : clips générés sans sous-titres.")

    project_dir = st.session_state.get("project_dir")
    if project_dir:
        archive_path = Path(project_dir) / "clip-creator-exports.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
            for clip in clips:
                archive.write(clip, clip.name)
        with archive_path.open("rb") as archive:
            st.download_button(
                "Télécharger tous les clips (.zip)", archive, "clips.zip",
                "application/zip", use_container_width=True,
            )

    columns = st.columns(3)
    for index, clip in enumerate(clips):
        with columns[index % 3]:
            render_clip_card(clip, key=f"clip-{index}")

    left, right = st.columns(2)
    if left.button("Régler à nouveau", use_container_width=True):
        for key in ("clips", "project_dir", "captions_skipped"):
            st.session_state.pop(key, None)
        st.rerun()
    if right.button("Nouvelle vidéo", use_container_width=True):
        reset_source()
        st.rerun()
    st.stop()

# --- Phase 2 : prévisualiser et régler --------------------------------------
st.markdown('<section class="hero"><h1>Clip Creator ✦</h1></section>', unsafe_allow_html=True)
preview, meta = st.columns([2, 1])
with preview:
    try:
        st.video(source["ref"])
    except Exception:  # noqa: BLE001 - lecteur indisponible pour cette source
        if source.get("thumbnail"):
            st.image(source["thumbnail"], use_container_width=True)
        else:
            st.info("Aperçu vidéo indisponible pour cette source.")
with meta:
    st.markdown(f"**{source['title']}**")
    facts = []
    duration = source.get("duration")
    if duration:
        facts.append(timecode(duration))
    if source.get("width") and source.get("height"):
        facts.append(f"{source['width']} × {source['height']}")
        facts.append("paysage" if source["width"] >= source["height"] else "portrait")
    st.caption(" · ".join(facts) or "Métadonnées indisponibles")
    if source.get("uploader"):
        st.caption(source["uploader"])
    if st.button("Changer de vidéo", use_container_width=True):
        reset_source()
        st.rerun()

st.subheader("Réglages")
clip_length = st.slider("Durée d'un clip", 10, 180, 30, 5, format="%d sec")

if duration:
    window = st.slider(
        "Portion à clipper", 0.0, float(duration), (0.0, float(duration)),
        step=1.0, format="%d s",
    )
    span = window[1] - window[0]
    estimated = math.ceil(span / clip_length) if span > 0 else 0
    st.metric("Clips à générer", f"≈ {estimated}")
    st.caption(f"{clip_length} s l'unité · sur {timecode(span)} de vidéo sélectionnée")
else:
    window = (0.0, None)
    st.caption("Durée inconnue : toute la vidéo sera traitée.")

quality_key = QUALITY_CHOICES[st.selectbox("Qualité", list(QUALITY_CHOICES))]
vertical = st.toggle("Exporter en 9:16", value=True)
background = "blur"
if vertical:
    background = BACKGROUND_CHOICES[st.selectbox("Arrière-plan vertical", list(BACKGROUND_CHOICES))]
    st.caption("La vidéo paysage nette reste entièrement visible au premier plan.")

captions_ready = transcription_available()
captions_on = st.toggle(
    "Sous-titres incrustés", value=False, disabled=not (captions_ready and vertical),
)
captions_style = None
if not captions_ready:
    st.caption("Nécessite `pip install -r requirements-transcribe.txt` (faster-whisper).")
elif not vertical:
    st.caption("Les sous-titres ne sont disponibles que sur l'export 9:16.")
elif captions_on:
    # Charge le modèle en tâche de fond pendant que l'utilisateur règle le style.
    if not st.session_state.get("model_warming"):
        st.session_state["model_warming"] = True
        threading.Thread(target=prewarm_model, daemon=True).start()
    with st.container(border=True):
        template_name = st.selectbox("Style", list(TEMPLATES))
        base = TEMPLATES[template_name]
        col_a, col_b, col_c = st.columns(3)
        font = col_a.selectbox(
            "Police", CAPTION_FONTS,
            index=CAPTION_FONTS.index(base.font) if base.font in CAPTION_FONTS else 0,
        )
        font_size = col_b.slider("Taille", 32, 130, base.font_size, 2)
        position_label = col_c.selectbox(
            "Position", list(CAPTION_POSITIONS),
            index=list(CAPTION_POSITIONS.values()).index(base.position),
        )
        col_d, col_e = st.columns(2)
        primary_color = col_d.color_picker("Couleur du texte", base.primary_color)
        highlight_color = col_e.color_picker("Couleur du mot actif", base.highlight_color)
        mode_label = st.selectbox(
            "Apparition", list(CAPTION_MODES),
            index=list(CAPTION_MODES.values()).index(base.mode),
        )
        uppercase = st.toggle("MAJUSCULES", value=base.uppercase)
        captions_style = replace(
            base, font=font, font_size=font_size,
            position=CAPTION_POSITIONS[position_label],
            primary_color=primary_color, highlight_color=highlight_color,
            mode=CAPTION_MODES[mode_label], uppercase=uppercase,
        )
        style_sig = json.dumps(asdict(captions_style), sort_keys=True)
        st.caption(
            "Le modèle se charge en arrière-plan. Le 1er aperçu prend quelques secondes de plus, "
            "les suivants sont quasi instantanés."
        )
        if st.button("Aperçu du style", use_container_width=True):
            with st.spinner("Rendu de l'aperçu…"):
                try:
                    preview = render_style_preview(
                        source, window[0] or 0.0, captions_style, background,
                    )
                    st.session_state["style_preview"] = str(preview)
                    st.session_state["style_preview_sig"] = style_sig
                except Exception as exc:  # noqa: BLE001 - message affiché tel quel
                    st.session_state.pop("style_preview", None)
                    st.error(f"Aperçu impossible : {exc}")
        preview_path = st.session_state.get("style_preview")
        if preview_path and Path(preview_path).is_file():
            if st.session_state.get("style_preview_sig") != style_sig:
                st.caption("↻ Réglages modifiés depuis cet aperçu — recliquez pour rafraîchir.")
            st.columns([1, 2, 1])[1].video(preview_path)

# Accélération matérielle détectée automatiquement (NVENC / Quick Sync / AMF / CPU).
encoder = "auto"
encoding_speed = "fast"

if st.button("Générer les clips  ✦", use_container_width=True):
    progress_bar = st.progress(0.0)
    status = st.empty()
    live = st.container()
    live_columns = live.columns(3)
    counter = {"n": 0}

    def on_progress(value: float, message: str) -> None:
        progress_bar.progress(min(max(value, 0.0), 1.0))
        status.caption(message)

    def on_clip(path: Path) -> None:
        with live_columns[counter["n"] % 3]:
            render_clip_card(Path(path), key=f"live-{counter['n']}")
        counter["n"] += 1

    try:
        project_dir, clips = process_video(
            url=source["ref"] if source["kind"] == "url" else None,
            uploaded_path=source["path"] if source["kind"] == "file" else None,
            clip_length=clip_length,
            vertical=vertical,
            encoder=encoder,
            export_quality=quality_key,
            encoding_speed=encoding_speed,
            vertical_background=background,
            source_start=window[0],
            source_end=window[1],
            captions_style=captions_style,
            on_clip=on_clip,
            progress=on_progress,
        )
        st.session_state.project_dir = str(project_dir)
        st.session_state.clips = [str(clip) for clip in clips]
        st.session_state["captions_skipped"] = bool(
            captions_style is not None and not _transcript_has_words(str(project_dir))
        )
        st.rerun()
    except Exception as exc:  # noqa: BLE001 - message affiché tel quel
        st.error(f"Le traitement a échoué : {exc}")
