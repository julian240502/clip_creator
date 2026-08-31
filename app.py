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
from src.downloader import download_clip, download_source, probe_url
from src.highlights import find_highlights
from src.llm import ollama_available, pick_model
from src.llm import prewarm as prewarm_llm
from src.paths import SOURCE_CACHE_DIR
from src.pipeline import process_video
from src.quality import ASPECT_USAGE, frame_size, get_quality_preset
from src.reframe import reframe_available
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
BACKGROUND_CHOICES = {
    "Fond vidéo flouté — recommandé": "blur",
    "Bandes noires": "black",
    "Recadrage sur le visage (podcast / interview)": "reframe",
}
FORMAT_CHOICES = {
    "9:16 · Vertical": "9:16",
    "4:5 · Portrait": "4:5",
    "1:1 · Carré": "1:1",
    "16:9 · Paysage": "16:9",
    "Format d'origine": "source",
}


def format_rect_svg(aspect: str, box: int = 60) -> str:
    ratio_w, ratio_h = (16, 9) if aspect == "16:9" else {
        "9:16": (9, 16), "4:5": (4, 5), "1:1": (1, 1),
    }.get(aspect, (9, 16))
    if ratio_w >= ratio_h:
        rw, rh = box, round(box * ratio_h / ratio_w)
    else:
        rw, rh = round(box * ratio_w / ratio_h), box
    return (
        f"<svg width='{rw}' height='{rh}' viewBox='0 0 {rw} {rh}' style='vertical-align:middle'>"
        f"<rect x='2' y='2' width='{rw - 4}' height='{rh - 4}' rx='4' "
        "fill='#7774ff33' stroke='#8b87ff' stroke-width='2'/></svg>"
    )


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
        "highlights", "highlights_model",
    ):
        st.session_state.pop(key, None)
    shutil.rmtree(session_dir(), ignore_errors=True)


def _highlight_thumb(media: str, at: float, out: Path) -> str | None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(max(at, 0.0)),
        "-i", media, "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", str(out),
    ]
    if subprocess.run(command, capture_output=True).returncode == 0 and out.is_file():
        return str(out)
    return None


def analyse_highlights(source: dict, quality_key: str, target_count: int,
                       dur_min: float, dur_max: float) -> tuple[list[dict], str | None]:
    """Télécharge (si URL), transcrit, note les moments et en extrait une vignette."""
    if source["kind"] == "url":
        max_h = get_quality_preset(quality_key).source_max_height
        media = download_source(source["ref"], SOURCE_CACHE_DIR, max_height=max_h)
    else:
        media = source["path"]
    transcript = transcribe(media, cache_dir=session_dir())
    model = pick_model() if ollama_available() else None
    found = find_highlights(
        transcript, target_count=target_count,
        min_duration=float(dur_min), max_duration=float(dur_max), model=model,
    )
    thumbs_dir = session_dir() / "highlights"
    shutil.rmtree(thumbs_dir, ignore_errors=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for index, item in enumerate(found):
        data = asdict(item)
        middle = (item.start + item.end) / 2
        data["thumb"] = _highlight_thumb(media, middle, thumbs_dir / f"hl_{index:02d}.jpg")
        items.append(data)
    return items, model


def timecode(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def render_clip_card(clip: Path, key: str) -> None:
    st.video(str(clip))
    st.caption(clip.name)
    sidecar = clip.with_suffix(".txt")
    if sidecar.is_file():
        with st.expander("Titre & hashtags"):
            st.code(sidecar.read_text(encoding="utf-8"), language=None)
    with clip.open("rb") as handle:
        st.download_button(
            "Télécharger", handle, clip.name, "video/mp4",
            key=key, use_container_width=True,
        )


def _preview_source(source: dict, at: float, seconds: float = 4.0) -> Path:
    """Extrait court et local pour l'aperçu ; réutilisé tant que la portion ne change pas."""
    preview_dir = session_dir() / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    marker = round(at, 2)
    existing = next(preview_dir.glob("preview_source.*"), None)
    if existing is not None and existing.is_file() and st.session_state.get("preview_at") == marker:
        return existing

    st.session_state.pop("preview_at", None)
    for stale in preview_dir.glob("preview_source.*"):
        stale.unlink(missing_ok=True)
    if source["kind"] == "url":
        clip = Path(download_clip(source["ref"], preview_dir, at, at + seconds, max_height=480))
    else:
        clip = preview_dir / "preview_source.mp4"
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(at), "-i", source["path"], "-t", str(seconds),
                "-c:v", "libx264", "-c:a", "aac", str(clip),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Extraction de l'extrait impossible.")
    if not clip.is_file():
        raise RuntimeError(
            "L'extrait d'aperçu n'a pas pu être produit — vérifiez la portion sélectionnée."
        )
    st.session_state["preview_at"] = marker
    return clip


def _transcript_has_words(project_dir: str) -> bool:
    path = Path(project_dir) / "transcript.json"
    if not path.is_file():
        return False
    try:
        return bool(load_transcript(path).words)
    except Exception:  # noqa: BLE001 - transcription illisible = pas de sous-titres
        return False


def render_captions_controls(source: dict, window, aspect: str, background: str):
    """Toggle + panneau de style des sous-titres. Renvoie le CaptionStyle ou None."""
    ready = transcription_available()
    reframed = aspect != "source"
    enabled = st.toggle(
        "Sous-titres incrustés", value=False, disabled=not (ready and reframed),
        key="captions_on",
    )
    if not ready:
        st.caption("Nécessite `pip install -r requirements-transcribe.txt`.")
        return None
    if not reframed:
        st.caption("Les sous-titres ne s'ajoutent qu'à un export recadré (pas au format d'origine).")
        return None
    if not enabled:
        return None

    if not st.session_state.get("model_warming"):
        st.session_state["model_warming"] = True
        threading.Thread(target=prewarm_model, daemon=True).start()
    with st.container(border=True):
        base = TEMPLATES[st.selectbox("Style", list(TEMPLATES))]
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
        style = replace(
            base, font=font, font_size=font_size,
            position=CAPTION_POSITIONS[position_label],
            primary_color=primary_color, highlight_color=highlight_color,
            mode=CAPTION_MODES[mode_label], uppercase=uppercase,
        )
        style_sig = json.dumps(asdict(style), sort_keys=True)
        st.caption(
            "Le modèle se charge en arrière-plan. Le 1er aperçu prend quelques secondes de plus, "
            "les suivants sont quasi instantanés."
        )
        if st.button("Aperçu du style", use_container_width=True):
            with st.spinner("Rendu de l'aperçu…"):
                try:
                    preview = render_style_preview(
                        source, window[0] or 0.0, style, aspect, background,
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
    return style


def render_style_preview(source: dict, at: float, style, aspect: str, background: str) -> Path:
    total = source.get("duration")
    if total:
        at = max(0.0, min(at, total - 4.0)) if total > 4.0 else 0.0
    short = _preview_source(source, at)
    duration = min(4.0, get_video_duration(short))
    frame_w, frame_h = frame_size(PREVIEW_QUALITY, aspect)
    # Même modèle que le rendu final pour que le texte de l'aperçu soit fidèle.
    transcript = transcribe(short, model=DEFAULT_MODEL, cache_dir=session_dir())
    if not transcript.words:
        raise RuntimeError(
            "Aucune parole détectée dans cet extrait — impossible de générer des sous-titres."
        )
    ass = write_clip_captions(
        transcript, session_dir() / "preview" / "preview.ass",
        clip_start=0.0, clip_end=duration,
        width=frame_w, height=frame_h, style=style,
    )
    output = session_dir() / "preview" / "preview.mp4"
    resize_clip_for_vertical(
        short, output, quality=PREVIEW_QUALITY, aspect=aspect, background=background,
        start=0.0, duration=duration, encoding_speed="fast", captions_file=ass,
    )
    return output


def render_diagnostics() -> None:
    """Panneau latéral du mode avancé : détails techniques masqués par défaut."""
    from src.encoder import encoder_label, resolve_video_encoder

    st.markdown("**Diagnostic**")
    llm_model = pick_model() if ollama_available() else None
    whisper = DEFAULT_MODEL if transcription_available() else None
    try:
        video_encoder = encoder_label(resolve_video_encoder("auto"))
    except Exception:  # noqa: BLE001 - aucun encodeur matériel/logiciel utilisable
        video_encoder = "indisponible"
    st.markdown(
        f"- IA locale : {f'`{llm_model}`' if llm_model else '_absente_'}\n"
        f"- Transcription : {f'`{whisper}`' if whisper else '_absente_'}\n"
        f"- Recadrage visage : {'ok' if reframe_available() else '_absent_'}\n"
        f"- Encodeur vidéo : `{video_encoder}`"
    )
    st.caption(f"Session · `{session_dir()}`")
    st.caption(f"Cache sources · `{SOURCE_CACHE_DIR}`")
    last_project = st.session_state.get("project_dir")
    if last_project:
        st.caption(f"Dernier projet · `{last_project}`")


# --- Mode avancé : détails techniques dans la barre latérale -------------------
if "advanced_mode" not in st.session_state:
    st.session_state["advanced_mode"] = (
        st.query_params.get("debug", "").lower() in ("1", "true", "on", "yes")
    )
with st.sidebar:
    ADVANCED = st.toggle(
        "Mode avancé", key="advanced_mode",
        help="Affiche les modèles IA utilisés, l'encodeur vidéo, les chemins de "
             "travail et le transcript brut. À réserver au débogage.",
    )
    if ADVANCED:
        render_diagnostics()


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
                sidecar = clip.with_suffix(".txt")
                if sidecar.is_file():
                    archive.write(sidecar, sidecar.name)
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
quality_key = QUALITY_CHOICES[st.selectbox("Qualité", list(QUALITY_CHOICES))]

format_label = st.radio(
    "Format d'export", list(FORMAT_CHOICES), horizontal=True,
    captions=[
        "Vidéo telle quelle" if key == "source"
        else f"{'×'.join(map(str, frame_size(quality_key, key)))} · {ASPECT_USAGE[key]}"
        for key in FORMAT_CHOICES.values()
    ],
)
export_format = FORMAT_CHOICES[format_label]
vertical = export_format != "source"

background = "blur"
if vertical:
    frame_w, frame_h = frame_size(quality_key, export_format)
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:.7rem;margin:.1rem 0 .5rem'>"
        f"{format_rect_svg(export_format)}"
        f"<span style='color:#bcbfd0;font-size:.82rem'>Cadre {frame_w} × {frame_h} px "
        f"— la vidéo entière tient dedans, le reste est comblé.</span></div>",
        unsafe_allow_html=True,
    )
    background = BACKGROUND_CHOICES[st.selectbox("Arrière-plan", list(BACKGROUND_CHOICES))]
    if background == "reframe":
        if reframe_available():
            st.caption(
                "Le cadre suit le visage principal (podcasts, interviews). "
                "Ajoute une passe d'analyse des visages avant le rendu."
            )
        else:
            st.warning(
                "Recadrage visage indisponible : `pip install -r requirements-reframe.txt`. "
                "En attendant, un recadrage centré sera appliqué."
            )

smart = st.radio(
    "Découpage", ["Régulier", "Sélection intelligente"], horizontal=True,
) == "Sélection intelligente"

clip_length = 30
window = (0.0, None)
if not smart:
    st.session_state.pop("highlights", None)
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
        st.caption("Durée inconnue : toute la vidéo sera traitée.")
elif not transcription_available():
    st.warning("Nécessite `pip install -r requirements-transcribe.txt`.")
else:
    rater = pick_model() if ollama_available() else None
    if not st.session_state.get("smart_warming"):
        st.session_state["smart_warming"] = True
        threading.Thread(target=prewarm_model, daemon=True).start()
        if rater:
            threading.Thread(target=prewarm_llm, daemon=True).start()
    col_n, col_d = st.columns(2)
    target_count = col_n.slider("Nombre de clips visés", 3, 15, 8)
    dur_max = col_d.slider("Durée max d'un clip (s)", 20, 120, 60, 5)
    dur_min = max(12.0, round(dur_max * 0.4))
    st.caption(
        f"Extraits de ~{int(dur_min)} à {dur_max} s. "
        + (
            "Notés par l'IA locale." if rater
            else "IA locale indisponible — notation basique."
        )
        + (f" · modèle `{rater}`" if ADVANCED and rater else "")
    )
    if st.button("Analyser les moments", use_container_width=True):
        with st.spinner("Analyse : téléchargement, transcription, notation…"):
            try:
                found, used = analyse_highlights(
                    source, quality_key, target_count, dur_min, dur_max,
                )
                st.session_state["highlights"] = found
                st.session_state["highlights_model"] = used
                if not found:
                    st.warning("Aucun moment exploitable détecté (pas de parole ?).")
            except Exception as exc:  # noqa: BLE001 - message affiché tel quel
                st.session_state.pop("highlights", None)
                st.error(f"Analyse impossible : {exc}")

# Accélération matérielle détectée automatiquement (NVENC / Quick Sync / AMF / CPU).
encoder = "auto"
encoding_speed = "fast"

# En mode régulier, les sous-titres se règlent ici ; en mode intelligent, ils
# apparaissent après la liste des moments (juste avant « Générer »).
captions_style = None
if not smart:
    captions_style = render_captions_controls(source, window, export_format, background)

# --- Phase 2.5 : choisir les moments (mode intelligent) -----------------------
clips_windows: list[tuple[float, float]] | None = None
clips_hints: list[tuple[str, str]] | None = None
gen_label = "Générer les clips  ✦"
gen_disabled = False
if smart:
    highlights = st.session_state.get("highlights")
    if not highlights:
        gen_disabled = True
        st.caption("Lance **Analyser les moments** pour voir les extraits proposés.")
    else:
        model_used = st.session_state.get("highlights_model")
        st.subheader(f"{len(highlights)} moments détectés")
        st.caption(
            ("Notés par l'IA locale." if model_used else "Notation basique (IA locale indisponible).")
            + (f" · modèle `{model_used}`" if ADVANCED and model_used else "")
        )
        picks: list[tuple[float, float]] = []
        pick_hints: list[tuple[str, str]] = []
        for index, item in enumerate(highlights):
            score = int(item["score"])
            tone = "#3ddc84" if score >= 75 else "#ffb020" if score >= 50 else "#ff5a5f"
            with st.container(border=True):
                row = st.columns([0.6, 2, 5, 1.6])
                keep = row[0].checkbox(
                    "sel", value=index < min(3, len(highlights)),
                    key=f"hl-{index}", label_visibility="collapsed",
                )
                if item.get("thumb") and Path(item["thumb"]).is_file():
                    row[1].image(item["thumb"], use_container_width=True)
                row[2].markdown(f"**{item['title']}**")
                row[2].caption(
                    f"{timecode(item['start'])} – {timecode(item['end'])} · "
                    f"{int(item['end'] - item['start'])} s"
                )
                row[3].markdown(
                    "<div style='text-align:right;line-height:1.05'>"
                    f"<span style='font-size:2rem;font-weight:800;color:{tone}'>{score}</span>"
                    "<div style='font-size:.6rem;letter-spacing:.06em;color:#9a9db0;"
                    "text-transform:uppercase'>viralité / 100</div></div>",
                    unsafe_allow_html=True,
                )
                st.write(item["summary"])
                reasons = item.get("reasons") or []
                if reasons:
                    st.caption(" · ".join(reasons))
                if ADVANCED and item.get("transcript"):
                    with st.expander("Transcript brut"):
                        st.write(item["transcript"])
            if keep:
                picks.append((float(item["start"]), float(item["end"])))
                pick_hints.append((str(item["title"]), str(item["summary"])))
        clips_windows = picks
        clips_hints = pick_hints

        st.markdown("#### Finalisation")
        captions_style = render_captions_controls(source, window, export_format, background)
        subtitle_state = "activés" if captions_style else "désactivés"
        st.caption(f"Sous-titres : **{subtitle_state}** · {len(picks)} clip(s) coché(s).")
        gen_label = f"Générer {len(picks)} clip(s)  ✦"
        gen_disabled = not picks

meta_ready = transcription_available()
meta_on = st.toggle(
    "Générer titres & hashtags", value=False, disabled=not meta_ready, key="meta_on",
)
meta_model = pick_model() if (meta_on and ollama_available()) else None
if not meta_ready:
    st.caption("Nécessite `pip install -r requirements-transcribe.txt`.")
elif meta_on:
    st.caption(
        "Un fichier .txt (titre, description, hashtags) par clip — "
        + ("rédigé par l'IA locale." if ollama_available() else "génération basique (IA locale indisponible).")
        + (f" · modèle `{meta_model}`" if ADVANCED and meta_model else "")
    )

if st.button(gen_label, use_container_width=True, disabled=gen_disabled):
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
            export_format=export_format,
            encoder=encoder,
            export_quality=quality_key,
            encoding_speed=encoding_speed,
            vertical_background=background,
            source_start=window[0],
            source_end=window[1],
            clips_windows=clips_windows,
            captions_style=captions_style,
            generate_meta=meta_on,
            meta_model=meta_model,
            clips_hints=clips_hints,
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
