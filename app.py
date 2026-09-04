from __future__ import annotations

import json
import math
import os
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
    font_for_language,
    write_clip_captions,
)
from src.downloader import download_clip, download_source, probe_url
from src.highlights import HOOK_STRONG, find_highlights
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

/* Barre latérale (mode avancé) accordée au fond sombre */
[data-testid="stSidebar"] {background:#12131c;border-right:1px solid #262838}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {color:#e2e3f0}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#9a9db0}
[data-testid="stSidebar"] code {color:#c9cbff;background:#20223a}
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stSidebarCollapseButton"] {color:#e2e3f0;fill:#e2e3f0}
</style>
""",
    unsafe_allow_html=True,
)

WORK_ROOT = Path(tempfile.gettempdir()) / "clip-creator"
# Pré-rempli dans « Copier les clips dans un dossier » — le dossier Google Drive
# synchronisé de l'owner. Modifiable dans l'UI, mémorisé pour la session.
DEFAULT_EXPORT_DIR = r"G:\Mon Drive\CLIPS"
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


def _forget_send_selection() -> None:
    for key in [k for k in st.session_state if k.startswith("send-clip-")]:
        st.session_state.pop(key, None)
    st.session_state.pop("send-all", None)
    st.session_state.pop("sent_clips", None)


def reset_source() -> None:
    for key in (
        "source", "clips", "project_dir", "captions_skipped",
        "style_preview", "style_preview_sig", "preview_at",
        "highlights", "highlights_model", "source_lang", "export_label",
    ):
        st.session_state.pop(key, None)
    _forget_send_selection()
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
                       dur_min: float, dur_max: float) -> tuple[list[dict], str | None, str]:
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
    return items, model, transcript.language


def timecode(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def render_clip_card(clip: Path, key: str, *, selectable: bool = False) -> None:
    st.video(str(clip))
    st.caption(clip.name)
    if selectable:
        sent = clip.name in st.session_state.get("sent_clips", set())
        st.checkbox(
            "Envoyer vers le dossier" + (" · ✓ déjà copié" if sent else ""),
            key=f"send-{key}",
        )
    sidecar = clip.with_suffix(".txt")
    if sidecar.is_file():
        with st.expander("Titre & hashtags"):
            st.code(sidecar.read_text(encoding="utf-8"), language=None)
    with clip.open("rb") as handle:
        st.download_button(
            "Télécharger", handle, clip.name, "video/mp4",
            key=key, use_container_width=True,
        )


def _preview_source(source: dict, at: float, seconds: float = 4.0, max_height: int = 480) -> Path:
    """Extrait court et local pour l'aperçu ; réutilisé tant que la portion ne change pas."""
    preview_dir = session_dir() / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    marker = (round(at, 2), max_height)
    existing = next(preview_dir.glob("preview_source.*"), None)
    if existing is not None and existing.is_file() and st.session_state.get("preview_at") == marker:
        return existing

    st.session_state.pop("preview_at", None)
    for stale in preview_dir.glob("preview_source.*"):
        stale.unlink(missing_ok=True)
    if source["kind"] == "url":
        clip = Path(download_clip(source["ref"], preview_dir, at, at + seconds, max_height=max_height))
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


def _friendly_session(project_name: str) -> str:
    """`20260903-134700-…` → `2026-09-03 13h47` (lisible pour le dossier d'export)."""
    parts = project_name.split("-")
    if len(parts) >= 2 and len(parts[0]) == 8 and parts[0].isdigit() and len(parts[1]) >= 4:
        d, t = parts[0], parts[1]
        return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}h{t[2:4]}"
    return project_name


def _export_lang(project_dir: str | None) -> str:
    """Code langue (majuscules) du dossier d'export : sous-titres choisis, sinon langue parlée."""
    chosen = st.session_state.get("caption_lang")
    if chosen:
        return chosen.upper()
    if project_dir and (Path(project_dir) / "transcript.json").is_file():
        try:
            code = (load_transcript(Path(project_dir) / "transcript.json").language or "")[:2]
            if code:
                return code.upper()
        except Exception:  # noqa: BLE001 - transcript illisible
            pass
    return "XX"


def render_captions_controls(source: dict, window, aspect: str, background: str, quality_key: str):
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
    caption_langs = {"Auto (langue parlée)": None, "Français": "fr", "English": "en",
                     "中文 (chinois)": "zh", "한국어 (coréen)": "ko"}
    with st.container(border=True):
        caption_lang = caption_langs[st.selectbox(
            "Langue des sous-titres", list(caption_langs), key="caption_lang_label",
        )]
        st.session_state["caption_lang"] = caption_lang
        if caption_lang:
            from src.llm import language_name

            source_lang = (st.session_state.get("source_lang") or "")[:2]
            forced_font = font_for_language(caption_lang)
            translation_note = (
                " Le calage du texte traduit sur les mots est **approximatif** (réparti "
                "sur la durée de la phrase, pas un vrai alignement audio)."
                + (f" Police `{forced_font}` utilisée pour la traduction." if forced_font else "")
            )
            if source_lang and source_lang == caption_lang:
                # Vidéo déjà dans la langue visée : pas de traduction, l'animation
                # choisie ci-dessous s'applique normalement.
                st.caption(f"La vidéo est déjà en {language_name(source_lang)} — pas de traduction.")
            elif source_lang:
                # Langue source connue (mode intelligent, après « Analyser les moments »)
                # et différente : la traduction aura lieu à coup sûr.
                st.caption(
                    f"La vidéo est en {language_name(source_lang)} → traduite en "
                    f"{language_name(caption_lang)} par l'IA locale." + translation_note
                )
            else:
                # Langue source inconnue (mode régulier / pas encore analysé) : une
                # traduction *pourrait* avoir lieu, le rendu décide une fois la vraie
                # langue connue (à la transcription).
                st.caption(
                    "Si la vidéo n'est pas déjà dans cette langue, elle sera traduite "
                    "par l'IA locale." + translation_note
                )
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
            index=list(CAPTION_MODES).index("Mot par mot"),  # défaut : mot par mot
            key="caption_mode",
        )
        col_x, col_y = st.columns(2)
        nudge_x = col_x.slider("Décalage horizontal (px)", -300, 300, base.nudge_x, 5,
                               help="+ vers la droite")
        nudge_y = col_y.slider("Décalage vertical (px)", -400, 400, base.nudge_y, 5,
                               help="+ vers le haut")
        uppercase = st.toggle("MAJUSCULES", value=base.uppercase)
        style = replace(
            base, font=font, font_size=font_size,
            position=CAPTION_POSITIONS[position_label],
            primary_color=primary_color, highlight_color=highlight_color,
            mode=CAPTION_MODES[mode_label], uppercase=uppercase,
            nudge_x=nudge_x, nudge_y=nudge_y,
        )
        # Le mode "lignes" + la police adaptée ne sont forcés que si une traduction a
        # réellement lieu (langue source ≠ langue visée) — décidé au rendu, avec la
        # vraie langue détectée : voir render_style_preview() et process_video().
        style_sig = json.dumps(asdict(style), sort_keys=True)
        st.caption(
            "Le modèle se charge en arrière-plan. Le 1er aperçu prend quelques secondes de plus, "
            "les suivants sont quasi instantanés."
        )
        if background == "reframe":
            st.caption("L'aperçu utilise un recadrage centré ; le rendu final suivra le visage.")
        if st.button("Aperçu du style", use_container_width=True):
            with st.spinner("Rendu de l'aperçu…"):
                try:
                    highlights = st.session_state.get("highlights")
                    anchor = float(highlights[0]["start"]) if highlights else (window[0] or 0.0)
                    preview = render_style_preview(
                        source, anchor, style, aspect, background, quality_key, caption_lang,
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


def render_style_preview(
    source: dict, at: float, style, aspect: str, background: str, quality_key: str,
    caption_lang: str | None = None,
) -> Path:
    total = source.get("duration")
    # Le recadrage découpe une tranche verticale : il faut une source assez nette
    # pour ne pas l'agrandir. Le flou / les bandes tolèrent un extrait plus léger.
    max_h = 720 if background == "reframe" else 480
    # Cherche un extrait de ~4 s qui contient vraiment de la parole (glisse en avant).
    short = None
    transcript = None
    for step in range(5):
        probe = at + step * 8.0
        if total:
            probe = max(0.0, min(probe, total - 4.0)) if total > 4.0 else 0.0
        candidate = _preview_source(source, probe, max_height=max_h)
        found = transcribe(candidate, model=DEFAULT_MODEL, cache_dir=session_dir())
        if found.words:
            short, transcript = candidate, found
            break
        if total and probe >= total - 4.0:
            break
    if transcript is None or not transcript.words:
        raise RuntimeError(
            "Aucune parole trouvée près de ce point — choisis une autre portion de la vidéo."
        )
    from src.translate import language_supported

    if caption_lang and language_supported(caption_lang) and (transcript.language or "")[:2] != caption_lang:
        from src.translate import translate_transcript

        model = pick_model() if ollama_available() else None
        if model is None:
            raise RuntimeError("Traduction impossible : IA locale (Ollama) indisponible.")
        transcript = translate_transcript(transcript, caption_lang, model)
        # Texte traduit : glyphes adaptés (CJK/Hangul…) si la police choisie n'en a pas.
        forced_font = font_for_language(caption_lang)
        if forced_font:
            style = replace(style, font=forced_font)
    duration = min(4.0, get_video_duration(short))
    frame_w, frame_h = frame_size(PREVIEW_QUALITY, aspect)          # taille réelle de l'aperçu
    ass_w, ass_h = frame_size(quality_key, aspect)                  # ASS calé sur la réso finale
    ass = write_clip_captions(
        transcript, session_dir() / "preview" / "preview.ass",
        clip_start=0.0, clip_end=duration,
        width=ass_w, height=ass_h, style=style,
    )
    crop_cmd = None
    if background == "reframe":
        # L'aperçu sert à juger le style des sous-titres : recadrage centré, sans
        # passe de détection (le rendu final, lui, suit le visage).
        from src.reframe import centred_crop_path, crop_box, write_sendcmd

        src_w, src_h = get_video_resolution(short)
        crop_w, crop_h = crop_box(src_w, src_h, frame_w, frame_h)
        crop_cmd = write_sendcmd(
            session_dir() / "preview" / "preview.cmd",
            centred_crop_path(src_w, src_h, crop_w, crop_h),
        )
    output = session_dir() / "preview" / "preview.mp4"
    resize_clip_for_vertical(
        short, output, quality=PREVIEW_QUALITY, aspect=aspect, background=background,
        start=0.0, duration=duration, encoding_speed="fast", captions_file=ass,
        crop_cmd_file=crop_cmd,
    )
    return output


def render_diagnostics() -> None:
    """Panneau latéral du mode avancé : détails techniques masqués par défaut."""
    from src.encoder import cuda_scaling_available, encoder_label, resolve_video_encoder
    from src.resizer import _cuda_blur_enabled

    st.markdown("**Diagnostic**")
    llm_model = pick_model() if ollama_available() else None
    whisper = DEFAULT_MODEL if transcription_available() else None
    try:
        video_encoder = encoder_label(resolve_video_encoder("auto"))
    except Exception:  # noqa: BLE001 - aucun encodeur matériel/logiciel utilisable
        video_encoder = "indisponible"
    if _cuda_blur_enabled():
        blur_mode = "CUDA (CLIP_CREATOR_CUDA_BLUR=1)" if cuda_scaling_available() else "logiciel (CUDA indisponible)"
    else:
        blur_mode = "logiciel"
    cookies = (
        os.environ.get("CLIP_CREATOR_YTDLP_COOKIES_BROWSER")
        or os.environ.get("CLIP_CREATOR_YTDLP_COOKIES_FILE")
    )
    st.markdown(
        f"- IA locale : {f'`{llm_model}`' if llm_model else '_absente_'}\n"
        f"- Transcription : {f'`{whisper}`' if whisper else '_absente_'}\n"
        f"- Recadrage visage : {'ok' if reframe_available() else '_absent_'}\n"
        f"- Encodeur vidéo : `{video_encoder}`\n"
        f"- Fond flou : {blur_mode}\n"
        f"- Cookies YouTube : {f'`{cookies}`' if cookies else '_non configurés_'}"
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
                        msg = str(exc).lower()
                        if "could not copy" in msg or "cookie database" in msg or "7271" in msg:
                            st.info(
                                "Chrome/Edge chiffrent leur base de cookies (Windows) — yt-dlp "
                                "ne peut pas la lire. Exporte plutôt un fichier `cookies.txt` "
                                "depuis YouTube (extension « Get cookies.txt LOCALLY »), puis "
                                "avant de relancer l'app dans **le même terminal** :\n\n"
                                "`$env:CLIP_CREATOR_YTDLP_COOKIES_FILE=\"C:\\chemin\\cookies.txt\"`\n\n"
                                "Alternative : `$env:CLIP_CREATOR_YTDLP_COOKIES_BROWSER=\"firefox\"` "
                                "(Firefox n'a pas ce blocage)."
                            )
                        elif "sign in to confirm" in msg or "cookies" in msg or "bot" in msg:
                            st.info(
                                "YouTube demande une authentification. Dans le terminal qui lance "
                                "l'app, puis relance-la :\n\n"
                                "`$env:CLIP_CREATOR_YTDLP_COOKIES_BROWSER=\"firefox\"`  "
                                "(ou `chrome`, `edge`, `brave`)\n\n"
                                "ou un fichier exporté : "
                                "`$env:CLIP_CREATOR_YTDLP_COOKIES_FILE=\"C:\\chemin\\cookies.txt\"`."
                            )
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

    export_dir = st.session_state.get("export_dir", "").strip()
    export_label = st.session_state.get("export_label", "").strip()

    columns = st.columns(3)
    for index, clip in enumerate(clips):
        with columns[index % 3]:
            render_clip_card(clip, key=f"clip-{index}", selectable=bool(export_dir))

    if export_dir:
        n = len(clips)
        for i in range(n):
            st.session_state.setdefault(f"send-clip-{i}", False)

        def _toggle_all_send() -> None:
            for i in range(len(st.session_state.get("clips") or [])):
                st.session_state[f"send-clip-{i}"] = st.session_state["send-all"]

        picked = [i for i in range(n) if st.session_state.get(f"send-clip-{i}")]
        st.session_state["send-all"] = len(picked) == n
        lang = _export_lang(project_dir)
        friendly_date = _friendly_session(Path(project_dir).name) if project_dir else "clips"
        with st.container(border=True):
            st.checkbox(
                f"Tout cocher ({len(picked)}/{n})", key="send-all", on_change=_toggle_all_send,
            )
            st.caption(
                f"Destination : `{export_dir}\\{lang}\\{export_label or 'Clips'}\\{friendly_date}\\`"
            )
            if st.button(
                f"Envoyer {len(picked)} clip(s) vers le dossier",
                use_container_width=True, disabled=not picked,
            ):
                from src.pipeline import _publish_to_folder

                try:
                    _publish_to_folder(
                        [clips[i] for i in picked], export_dir, lang, export_label, friendly_date,
                    )
                    sent = st.session_state.setdefault("sent_clips", set())
                    sent.update(clips[i].name for i in picked)
                    st.success(f"{len(picked)} clip(s) copié(s) dans {export_dir}.")
                except Exception as exc:  # noqa: BLE001 - message affiché tel quel
                    st.error(f"Copie impossible : {exc}")
    else:
        st.caption(
            "Renseigne un « Dossier de destination » à l'étape des réglages pour "
            "pouvoir envoyer une sélection de clips (Google Drive…)."
        )

    left, right = st.columns(2)
    if left.button("Régler à nouveau", use_container_width=True):
        for key in ("clips", "project_dir", "captions_skipped"):
            st.session_state.pop(key, None)
        _forget_send_selection()
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
                found, used, src_lang = analyse_highlights(
                    source, quality_key, target_count, dur_min, dur_max,
                )
                st.session_state["highlights"] = found
                st.session_state["highlights_model"] = used
                st.session_state["source_lang"] = src_lang
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
    captions_style = render_captions_controls(source, window, export_format, background, quality_key)

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
        n_hooked = sum(1 for h in highlights if int(h.get("hook_score", 0)) >= HOOK_STRONG)
        st.subheader(f"{len(highlights)} moments détectés")
        st.caption(
            ("Notés par l'IA locale." if model_used else "Notation basique (IA locale indisponible).")
            + (f" · {n_hooked} avec une accroche forte ⚡" if n_hooked else "")
            + (f" · modèle `{model_used}`" if ADVANCED and model_used else "")
        )
        n_highlights = len(highlights)
        for i in range(n_highlights):
            st.session_state.setdefault(f"hl-{i}", i < min(3, n_highlights))

        def _toggle_all_highlights() -> None:
            target = st.session_state["hl-all"]
            for i in range(len(st.session_state.get("highlights") or [])):
                st.session_state[f"hl-{i}"] = target

        n_selected = sum(st.session_state[f"hl-{i}"] for i in range(n_highlights))
        st.session_state["hl-all"] = n_selected == n_highlights
        st.checkbox(
            f"Tout sélectionner ({n_selected}/{n_highlights})",
            key="hl-all", on_change=_toggle_all_highlights,
        )

        picks: list[tuple[float, float]] = []
        pick_hints: list[tuple[str, str]] = []
        for index, item in enumerate(highlights):
            score = int(item["score"])
            tone = "#3ddc84" if score >= 75 else "#ffb020" if score >= 50 else "#ff5a5f"
            with st.container(border=True):
                row = st.columns([0.6, 2, 5, 1.6])
                keep = row[0].checkbox(
                    "sel", key=f"hl-{index}", label_visibility="collapsed",
                )
                if item.get("thumb") and Path(item["thumb"]).is_file():
                    row[1].image(item["thumb"], use_container_width=True)
                hook_score = int(item.get("hook_score", 0))
                strong_hook = hook_score >= HOOK_STRONG
                title_html = f"**{item['title']}**"
                if strong_hook:
                    title_html += (
                        " <span style='background:#123a2a;color:#3ddc84;border:1px solid #1f6b4a;"
                        "border-radius:999px;padding:.05rem .45rem;font-size:.68rem;"
                        "font-weight:700;white-space:nowrap'>⚡ Accroche forte</span>"
                    )
                if ADVANCED:
                    title_html += (
                        f" <span style='color:#9a9db0;font-size:.68rem'>hook {hook_score}/100</span>"
                    )
                row[2].markdown(title_html, unsafe_allow_html=True)
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
                if item.get("hook_line"):
                    st.caption(f"Accroche · « {item['hook_line']} »")
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
        captions_style = render_captions_controls(source, window, export_format, background, quality_key)
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

st.session_state.setdefault("export_dir", DEFAULT_EXPORT_DIR)
st.session_state.setdefault(
    "export_label", (source.get("uploader") or source.get("title") or "").strip(),
)
with st.expander("Dossier d'envoi des clips (Google Drive…)"):
    export_dir = st.text_input("Dossier de destination", key="export_dir").strip()
    export_label = st.text_input(
        "Créateur / streamer (nom du sous-dossier)", key="export_label",
    ).strip()
    if export_dir:
        base = f"{export_dir}\\{export_label or 'Clips'}\\<date>"
        st.caption(
            f"Après génération, tu **choisis** les clips à envoyer dans "
            f"`{base}\\clips\\` (leur `.txt` dans `{base}\\textes\\`). "
            "Rien n'est copié automatiquement — tu coches ce que tu valides. "
            "Pointe vers ton Google Drive synchronisé pour retrouver la sélection sur le téléphone."
        )
    else:
        st.caption(
            "Vide = aucun envoi possible. Les clips restent téléchargeables "
            "(individuel / ZIP) après génération."
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
            caption_lang=st.session_state.get("caption_lang"),
            generate_meta=meta_on,
            meta_model=meta_model,
            clips_hints=clips_hints,
            video_title=source.get("title"),
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
