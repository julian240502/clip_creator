from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from src.encoder import (
    available_hardware_encoders,
    cuda_blur_compositing_available,
    cuda_scaling_available,
    encoder_label,
    resolve_video_encoder,
)
from src.pipeline import process_video
from src.quality import get_quality_preset


st.set_page_config(page_title="Clip Creator", page_icon="✦", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif}
.stApp {background:radial-gradient(circle at 15% 0%,#20234b 0,#0d0e18 38%,#090a10 100%);color:#f7f7fb}
.hero {padding:2.4rem 0 1.6rem}.eyebrow {color:#a6a9ff;font-weight:700;letter-spacing:.13em;text-transform:uppercase;font-size:.75rem}
.hero h1 {font-size:4rem;line-height:1;margin:.5rem 0;background:linear-gradient(90deg,#fff,#a9adff);-webkit-background-clip:text;color:transparent}
.hero p {font-size:1.15rem;color:#a9abba;max-width:690px}
[data-testid="stForm"] {background:rgba(25,27,44,.82);border:1px solid #343751;border-radius:22px;padding:1.5rem;box-shadow:0 20px 60px rgba(0,0,0,.25)}
[data-testid="stMetric"] {background:#171924;border:1px solid #292c3d;padding:1rem;border-radius:16px}
.stButton>button,.stDownloadButton>button {border-radius:12px;font-weight:700;border:0;background:linear-gradient(90deg,#7774ff,#a855f7);color:white}
.clip-card {background:#171924;border:1px solid #292c3d;border-radius:16px;padding:1rem;margin:.7rem 0}
</style>
""", unsafe_allow_html=True)

st.markdown("""<section class="hero"><div class="eyebrow">Studio vidéo local</div><h1>Clip Creator ✦</h1><p>Transformez une vidéo longue en clips nets, verticaux et prêts à finaliser dans CapCut ou Camtasia.</p></section>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Réglages d'export")
    clip_length = st.slider("Durée d'un clip", 10, 180, 30, 5, format="%d sec")
    quality_labels = {
        "Full HD · 1080p — recommandé": "1080p",
        "HD · 720p — plus rapide": "720p",
        "4K · 2160p — meilleure qualité": "4k",
    }
    quality_label = st.selectbox("Qualité", list(quality_labels))
    export_quality = quality_labels[quality_label]
    quality = get_quality_preset(export_quality)
    vertical = st.toggle("Exporter en 9:16", value=True)
    if vertical:
        background_options = {
            "Fond vidéo flouté — recommandé": "blur",
            "Bandes noires": "black",
        }
        background_choice = st.selectbox("Arrière-plan vertical", list(background_options))
        vertical_background = background_options[background_choice]
        st.caption("La vidéo paysage nette reste entièrement visible au premier plan.")
    else:
        vertical_background = "blur"
    detected_encoder = resolve_video_encoder("auto")
    automatic_label = f"Automatique · {encoder_label(detected_encoder)}"
    encoder_preferences = {
        "h264_nvenc": "nvidia",
        "h264_qsv": "intel",
        "h264_amf": "amd",
    }
    encoder_options = {automatic_label: "auto"}
    for hardware_encoder in available_hardware_encoders():
        encoder_options[encoder_label(hardware_encoder)] = encoder_preferences[hardware_encoder]
    encoder_options["CPU · x264"] = "cpu"
    encoder_choice = st.selectbox("Accélération", list(encoder_options))
    encoder = encoder_options[encoder_choice]
    speed_options = {
        "Rapide — recommandé": "fast",
        "Équilibrée": "balanced",
        "Qualité maximale": "quality",
    }
    speed_choice = st.selectbox("Vitesse d'encodage", list(speed_options))
    encoding_speed = speed_options[speed_choice]
    st.divider()
    dimensions = f"{quality.width} × {quality.height}" if vertical else f"source ≤ {quality.source_max_height}p"
    active_encoder = resolve_video_encoder(encoder)
    cuda_available = (
        cuda_blur_compositing_available()
        if vertical_background == "blur"
        else cuda_scaling_available()
    )
    cuda_active = vertical and active_encoder == "h264_nvenc" and cuda_available
    cuda_label = " · redimensionnement CUDA" if cuda_active else ""
    st.caption(f"{dimensions} · H.264 · AAC · {encoder_label(active_encoder)}{cuda_label}")
    if export_quality == "4k":
        st.caption("La 4K produit des fichiers plus lourds et demande plus de temps d'encodage.")

with st.form("creator"):
    source_type = st.radio("Source", ["Lien vidéo", "Fichier local"], horizontal=True)
    url = st.text_input("URL de la vidéo", placeholder="https://www.youtube.com/watch?v=…", disabled=source_type != "Lien vidéo")
    uploaded = st.file_uploader("Déposer une vidéo", type=["mp4", "mov", "mkv", "webm"], disabled=source_type != "Fichier local")
    submitted = st.form_submit_button("Créer mes clips  ✦", use_container_width=True)

if submitted:
    if source_type == "Lien vidéo" and not url.strip():
        st.warning("Ajoutez une URL vidéo.")
        st.stop()
    if source_type == "Fichier local" and uploaded is None:
        st.warning("Sélectionnez une vidéo.")
        st.stop()
    progress_bar, status = st.progress(0), st.empty()
    temp_dir = Path(tempfile.mkdtemp(prefix="clip-creator-"))
    try:
        upload_path = None
        if uploaded is not None:
            upload_path = temp_dir / Path(uploaded.name).name
            with upload_path.open("wb") as handle:
                handle.write(uploaded.getbuffer())
        def update_progress(value: float, message: str) -> None:
            progress_bar.progress(value)
            status.caption(message)
        project_dir, clips = process_video(
            url=url.strip() or None, uploaded_path=upload_path, clip_length=clip_length,
            vertical=vertical, encoder=encoder,
            export_quality=export_quality, encoding_speed=encoding_speed,
            vertical_background=vertical_background,
            progress=update_progress,
        )
        st.success(f"{len(clips)} clip(s) prêt(s) pour le montage.")
        archive_path = project_dir / "clip-creator-exports.zip"
        # Les MP4 sont déjà compressés : les stocker évite une seconde passe CPU inutile.
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
            for clip in clips:
                archive.write(clip, clip.name)
        with archive_path.open("rb") as archive:
            st.download_button("Télécharger tous les clips (.zip)", archive, archive_path.name, "application/zip", use_container_width=True)
        st.subheader("Aperçu des exports")
        for clip in clips:
            with st.container():
                st.markdown(f'<div class="clip-card"><b>{clip.name}</b></div>', unsafe_allow_html=True)
                st.video(str(clip))
                with clip.open("rb") as video:
                    st.download_button(f"Télécharger {clip.name}", video, clip.name, "video/mp4", key=str(clip))
    except Exception as exc:
        st.error(f"Le traitement a échoué : {exc}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
