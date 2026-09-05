"""Traduction des segments d'un transcript via Ollama, avec mise en cache.

Sert aux sous-titres dans une autre langue que celle parlée. Le texte traduit
n'a pas de vrai alignement mot à mot (impossible à récupérer depuis l'audio
source, qui est dans une autre langue) : on affiche donc le segment entier
comme un bloc (façon sous-titres de film), plutôt que de tenter un mode
d'apparition mot par mot / karaoké basé sur un minutage inventé.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from src.paths import TRANSCRIPTIONS_DIR
from src.transcribe import Transcript, TranscriptSegment

_LANG_NAMES = {
    "fr": "français",
    "en": "anglais",
    "zh": "chinois simplifié",
    "es": "espagnol",
    "de": "allemand",
    "pt": "portugais",
    "it": "italien",
    "ja": "japonais",
    "ko": "coréen",
    "ar": "arabe",
}
_BATCH = 12


def language_supported(code: str | None) -> bool:
    return bool(code) and code.lower() in _LANG_NAMES


def _system_prompt(target_name: str) -> str:
    return (
        f"Tu traduis des sous-titres vidéo en {target_name}. On te donne une liste "
        "numérotée de segments. Réponds UNIQUEMENT en JSON "
        '{"t": ["<traduction du segment 0>", "<traduction du segment 1>", ...]} — '
        "exactement le même nombre d'éléments, dans le même ordre. Traduis "
        "naturellement, en phrases courtes, sans fusionner ni ajouter de segments."
    )


def _cache_path(texts: list[str], model: str, target: str) -> Path:
    raw = f"{target}|{model}|{'|'.join(texts)}"
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return Path(TRANSCRIPTIONS_DIR) / f"tr_{target}_{key}.json"


def _translate_batch(system: str, chunk: list[str], model: str) -> list[str] | None:
    """Traduit un lot ; renvoie None si la réponse est inexploitable."""
    from src.llm import chat_json

    body = "\n".join(f"[{i}] {t}" for i, t in enumerate(chunk))
    try:
        data = chat_json(system, body, model=model, timeout=150.0)
        translated = data.get("t") or data.get("translations") or data.get("segments") or []
    except Exception:  # noqa: BLE001 - Ollama absent/en erreur -> VO gardée
        return None
    return translated if isinstance(translated, list) else None


def _padded(result: list[str] | None, n: int) -> list[str]:
    """Aligne `result` sur `n` éléments (comble par des chaînes vides -> VO gardée)."""
    result = list(result or [])[:n]
    return result + [""] * (n - len(result))


def _translate_segments(texts: list[str], target: str, model: str) -> list[str]:
    system = _system_prompt(_LANG_NAMES[target])
    out = list(texts)
    for start in range(0, len(texts), _BATCH):
        chunk = texts[start : start + _BATCH]
        translated = _translate_batch(system, chunk, model)
        if translated is None and len(chunk) > 1:
            # Lot mal formé ou en erreur : on retente en deux moitiés plus petites,
            # plus fiables, plutôt que d'abandonner tout le lot en VO. Chaque moitié
            # est calée sur sa taille attendue pour ne pas décaler l'autre moitié.
            mid = len(chunk) // 2
            left = _padded(_translate_batch(system, chunk[:mid], model), mid)
            right = _padded(_translate_batch(system, chunk[mid:], model), len(chunk) - mid)
            translated = left + right
        if not translated:
            continue
        for i in range(len(chunk)):
            if i < len(translated) and str(translated[i]).strip():
                out[start + i] = str(translated[i]).strip()
    return out


def _rebuilt(transcript: Transcript, target: str, keep: list[int], texts: list[str]) -> Transcript:
    segments = list(transcript.segments)
    for idx, text in zip(keep, texts, strict=True):
        seg = segments[idx]
        # Pas de mots : un bloc par segment, comme des sous-titres de film — voir
        # src/captions.py::build_ass (bascule automatique en mode "lignes" dès
        # qu'une fenêtre n'a pas de minutage mot à mot).
        segments[idx] = TranscriptSegment(start=seg.start, end=seg.end, text=text, words=[])
    return replace(transcript, segments=segments, language=target)


def translate_transcript(
    transcript: Transcript,
    target: str,
    model: str | None,
    *,
    cache: bool = True,
    windows: list[tuple[float, float]] | None = None,
) -> Transcript:
    """Transcript avec les segments utiles traduits en `target` (affichage en bloc).

    `windows` restreint la traduction aux segments qui chevauchent au moins une
    fenêtre `(start, end)` — typiquement les clips réellement exportés — pour ne
    pas traduire (lentement, via l'IA locale) des minutes de transcript jamais
    utilisées. Sans `windows`, tout le transcript est traduit.

    Renvoie le transcript inchangé si la cible est déjà la langue parlée, si la
    langue n'est pas gérée, ou si aucun modèle Ollama n'est disponible.
    """
    target = (target or "").lower()
    if not language_supported(target) or not transcript.segments:
        return transcript
    if (transcript.language or "")[:2] == target:
        return transcript

    if windows:
        keep = [
            i for i, seg in enumerate(transcript.segments)
            if any(seg.end > w0 and seg.start < w1 for w0, w1 in windows)
        ]
    else:
        keep = list(range(len(transcript.segments)))
    if not keep:
        return transcript

    originals = [transcript.segments[i].text.strip() for i in keep]

    cache_file = _cache_path(originals, model or "", target)
    if cache and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, list) and len(cached) == len(keep):
            return _rebuilt(transcript, target, keep, cached)

    if not model:
        return transcript

    texts = _translate_segments(originals, target, model)
    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    return _rebuilt(transcript, target, keep, texts)
