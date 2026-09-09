"""Traduction des sous-titres via Ollama, avec mise en cache.

Sert aux sous-titres dans une autre langue que celle parlée. Le texte traduit
n'a pas de vrai minutage mot à mot (impossible à récupérer : l'audio est dans une
autre langue). Pour rester **calé sur la parole** malgré tout, chaque segment
Whisper est d'abord redécoupé en unités ~phrases via le minutage réel des mots
(`src.highlights._sentence_units`) ; chaque phrase traduite s'affiche ensuite sur
sa propre fenêtre `[premier mot, dernier mot]`, en bloc (voir
`src/captions.py::build_ass`). Sans ça, un bloc unique couvrant plusieurs phrases
débordait sur les silences et les fragments dérivaient.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from src.highlights import _sentence_units
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
    raw = f"{target}|{model}|v2|{'|'.join(texts)}"
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
            # Lot mal formé : on retente en deux moitiés plus petites, plus
            # fiables. Chaque moitié est calée sur sa taille attendue pour ne pas
            # décaler l'autre.
            mid = len(chunk) // 2
            left = _padded(_translate_batch(system, chunk[:mid], model), mid)
            right = _padded(_translate_batch(system, chunk[mid:], model), len(chunk) - mid)
            translated = left + right
        if not translated:
            continue
        for i in range(len(chunk)):
            if i < len(translated) and str(translated[i]).strip():
                out[start + i] = str(translated[i]).strip()

    # Passe finale : les items encore identiques à la source sont retentés **un
    # par un** (une requête à un seul élément ne "perd" quasi jamais). Sautée si
    # plus de la moitié a échoué (Ollama HS : inutile de le marteler).
    stragglers = [
        i for i, (src, got) in enumerate(zip(texts, out, strict=True))
        if got == src and src.strip()
    ]
    if stragglers and len(stragglers) <= len(texts) / 2:
        for i in stragglers:
            one = _translate_batch(system, [texts[i]], model)
            if one and str(one[0]).strip():
                out[i] = str(one[0]).strip()
    return out


def translate_transcript(
    transcript: Transcript,
    target: str,
    model: str | None,
    *,
    cache: bool = True,
    windows: list[tuple[float, float]] | None = None,
) -> Transcript:
    """Transcript où chaque **unité ~phrase** utile est traduite en `target`.

    `windows` : ne traiter que les segments qui chevauchent un clip réellement
    exporté (perf). Chaque phrase traduite est portée sur sa fenêtre temporelle
    réelle `[premier mot, dernier mot]`. Transcript renvoyé sans mots -> affichage
    en bloc par unité (`src/captions.py::build_ass`).

    Inchangé si la cible est déjà la langue parlée, si la langue n'est pas gérée,
    ou si aucun modèle Ollama n'est disponible.
    """
    target = (target or "").lower()
    if not language_supported(target) or not transcript.segments:
        return transcript
    if (transcript.language or "")[:2] == target:
        return transcript

    if windows:
        kept = [
            seg for seg in transcript.segments
            if any(seg.end > w0 and seg.start < w1 for w0, w1 in windows)
        ]
    else:
        kept = list(transcript.segments)
    if not kept:
        return transcript
    units = _sentence_units(replace(transcript, segments=kept))
    if not units:
        return transcript

    originals = [unit.text.strip() for unit in units]
    cache_file = _cache_path(originals, model or "", target)
    translations: list[str] | None = None
    if cache and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, list) and len(cached) == len(units):
            translations = [str(item) for item in cached]

    if translations is None:
        if not model:
            return transcript
        translations = _translate_segments(originals, target, model)
        if cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(translations, ensure_ascii=False), encoding="utf-8")

    segments = [
        TranscriptSegment(start=unit.start, end=unit.end, text=text.strip(), words=[])
        for unit, text in zip(units, translations, strict=True)
        if text.strip()
    ]
    return replace(transcript, segments=segments, language=target)
