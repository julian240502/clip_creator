"""Traduction des segments d'un transcript via Ollama, avec mise en cache.

Sert aux sous-titres dans une autre langue que celle parlée. Le texte traduit
n'a pas de vrai alignement mot à mot (impossible à récupérer depuis l'audio
source, qui est dans une autre langue) : chaque segment traduit reçoit des
timings **synthétiques**, répartis proportionnellement à la longueur des mots
sur la durée du segment. Approximatif mais suffisant pour que les modes
d'apparition (mot actif, karaoké, mot par mot) restent utilisables.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from src.paths import TRANSCRIPTIONS_DIR
from src.transcribe import Transcript, TranscriptSegment, Word

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


def _cache_path(transcript: Transcript, target: str) -> Path:
    raw = f"{target}|{transcript.model}|{transcript.text}"
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return Path(TRANSCRIPTIONS_DIR) / f"tr_{target}_{key}.json"


def _translate_segments(texts: list[str], target: str, model: str) -> list[str]:
    from src.llm import chat_json

    system = _system_prompt(_LANG_NAMES[target])
    out = list(texts)
    for start in range(0, len(texts), _BATCH):
        chunk = texts[start : start + _BATCH]
        body = "\n".join(f"[{i}] {t}" for i, t in enumerate(chunk))
        try:
            data = chat_json(system, body, model=model, timeout=150.0)
            translated = data.get("t") or data.get("translations") or data.get("segments") or []
        except Exception:  # noqa: BLE001 - Ollama absent/incohérent -> segment gardé en VO
            continue
        for i in range(len(chunk)):
            if i < len(translated) and str(translated[i]).strip():
                out[start + i] = str(translated[i]).strip()
    return out


def _synthetic_words(text: str, start: float, end: float, target: str) -> list[Word]:
    """Répartit `text` en « mots » horodatés sur [start, end], proportionnellement
    à leur longueur. Le chinois n'a pas d'espaces : découpage au caractère.
    """
    tokens = [t for t in (list(text) if target == "zh" else text.split()) if t.strip()]
    if not tokens:
        return []
    duration = max(0.05, end - start)
    weights = [max(1, len(t)) for t in tokens]
    # Plancher par mot pour éviter un flash trop court, sans dépasser la durée totale.
    floor = min(0.12, duration / (len(tokens) * 2))
    remaining = duration - floor * len(tokens)
    total_weight = sum(weights)
    words: list[Word] = []
    cursor = start
    for token, weight in zip(tokens, weights, strict=True):
        span = floor + remaining * (weight / total_weight)
        w_end = min(end, cursor + span)
        words.append(Word(start=cursor, end=w_end, text=token))
        cursor = w_end
    words[-1] = Word(words[-1].start, end, words[-1].text)  # colle pile à la fin du segment
    return words


def _rebuilt(transcript: Transcript, target: str, texts: list[str]) -> Transcript:
    segments = [
        TranscriptSegment(
            start=seg.start, end=seg.end, text=text,
            words=_synthetic_words(text, seg.start, seg.end, target),
        )
        for seg, text in zip(transcript.segments, texts, strict=False)
    ]
    return replace(transcript, segments=segments, language=target)


def translate_transcript(
    transcript: Transcript, target: str, model: str | None, *, cache: bool = True,
) -> Transcript:
    """Transcript avec chaque segment traduit en `target` (mots synthétiques, voir module).

    Renvoie le transcript inchangé si la cible est déjà la langue parlée, si la
    langue n'est pas gérée, ou si aucun modèle Ollama n'est disponible.
    """
    target = (target or "").lower()
    if not language_supported(target) or not transcript.segments:
        return transcript
    if (transcript.language or "")[:2] == target:
        return transcript

    cache_file = _cache_path(transcript, target)
    if cache and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, list) and len(cached) == len(transcript.segments):
            return _rebuilt(transcript, target, cached)

    if not model:
        return transcript

    texts = _translate_segments([s.text.strip() for s in transcript.segments], target, model)
    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    return _rebuilt(transcript, target, texts)
