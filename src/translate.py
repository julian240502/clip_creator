"""Traduction des sous-titres via Ollama, avec mise en cache.

Sert aux sous-titres dans une autre langue que celle parlée. Le texte traduit
n'a pas de vrai minutage mot à mot (impossible à récupérer : l'audio est dans une
autre langue). Pour rester **calé sur la parole**, chaque segment Whisper est
d'abord redécoupé en unités ~phrases via le minutage réel des mots
(`src.highlights._sentence_units`) ; chaque phrase traduite s'affiche ensuite sur
sa propre fenêtre `[premier mot, dernier mot]`, en bloc.

Avant traduction : on **nettoie** les annotations non parlées (`[Music]`, `(rires)`,
`♪`), on **fusionne** les micro-unités collées (« Ouais. » + phrase suivante), et
on donne au modèle la **durée à l'écran** de chaque réplique (pour qu'il condense
si besoin) ainsi que la **réplique précédente** en contexte (cohérence des
pronoms / temps).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from src.highlights import _sentence_units, _Unit
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

# Annotations non parlées émises par Whisper.
_ANNOTATION_RE = re.compile(r"[\[(][^\])]*[\])]|[♪♫♬🎵🎶]+|\bBLANK_AUDIO\b", re.IGNORECASE)
_HAS_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)

# Fusion des micro-unités : trou court + résultat qui reste court et bref.
_MERGE_GAP = 0.6
_MERGE_MAX_CHARS = 64
_MERGE_MAX_SEC = 7.0


def language_supported(code: str | None) -> bool:
    return bool(code) and code.lower() in _LANG_NAMES


def _clean_unit_text(text: str) -> str:
    """Retire les annotations non parlées ; renvoie « » si rien de lexical ne reste."""
    text = _ANNOTATION_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—*·_~")
    return text if _HAS_LETTER_RE.search(text) else ""


def _merge_short_units(units: list[_Unit]) -> list[_Unit]:
    """Recolle une unité à la précédente quand elles sont collées dans le temps et
    que le résultat reste court (évite les « Ouais. » qui flashent seuls)."""
    if not units:
        return units
    merged = [units[0]]
    for unit in units[1:]:
        prev = merged[-1]
        joined = len(prev.text) + 1 + len(unit.text)
        if (
            unit.start - prev.end <= _MERGE_GAP
            and joined <= _MERGE_MAX_CHARS
            and unit.end - prev.start <= _MERGE_MAX_SEC
        ):
            merged[-1] = _Unit(prev.start, unit.end, f"{prev.text} {unit.text}")
        else:
            merged.append(unit)
    return merged


def _system_prompt(target_name: str) -> str:
    return (
        f"Tu traduis des sous-titres vidéo en {target_name}. On te donne une liste "
        "numérotée de répliques successives d'un même dialogue : garde pronoms, "
        "temps et vocabulaire cohérents d'une réplique à l'autre. Chaque réplique "
        "indique entre parenthèses sa durée à l'écran ; si la traduction ne s'y lit "
        "pas confortablement (~15 caractères par seconde), condense-la sans perdre "
        "le sens. Réponds UNIQUEMENT en JSON "
        '{"t": ["<traduction 0>", "<traduction 1>", ...]} — exactement le même '
        "nombre d'éléments, même ordre, sans fusionner ni ajouter de répliques."
    )


def _cache_path(texts: list[str], model: str, target: str) -> Path:
    raw = f"{target}|{model}|v3|{'|'.join(texts)}"
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return Path(TRANSCRIPTIONS_DIR) / f"tr_{target}_{key}.json"


def _translate_batch(
    system: str, texts: list[str], durations: list[float], model: str, context: str = "",
) -> list[str] | None:
    """Traduit un lot ; renvoie None si la réponse est inexploitable."""
    from src.llm import chat_json

    lines = []
    if context:
        lines.append(f"Contexte déjà dit (ne pas traduire) : {context}")
    lines += [
        f"[{i}] ({d:.1f}s) {t}"
        for i, (t, d) in enumerate(zip(texts, durations, strict=True))
    ]
    try:
        data = chat_json(system, "\n".join(lines), model=model, timeout=150.0)
        translated = data.get("t") or data.get("translations") or data.get("segments") or []
    except Exception:  # noqa: BLE001 - Ollama absent/en erreur -> VO gardée
        return None
    return translated if isinstance(translated, list) else None


def _padded(result: list[str] | None, n: int) -> list[str]:
    """Aligne `result` sur `n` éléments (comble par des chaînes vides -> VO gardée)."""
    result = list(result or [])[:n]
    return result + [""] * (n - len(result))


def _translate_segments(
    texts: list[str], durations: list[float], target: str, model: str,
) -> list[str]:
    system = _system_prompt(_LANG_NAMES[target])
    out = list(texts)
    for start in range(0, len(texts), _BATCH):
        ct, cd = texts[start : start + _BATCH], durations[start : start + _BATCH]
        context = texts[start - 1] if start > 0 else ""
        translated = _translate_batch(system, ct, cd, model, context)
        if translated is None and len(ct) > 1:
            # Lot mal formé : on retente en deux moitiés plus petites, plus
            # fiables. Chaque moitié est calée sur sa taille attendue.
            mid = len(ct) // 2
            left = _padded(_translate_batch(system, ct[:mid], cd[:mid], model), mid)
            right = _padded(_translate_batch(system, ct[mid:], cd[mid:], model), len(ct) - mid)
            translated = left + right
        if not translated:
            continue
        for i in range(len(ct)):
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
            one = _translate_batch(system, [texts[i]], [durations[i]], model)
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

    units: list[_Unit] = []
    for unit in _sentence_units(replace(transcript, segments=kept)):
        cleaned = _clean_unit_text(unit.text)
        if cleaned:
            units.append(_Unit(unit.start, unit.end, cleaned))
    units = _merge_short_units(units)
    if not units:
        return transcript

    originals = [unit.text for unit in units]
    durations = [max(0.1, unit.end - unit.start) for unit in units]
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
        translations = _translate_segments(originals, durations, target, model)
        if cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(translations, ensure_ascii=False), encoding="utf-8")

    segments = [
        TranscriptSegment(start=unit.start, end=unit.end, text=text.strip(), words=[])
        for unit, text in zip(units, translations, strict=True)
        if text.strip()
    ]
    return replace(transcript, segments=segments, language=target)
