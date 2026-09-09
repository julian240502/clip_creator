"""Traduction des sous-titres via Ollama, avec mise en cache.

Sert aux sous-titres dans une autre langue que celle parlée. Le texte traduit
n'a pas de vrai minutage mot à mot (impossible à récupérer : l'audio est dans une
autre langue). Pour rester **calé sur la parole**, chaque segment Whisper est
d'abord redécoupé en unités ~phrases via le minutage réel des mots
(`src.highlights._sentence_units`) ; chaque phrase traduite s'affiche ensuite sur
sa propre fenêtre `[premier mot, dernier mot]`, en bloc.

Avant traduction : on **nettoie** les annotations non parlées connues
(`[Music]`, `(rires)`, `♪` — sans jamais supprimer de vraie parole entre
parenthèses), on **fusionne** les micro-unités collées, et on donne au modèle la
**durée à l'écran** + la **réplique précédente** en contexte. La consigne
privilégie la **fidélité** : garder tous les éléments concrets cités (noms,
chiffres, marques, lieux), ne raccourcir que les hésitations.
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

# Annotations non parlées connues (liste blanche : on ne supprime PAS n'importe
# quel texte entre parenthèses — un aparté du streamer doit être conservé).
_ANNOTATION_RE = re.compile(
    r"[\[(]\s*(?:music|musique|song|applause|applaudissements|cheer(?:s|ing)?|"
    r"laugh(?:s|ing|ter)?|rires?|sighs?|soupirs?|coughs?|toux|gasps?|"
    r"background\s+(?:noise|music)|bruit\s+de\s+fond|crosstalk|inaudible|"
    r"indistinct|silence|blank[_ ]audio|no\s+audio|pas\s+de\s+son)\s*[\])]"
    r"|[♪♫♬🎵🎶]+",
    re.IGNORECASE,
)
_HAS_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)

# Fusion des micro-unités : trou court + résultat qui reste court et bref.
_MERGE_GAP = 0.6
_MERGE_MAX_CHARS = 64
_MERGE_MAX_SEC = 7.0

# Mots d'un seul tenant qui ont leur place seuls à l'écran (réactions, argot) —
# on ne les jette PAS même isolés et courts.
_KEEP_SOLO = {
    "wow", "oh", "ah", "eh", "hey", "yo", "ok", "okay", "yes", "no", "yeah", "yep",
    "nope", "nah", "huh", "what", "damn", "bruh", "sheesh", "omg", "lol", "lmao",
    "gg", "wtf", "bro", "man", "dude", "stop", "go", "run", "wait", "look", "true",
    "facts", "cap", "based", "clip", "clutch", "chat", "ratio", "w", "l",
    "oui", "non", "quoi", "hein", "ouais", "ouah", "putain", "wesh", "mec", "frère",
    "bah", "ben", "grave", "carrément",
}


def _looks_like_fragment(text: str) -> bool:
    """Un mot isolé, court, plat et non ponctué = presque toujours un bout de
    transcription attrapé sur du bruit / une coupure de mot (« saint », « the »…),
    pas une vraie réplique. On garde en revanche « Quoi ?! », « Wow », « Non »…"""
    stripped = text.strip()
    if len(stripped.split()) > 1:
        return False
    if stripped[-1:] in "!?…" or stripped[-2:] in {"?!", "!?"}:
        return False
    core = stripped.strip(" .,;:!?…-–—\"'«»()[]").lower()
    if not core or core in _KEEP_SOLO:
        return False
    return len(core) <= 5


def language_supported(code: str | None) -> bool:
    return bool(code) and code.lower() in _LANG_NAMES


def _clean_unit_text(text: str) -> str:
    """Retire les annotations non parlées **connues**. Une parenthèse contenant de
    la vraie parole est conservée (on enlève juste les crochets). « » si rien de
    lexical ne reste."""
    text = _ANNOTATION_RE.sub(" ", text)
    text = re.sub(r"[\[\]()]", "", text)  # crochets résiduels -> on garde le contenu
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
        "temps et vocabulaire cohérents d'une réplique à l'autre.\n"
        "FIDÉLITÉ AVANT TOUT : traduis chaque réplique intégralement, en gardant "
        "TOUS les éléments concrets (noms de personnes, marques, lieux, chiffres, "
        "titres de jeux, faits précis). Ne raccourcis QUE les hésitations et "
        "répétitions ('euh', 'genre', mots répétés). Une ligne un peu dense vaut "
        "mieux qu'une information perdue. La durée entre parenthèses est indicative.\n"
        "Si un terme d'argot / de jeu / une expression anglaise n'a pas "
        "d'équivalent courant, garde-le tel quel plutôt que d'inventer.\n"
        "Réponds UNIQUEMENT en JSON "
        '{"t": ["<traduction 0>", "<traduction 1>", ...]} — exactement le même '
        "nombre d'éléments, même ordre, sans fusionner ni ajouter de répliques."
    )


def _cache_path(texts: list[str], model: str, target: str) -> Path:
    raw = f"{target}|{model}|v4|{'|'.join(texts)}"
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


def _format_seconds(value: float) -> str:
    total = int(value)
    return f"{total // 60}:{total % 60:02d}"


def _dump_pairs(path: Path, units: list[_Unit], translations: list[str]) -> None:
    """Écrit `[m:ss] VO -> traduction` par unité, pour vérifier la fidélité à l'œil."""
    lines = [
        f"[{_format_seconds(u.start)}] {u.text}\n        -> {t.strip()}"
        for u, t in zip(units, translations, strict=False)
    ]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass


def translate_transcript(
    transcript: Transcript,
    target: str,
    model: str | None,
    *,
    cache: bool = True,
    windows: list[tuple[float, float]] | None = None,
    debug_out: Path | str | None = None,
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
    # Fragments isolés (mot court, plat, non ponctué) survivants à la fusion : on
    # les écarte, ils n'apportent rien et cassent la lecture.
    units = [unit for unit in units if not _looks_like_fragment(unit.text)]
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

    if debug_out is not None:
        _dump_pairs(Path(debug_out), units, translations)

    segments = [
        TranscriptSegment(start=unit.start, end=unit.end, text=text.strip(), words=[])
        for unit, text in zip(units, translations, strict=True)
        if text.strip()
    ]
    return replace(transcript, segments=segments, language=target)
