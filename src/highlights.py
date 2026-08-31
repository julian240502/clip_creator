"""Sélection intelligente : repère les extraits au plus fort potentiel viral.

Pipeline : reconstruction d'unités ~phrases à partir du flux de mots →
fenêtres candidates calées sur ces frontières (jamais un début en plein
milieu d'une phrase) → pré-score heuristique (sans dépendance) → notation +
résumé + justification par Ollama (repli heuristique si absent) →
déduplication → tri par score.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from src.transcribe import Transcript, Word

ProgressCallback = Callable[[float, str], None]

_HOOK_RE = re.compile(
    r"^\s*(comment|pourquoi|combien|qui\b|quoi|est-ce|saviez|imagine|voici|le secret|"
    r"how|why|what|the secret|here'?s|\d)",
    re.IGNORECASE,
)
_DANGLING_RE = re.compile(
    r"^\s*(et|mais|donc|alors|parce que|car|puis|ensuite|enfin|du coup|"
    r"and|but|so|because|then|also|plus)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d")

# Reconstruction des phrases -------------------------------------------------
_SENT_END_RE = re.compile(r"[.!?…]+[\"'»)\]]?$")
_ABBREV_RE = re.compile(
    r"\b(m|mme|mr|dr|prof|etc|vs|cf|p|ex|no|n°|art|min|sec|km|kg|ch)\.?$",
    re.IGNORECASE,
)
_GAP_SPLIT = 0.40  # silence (s) entre deux mots qui marque une frontière de phrase
_LEAD_IN = 0.10    # petit pré-roll pour ne pas rogner la première syllabe


@dataclass(frozen=True)
class _Unit:
    start: float
    end: float
    text: str


def _join_words(words: list[Word]) -> str:
    text = " ".join(word.text.strip() for word in words if word.text.strip())
    text = re.sub(r"\s+([,.!?…;:»])", r"\1", text)
    text = re.sub(r"(«)\s+", r"\1", text)
    text = re.sub(r"(\w)\s+'\s*(\w)", r"\1'\2", text)  # l 'équipe -> l'équipe
    text = re.sub(r"\s+-(\w)", r"-\1", text)           # est -ce -> est-ce
    return text.strip()


def _flush(buffer: list[Word], units: list[_Unit]) -> None:
    text = _join_words(buffer)
    if text:
        text = text[:1].upper() + text[1:]  # Whisper capitalise mal après ses points
        units.append(_Unit(buffer[0].start, buffer[-1].end, text))
    buffer.clear()


def _sentence_units(transcript: Transcript) -> list[_Unit]:
    """Reconstruit des unités ~phrases, chacune démarrant sur une vraie frontière.

    Frontière = ponctuation finale portée par un mot, OU silence marqué avant
    le mot suivant, OU début d'un segment Whisper que le modèle a capitalisé
    (c.-à-d. qu'il considère comme une nouvelle phrase). Reste fiable même
    quand Whisper ponctue peu et coupe ses segments en plein milieu d'une phrase.
    """
    segments = [seg for seg in transcript.segments if seg.text.strip()]
    if not segments:
        return []
    units: list[_Unit] = []
    buffer: list[Word] = []
    prev_end: float | None = None
    for seg in segments:
        seg_words = seg.words or [Word(seg.start, seg.end, seg.text.strip())]
        for position, word in enumerate(seg_words):
            token = word.text.strip()
            if buffer:
                gap = word.start - prev_end if prev_end is not None else 0.0
                fresh_segment = position == 0 and token[:1].isupper()
                if gap >= _GAP_SPLIT or fresh_segment:
                    _flush(buffer, units)
            buffer.append(word)
            prev_end = word.end
            if _SENT_END_RE.search(token) and not _ABBREV_RE.search(token):
                _flush(buffer, units)
    _flush(buffer, units)
    return units


@dataclass(frozen=True)
class Highlight:
    start: float
    end: float
    score: int              # 0-100
    title: str
    summary: str
    reasons: list[str] = field(default_factory=list)
    transcript: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def _candidate_windows(
    units: list, *, min_dur: float, max_dur: float, stride: int = 1,
) -> list[tuple[float, float, str]]:
    """Fenêtres calées sur des frontières de phrases (début ET fin).

    Un début qui tombe sur un connecteur suspendu (« Et donc… », « Du coup… »)
    est écarté : on glisse à l'unité suivante.
    """
    windows: list[tuple[float, float, str]] = []
    seen: set[tuple[float, float]] = set()
    count = len(units)
    for i in range(0, count, stride):
        if _DANGLING_RE.match(units[i].text):
            continue
        j = i
        while j < count and units[j].end - units[i].start < min_dur:
            j += 1
        if j >= count:  # plus assez de matière pour atteindre min_dur
            break
        while j + 1 < count and units[j + 1].end - units[i].start <= max_dur:
            j += 1
        start, end = units[i].start, units[j].end
        span = end - start
        if span < min_dur * 0.8 or span > max_dur + 0.5:
            continue
        key = (round(start, 1), round(end, 1))
        if key in seen:
            continue
        seen.add(key)
        text = " ".join(units[k].text for k in range(i, j + 1)).strip()
        windows.append((round(start, 2), round(end, 2), text))
    return windows


def _pre_score(text: str, duration: float) -> float:
    words = text.split()
    if len(words) < 15:
        return 0.0
    score = 0.0
    if _HOOK_RE.match(text):
        score += 0.25
    score += min(0.20, 0.10 * text.count("?"))
    score += min(0.15, 0.05 * len(_NUMBER_RE.findall(text)))
    score += min(0.10, 0.05 * text.count("!"))
    long_words = [w.strip(".,!?;:()").lower() for w in words if len(w) > 4]
    if long_words:
        repeats = max(long_words.count(w) for w in set(long_words))
        score += min(0.20, 0.06 * (repeats - 1))
    if 25 <= duration <= 60:
        score += 0.15
    elif duration < 18 or duration > 80:
        score -= 0.15
    if _DANGLING_RE.match(text):
        score -= 0.10
    return max(0.0, min(1.0, score))


def _dedupe(
    scored: list[tuple[float, float, str, float]], overlap: float = 0.5,
) -> list[tuple[float, float, str, float]]:
    kept: list[tuple[float, float, str, float]] = []
    for start, end, text, pre in sorted(scored, key=lambda item: item[3], reverse=True):
        span = end - start
        clashes = any(
            (min(end, k_end) - max(start, k_start)) / span > overlap
            for k_start, k_end, _, _ in kept
            if not (end <= k_start or start >= k_end)
        )
        if not clashes:
            kept.append((start, end, text, pre))
    return kept


_BATCH_SIZE = 4
_SYSTEM_BATCH = (
    "Tu es un expert du montage de clips courts viraux (TikTok, Reels, Shorts). "
    "On te donne une liste numérotée d'extraits (transcriptions). Pour CHAQUE extrait, évalue "
    "le potentiel viral. Réponds UNIQUEMENT en JSON : "
    '{"clips": [{"i": <numéro de l\'extrait>, "score": <entier 0-100>, '
    '"title": "<accroche FR, max 12 mots>", "summary": "<une phrase: de quoi ça parle>", '
    '"reasons": ["<justif courte FR>", ...]}, ...]} — un objet par extrait, dans l\'ordre.'
)
_SYSTEM_ONE = (
    "Tu es un expert des clips courts viraux. À partir de la transcription d'un extrait, "
    'réponds UNIQUEMENT en JSON : {"score": <entier 0-100>, "title": "<accroche FR, max 12 mots, '
    'pas la transcription brute>", "summary": "<une phrase: de quoi parle l\'extrait>", '
    '"reasons": ["<justification courte FR>", ...]}.'
)


def _short_label(text: str, max_words: int) -> str:
    words = re.sub(r"\s+", " ", text.strip()).split()
    label = " ".join(words[:max_words]).rstrip(" ,;:.—-")
    if len(words) > max_words:
        label += "…"
    return (label[:1].upper() + label[1:]) if label else "Extrait"


def _looks_raw(value: str, text: str) -> bool:
    """Un titre qui n'est en fait qu'un long bout de transcription brute."""
    value = value.strip()
    if not value or value[:1].islower():
        return True
    return len(value) > 70 and value.lower()[:40] == text.strip().lower()[:40]


def _normalise_rating(raw: dict, fallback_text: str) -> dict:
    reasons = [str(r).strip() for r in raw.get("reasons", []) if str(r).strip()]
    title = str(raw.get("title") or "").strip()
    if _looks_raw(title, fallback_text):
        title = _short_label(fallback_text, 10)
    summary = str(raw.get("summary") or "").strip()
    if _looks_raw(summary, fallback_text) and summary.lower()[:20] == title.lower()[:20]:
        summary = _short_label(fallback_text, 26)
    return {
        "score": max(0, min(100, int(float(raw.get("score", 50))))),
        "title": title[:110],
        "summary": summary or _short_label(fallback_text, 26),
        "reasons": reasons[:3],
    }


def _rate_batch_with_llm(texts: list[str], model: str) -> list[dict | None]:
    from src.llm import chat_json

    body = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(texts))
    data = chat_json(_SYSTEM_BATCH, body, model=model, timeout=180.0)
    items = data.get("clips") or data.get("results") or data.get("extraits") or []
    aligned: list[dict | None] = [None] * len(texts)
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("i", position))
        except (TypeError, ValueError):
            index = position
        if 0 <= index < len(texts):
            aligned[index] = _normalise_rating(item, texts[index])
    return aligned


def _rate_one_with_llm(text: str, model: str) -> dict:
    from src.llm import chat_json

    data = chat_json(_SYSTEM_ONE, f'Transcription :\n"""\n{text}\n"""', model=model, timeout=60.0)
    return _normalise_rating(data, text)


def _rate_heuristic(text: str, pre: float) -> dict:
    reasons: list[str] = []
    if _HOOK_RE.match(text):
        reasons.append("Ouverture accrocheuse")
    if "?" in text:
        reasons.append("Contient une ou plusieurs questions")
    if _NUMBER_RE.search(text):
        reasons.append("Données chiffrées concrètes")
    if not reasons:
        reasons.append("Densité de mots-clés et longueur adaptées")
    return {
        "score": int(round(20 + pre * 70)),
        "title": _short_label(text, 10),
        "summary": _short_label(text, 26),
        "reasons": reasons[:3],
    }


def find_highlights(
    transcript: Transcript,
    *,
    target_count: int = 8,
    min_duration: float = 20.0,
    max_duration: float = 75.0,
    model: str | None = None,
    progress: ProgressCallback | None = None,
) -> list[Highlight]:
    """Renvoie les meilleurs extraits, classés par score décroissant."""
    report = progress or (lambda _value, _message: None)
    units = _sentence_units(transcript)
    if not units:
        return []

    report(0.1, "Repérage des phrases…")
    raw = _candidate_windows(units, min_dur=min_duration, max_dur=max_duration)
    scored = [(s, e, t, _pre_score(t, e - s)) for (s, e, t) in raw]
    scored = [item for item in scored if item[3] > 0.0]
    finalists = _dedupe(scored)[: max(target_count + 4, 10)]
    if not finalists:
        return []

    use_llm = model is not None
    highlights: list[Highlight] = []
    for offset in range(0, len(finalists), _BATCH_SIZE):
        chunk = finalists[offset : offset + _BATCH_SIZE]
        ratings: list[dict | None] = [None] * len(chunk)
        if use_llm:
            try:
                ratings = _rate_batch_with_llm([text for (_s, _e, text, _p) in chunk], model)
            except Exception:  # noqa: BLE001 - Ollama absent/incohérent -> heuristique
                use_llm = False
        for position, (start, end, text, pre) in enumerate(chunk):
            rated = ratings[position] if position < len(ratings) else None
            if rated is None and use_llm:
                # Le lot a sauté cet extrait : deuxième essai, un par un (fiable).
                try:
                    rated = _rate_one_with_llm(text, model)
                except Exception:  # noqa: BLE001 - Ollama tombé -> heuristique
                    use_llm = False
            if rated is None:
                rated = _rate_heuristic(text, pre)
            highlights.append(
                Highlight(
                    start=round(max(0.0, start - _LEAD_IN), 2), end=end,
                    score=rated["score"], title=rated["title"],
                    summary=rated["summary"], reasons=rated["reasons"], transcript=text,
                )
            )
        report(
            0.15 + 0.8 * min(offset + _BATCH_SIZE, len(finalists)) / len(finalists),
            "Notation des extraits…",
        )

    highlights.sort(key=lambda item: item.score, reverse=True)
    report(1.0, "Analyse terminée.")
    return highlights[:target_count]
