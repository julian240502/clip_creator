"""Sélection intelligente : repère les extraits au plus fort potentiel viral.

Pipeline : fenêtres candidates aux frontières de phrases → pré-score heuristique
(sans dépendance) → notation + résumé + justification par Ollama (repli
heuristique si absent) → déduplication → tri par score.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from src.transcribe import Transcript, TranscriptSegment

ProgressCallback = Callable[[float, str], None]

_HOOK_RE = re.compile(
    r"^\s*(comment|pourquoi|combien|qui\b|quoi|est-ce|saviez|imagine|voici|le secret|"
    r"how|why|what|the secret|here'?s|\d)",
    re.IGNORECASE,
)
_DANGLING_RE = re.compile(
    r"^\s*(et|mais|donc|alors|parce que|car|puis|ensuite|and|but|so|because|then)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d")


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


def _sentences(transcript: Transcript) -> list[TranscriptSegment]:
    return [segment for segment in transcript.segments if segment.text.strip()]


def _candidate_windows(
    sentences: list[TranscriptSegment], *, min_dur: float, max_dur: float, stride: int = 2,
) -> list[tuple[float, float, str]]:
    windows: list[tuple[float, float, str]] = []
    count = len(sentences)
    for i in range(0, count, stride):
        j = i
        while j < count and sentences[j].end - sentences[i].start < min_dur:
            j += 1
        while j < count and sentences[j].end - sentences[i].start <= max_dur:
            j += 1
        if j <= i:
            continue
        start = sentences[i].start
        end = min(sentences[j - 1].end, start + max_dur)
        if end - start < min_dur * 0.8:
            continue
        text = " ".join(segment.text.strip() for segment in sentences[i:j])
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
    sentences = _sentences(transcript)
    if not sentences:
        return []

    report(0.1, "Repérage des segments…")
    raw = _candidate_windows(sentences, min_dur=min_duration, max_dur=max_duration)
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
                    start=start, end=end, score=rated["score"], title=rated["title"],
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
