"""Titres, descriptions et hashtags par clip (jalon F).

Utilise Ollama si un modèle est fourni, sinon une génération heuristique.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_STOPWORDS = {
    "dans", "pour", "avec", "cette", "votre", "nous", "vous", "mais", "donc", "alors",
    "être", "fait", "plus", "tout", "tous", "comme", "leur", "sont", "cela",
    "that", "this", "with", "have", "from", "your", "there", "about", "which",
}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]{3,}")

_SYSTEM_PROMPT = (
    "Tu génères les métadonnées d'un clip court à partir de sa transcription. "
    'Réponds UNIQUEMENT en JSON : {"title": "<accroche en français, max 12 mots>", '
    '"description": "<2 phrases max, en français, qui donnent envie de regarder>", '
    '"hashtags": ["#motcle", ...] (5 à 8, en français, sans espace, pertinents)}.'
)


@dataclass(frozen=True)
class ClipMeta:
    title: str
    description: str
    hashtags: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        blocks = [self.title.strip(), self.description.strip(), " ".join(self.hashtags).strip()]
        return "\n\n".join(block for block in blocks if block) + "\n"


def _first_sentences(text: str, count: int = 2) -> str:
    return " ".join(_SENTENCE_SPLIT_RE.split(text.strip())[:count]).strip()


def _short_title(text: str, max_words: int = 12) -> str:
    first = _first_sentences(text, 1)
    words = first.split()
    return first if len(words) <= max_words else " ".join(words[:max_words]) + "…"


def _keywords(text: str, count: int = 6) -> list[str]:
    freq: dict[str, int] = {}
    for word in _WORD_RE.findall(text.lower()):
        if word in _STOPWORDS:
            continue
        freq[word] = freq.get(word, 0) + 1
    ranked = sorted(freq, key=lambda word: (freq[word], len(word)), reverse=True)
    return ["#" + re.sub(r"[^a-zà-ÿ0-9]", "", word) for word in ranked[:count]]


def _heuristic(text: str, hint_title: str, hint_summary: str) -> ClipMeta:
    return ClipMeta(
        title=hint_title.strip() or _short_title(text),
        description=hint_summary.strip() or _first_sentences(text, 2),
        hashtags=_keywords(text),
    )


def generate_metadata(
    text: str,
    *,
    hint_title: str = "",
    hint_summary: str = "",
    model: str | None = None,
) -> ClipMeta:
    text = text.strip()
    if not text:
        return ClipMeta(title=hint_title or "Clip", description=hint_summary, hashtags=[])
    if model:
        try:
            from src.llm import chat_json

            data = chat_json(
                _SYSTEM_PROMPT, f'Transcription :\n"""\n{text}\n"""', model=model, timeout=90.0,
            )
            tags = [str(tag).strip() for tag in data.get("hashtags", []) if str(tag).strip()]
            tags = [tag if tag.startswith("#") else f"#{tag}" for tag in tags]
            return ClipMeta(
                title=(str(data.get("title") or "").strip() or hint_title or "Clip")[:120],
                description=str(data.get("description") or "").strip() or hint_summary,
                hashtags=tags[:8] or _keywords(text),
            )
        except Exception:  # noqa: BLE001 - Ollama absent/incohérent -> heuristique
            pass
    return _heuristic(text, hint_title, hint_summary)


def write_metadata(meta: ClipMeta, path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(meta.as_text(), encoding="utf-8")
    return destination
