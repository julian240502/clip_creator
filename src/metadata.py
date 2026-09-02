"""Titres, descriptions et hashtags par clip (jalon F).

Utilise Ollama si un modèle est fourni, sinon une génération heuristique.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.llm import language_name

_STOPWORDS = {
    "dans", "pour", "avec", "cette", "votre", "nous", "vous", "mais", "donc", "alors",
    "être", "fait", "plus", "tout", "tous", "comme", "leur", "sont", "cela",
    "that", "this", "with", "have", "from", "your", "there", "about", "which",
}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]{3,}")
_MAX_CONTEXT_CHARS = 1500

def _system_prompt(language: str | None) -> str:
    name = language_name(language)
    target = name if name != "la langue de la transcription" else "la même langue que la transcription"
    return (
        "Tu génères les métadonnées d'un clip court à partir de sa transcription. "
        "Le titre de la vidéo source et un extrait plus large de la transcription "
        "te sont fournis pour comprendre le sujet général et éviter les contresens : "
        "utilise-les pour choisir des hashtags et un vocabulaire pertinents, mais le "
        "titre et la description doivent porter sur CE clip précis, pas sur la vidéo entière. "
        'Réponds UNIQUEMENT en JSON : {"title": "<accroche, max 12 mots>", '
        '"description": "<2 phrases max qui donnent envie de regarder>", '
        '"hashtags": ["#motcle", ...] (5 à 8, sans espace, pertinents)}. '
        f"Rédige le titre, la description ET les hashtags en {target}, jamais dans une autre langue."
    )


def _user_prompt(text: str, video_title: str, video_context: str) -> str:
    parts = []
    if video_title.strip():
        parts.append(f'Titre de la vidéo source : "{video_title.strip()}"')
    if video_context.strip():
        parts.append(
            "Extrait plus large de la vidéo (contexte, pas le clip) :\n\"\"\"\n"
            f"{video_context.strip()[:_MAX_CONTEXT_CHARS]}\n\"\"\""
        )
    parts.append(f'Transcription du clip :\n"""\n{text}\n"""')
    return "\n\n".join(parts)


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
    language: str | None = None,
    video_title: str = "",
    video_context: str = "",
) -> ClipMeta:
    text = text.strip()
    if not text:
        return ClipMeta(title=hint_title or "Clip", description=hint_summary, hashtags=[])
    if model:
        try:
            from src.llm import chat_json

            data = chat_json(
                _system_prompt(language), _user_prompt(text, video_title, video_context),
                model=model, timeout=90.0,
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
