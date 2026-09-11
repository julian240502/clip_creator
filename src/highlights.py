"""Sélection intelligente : repère les extraits au plus fort potentiel viral.

Pipeline : reconstruction d'unités ~phrases à partir du flux de mots →
fenêtres candidates calées sur ces frontières (jamais un début en plein
milieu d'une phrase) → pré-score heuristique (sans dépendance) → notation +
résumé + justification par Ollama (repli heuristique si absent) →
déduplication → tri par score.

Chaque extrait porte aussi un `hook_score` (0-100) et une `hook_line` : ils
servent UNIQUEMENT à mettre en avant les clips qui ouvrent fort. Ils ne
filtrent rien et ne changent pas le classement (par score viral).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from src.llm import language_name
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

# Accroche (« hook ») ------------------------------------------------------------
HOOK_STRONG = 65   # seuil d'affichage du badge « accroche forte »
_HOOK_OPENING_WORDS = 22
_HOOK_QUESTION_RE = re.compile(
    r"\b(comment|pourquoi|combien|qui\b|quoi|est-ce|et si|savais?-tu|saviez-vous|"
    r"how|why|what\b)\b",
    re.IGNORECASE,
)
_HOOK_ADDRESS_RE = re.compile(r"\b(tu|t'|toi|ton|ta|tes|vous|votre|vos|you|your)\b", re.IGNORECASE)
_HOOK_CURIOSITY_RE = re.compile(
    r"(personne ne|la vérité|le secret|le vrai|voici pourquoi|la raison|le problème c'est|"
    r"le truc c'est|ce que .{0,30} c'est|j'ai (compris|découvert|réalisé|appris)|"
    r"la plupart des gens|tout le monde (croit|pense)|imagine|écoute|attends|le pire|le meilleur)",
    re.IGNORECASE,
)
_HOOK_BOLD_RE = re.compile(
    r"\b(jamais|toujours|incroyable|fou|folle|dingue|énorme|choquant|hallucinant|"
    r"personne|rien|aucun|le plus|la plus)\b",
    re.IGNORECASE,
)
_HOOK_FILLER_RE = re.compile(r"^\s*(euh|bah|ben|hmm|alors euh|donc euh|en fait euh)\b", re.IGNORECASE)


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
    score: int              # 0-100 (potentiel viral, sert au classement)
    title: str
    summary: str
    reasons: list[str] = field(default_factory=list)
    transcript: str = ""
    hook_score: int = 0     # 0-100, qualité des toutes premières secondes
    hook_line: str = ""     # phrase d'accroche mise en avant ("" si ouverture molle)
    chat_intensity: float = 0.0  # pic de chat Twitch sur l'extrait (0 = aucun, ~2.4+ = net)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def has_hook(self) -> bool:
        return self.hook_score >= HOOK_STRONG


def _opening(text: str, max_words: int = _HOOK_OPENING_WORDS) -> str:
    """Première phrase de l'extrait (à défaut, ses premiers mots)."""
    stripped = text.strip()
    match = re.match(r"(.{0,240}?[.!?…])(?:\s|$)", stripped)
    head = match.group(1) if match else " ".join(stripped.split()[:max_words])
    return head.strip()


def _hook_score(opening: str) -> int:
    """Note 0-100 de l'accroche à partir de sa seule première phrase."""
    text = opening.strip()
    if not text:
        return 0
    words = text.split()
    score = 25.0
    if "?" in text or _HOOK_QUESTION_RE.search(text):
        score += 24
    if _HOOK_CURIOSITY_RE.search(text):
        score += 20
    if _HOOK_ADDRESS_RE.search(text):
        score += 12
    if _NUMBER_RE.search(text):
        score += 10
    if _HOOK_BOLD_RE.search(text):
        score += 10
    if len(words) <= 12:
        score += 8
    elif len(words) >= 30:
        score -= 12
    if _DANGLING_RE.match(text):
        score -= 20
    if _HOOK_FILLER_RE.match(text):
        score -= 20
    return int(max(0, min(100, round(score))))


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


def _seed_windows_from_spikes(
    units: list, spikes: list[tuple[float, float]], *,
    min_dur: float, max_dur: float, existing: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Fenêtres candidates ancrées sur les pics de chat pas déjà couverts par une
    fenêtre existante — un moment où le chat explose devient un extrait même si
    le texte, seul, ne l'aurait pas retenu."""
    out: list[tuple[float, float, str]] = []
    for spike_t, _intensity in spikes:
        if any(s <= spike_t <= e for s, e, _ in existing + out):
            continue
        i = max((k for k in range(len(units)) if units[k].start <= spike_t + 0.5), default=0)
        if _DANGLING_RE.match(units[i].text) and i + 1 < len(units):
            i += 1
        j = i
        while j < len(units) and units[j].end - units[i].start < min_dur:
            j += 1
        if j >= len(units):
            continue
        while j + 1 < len(units) and units[j + 1].end - units[i].start <= max_dur * 0.9:
            j += 1
        start, end = units[i].start, units[j].end
        if end - start < min_dur * 0.8:
            continue
        text = " ".join(units[k].text for k in range(i, j + 1)).strip()
        out.append((round(start, 2), round(end, 2), text))
    return out


def _context_bonus(
    start: float, end: float,
    audio_curve: tuple[list[float], float, float] | None,
    spikes: list[tuple[float, float]] | None,
) -> tuple[float, float]:
    """(énergie 0..1, chat 0..1) pour la fenêtre `[start, end]`."""
    energy = 0.0
    if audio_curve:
        from src.audio_energy import excitement

        values, hop, offset = audio_curve
        energy = excitement(values, hop, start - offset, end - offset)
    chat = 0.0
    if spikes:
        hits = [inten for t, inten in spikes if start - 2.0 <= t <= end]
        chat = min(1.0, (max(hits) if hits else 0.0) / 3.0)
    return energy, chat


def _chat_peak(start: float, end: float, spikes: list[tuple[float, float]] | None) -> float:
    """Intensité brute du plus fort pic de chat sur `[start, end]` (0 si aucun)."""
    if not spikes:
        return 0.0
    hits = [inten for t, inten in spikes if start - 2.0 <= t <= end]
    return round(max(hits), 2) if hits else 0.0


# Score de viralité : pondération par priorité de signal (chat > ambiance >
# accroche > dialogue), réglable par profil de contenu — l'utilisateur choisit
# (pas de détection auto : trop de façons de se tromper sur un simple VOD).
# Chaque profil est (chat, énergie, hook, dialogue), toujours somme 100.
CONTENT_PROFILES: dict[str, tuple[float, float, float, float]] = {
    "gaming": (40.0, 25.0, 20.0, 15.0),      # gaming / réaction : chat & ambiance priorisés
    "podcast": (5.0, 15.0, 35.0, 45.0),      # podcast / interview : dialogue & accroche priorisés
    "balanced": (20.0, 20.0, 30.0, 30.0),    # équilibré
}
DEFAULT_CONTENT_PROFILE = "gaming"
# En dessous de ce score dialogue (texte quasi inexploitable : silence,
# transcription bruitée…), le score final est amorti — pas mis à zéro, un cri
# sans phrase claire doit pouvoir remonter sur le seul chat/ambiance.
_DIALOGUE_GATE_FLOOR = 15.0


def _score_weights(
    *, chat_available: bool, profile: str = DEFAULT_CONTENT_PROFILE,
) -> tuple[float, float, float, float]:
    """(chat, énergie, hook, dialogue) du profil demandé (repli sur le profil
    par défaut si inconnu). Le chat n'existe que sur un VOD Twitch **avec du
    chat récupéré** — quand ce n'est pas le cas (source non-Twitch, case
    décochée, chat vide/échoué), son poids ne se perd pas : il se redistribue
    au prorata sur les trois autres, sinon chaque extrait plafonnerait
    artificiellement bas pour une raison qui n'a rien à voir avec sa qualité.
    """
    w_chat, w_energy, w_hook, w_dialogue = CONTENT_PROFILES.get(
        profile, CONTENT_PROFILES[DEFAULT_CONTENT_PROFILE],
    )
    if chat_available or w_chat <= 0:
        return w_chat, w_energy, w_hook, w_dialogue
    rest = w_energy + w_hook + w_dialogue
    if rest <= 0:
        return 0.0, w_energy, w_hook, w_dialogue
    scale = (w_chat + rest) / rest
    return 0.0, w_energy * scale, w_hook * scale, w_dialogue * scale


def _weighted_score(
    *, dialogue_score: float, chat: float, energy: float, hook_score: float,
    weights: tuple[float, float, float, float],
) -> int:
    """Score 0-100 : somme pondérée des 4 signaux (chat, énergie, hook,
    dialogue — chacun 0..1), amortie sous `_DIALOGUE_GATE_FLOOR` (voir plus
    haut) pour ne pas laisser un pic isolé porter un extrait vide de contenu.
    """
    w_chat, w_energy, w_hook, w_dialogue = weights
    if dialogue_score >= _DIALOGUE_GATE_FLOOR:
        gate = 1.0
    else:
        gate = 0.6 + 0.4 * (dialogue_score / _DIALOGUE_GATE_FLOOR)
    raw = (
        w_chat * chat + w_energy * energy
        + w_hook * (hook_score / 100.0) + w_dialogue * (dialogue_score / 100.0)
    )
    return max(0, min(100, round(gate * raw)))


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


# Deux extraits notés séparément qui se suivent quasiment sans coupure sont, en
# pratique, la suite l'un de l'autre — _dedupe ne traite que le chevauchement
# (> 50 % du plus court), pas la simple contiguïté.
_ADJACENT_GAP = 6.0  # secondes


def _merge_adjacent_highlights(
    highlights: list[Highlight], *, max_duration: float, gap: float = _ADJACENT_GAP,
) -> list[Highlight]:
    """Fusionne deux extraits retenus qui se touchent ou se chevauchent avec un
    petit écart (< `gap`), tant que l'union tient dans `max_duration` — sinon
    ils restent deux extraits distincts (le classement par score tranche)."""
    if len(highlights) < 2:
        return highlights
    ordered = sorted(highlights, key=lambda h: h.start)
    merged: list[Highlight] = [ordered[0]]
    for current in ordered[1:]:
        prev = merged[-1]
        union_start, union_end = min(prev.start, current.start), max(prev.end, current.end)
        if current.start - prev.end <= gap and union_end - union_start <= max_duration:
            # Le hook vient forcément de celui qui commence en premier (ce sont
            # ses toutes premières secondes qui ouvrent le clip fusionné) ; le
            # reste (titre/résumé) vient du mieux noté des deux.
            first, second = (prev, current) if prev.start <= current.start else (current, prev)
            better = prev if prev.score >= current.score else current
            merged[-1] = Highlight(
                start=union_start, end=union_end, score=max(prev.score, current.score),
                title=better.title, summary=better.summary,
                reasons=list(dict.fromkeys(prev.reasons + current.reasons))[:3],
                transcript=f"{first.transcript} {second.transcript}".strip(),
                hook_score=first.hook_score, hook_line=first.hook_line,
                chat_intensity=max(prev.chat_intensity, current.chat_intensity),
            )
        else:
            merged.append(current)
    return merged


_BATCH_SIZE = 4
_HOOK_INSTRUCTION = (
    "Un clip vit ou meurt sur sa PREMIÈRE phrase : \"hook\" = à quel point cette "
    "première phrase donne envie de rester (question, promesse, chiffre, "
    "affirmation forte, curiosité). \"hook_line\" = cette phrase d'accroche "
    'recopiée telle quelle, ou "" si l\'ouverture est molle.'
)
def _output_language_clause(language: str | None) -> str:
    name = language_name(language)
    target = name if name != "la langue de la transcription" else "la même langue que la transcription"
    return (
        f' IMPORTANT : rédige "title", "summary", "hook_line" et "reasons" en {target}, '
        "jamais dans une autre langue."
    )


def _system_batch(language: str | None) -> str:
    return (
        "Tu es un expert du montage de clips courts viraux (TikTok, Reels, Shorts). "
        "On te donne une liste numérotée d'extraits (transcriptions). Pour CHAQUE extrait, évalue "
        "le potentiel viral. " + _HOOK_INSTRUCTION + " Réponds UNIQUEMENT en JSON : "
        '{"clips": [{"i": <numéro de l\'extrait>, "score": <entier 0-100>, '
        '"hook": <entier 0-100>, "hook_line": "<phrase ou \\"\\">", '
        '"title": "<accroche, max 12 mots>", "summary": "<une phrase: de quoi ça parle>", '
        '"reasons": ["<justif courte>", ...]}, ...]} — un objet par extrait, dans l\'ordre.'
        + _output_language_clause(language)
    )


def _system_one(language: str | None) -> str:
    return (
        "Tu es un expert des clips courts viraux. À partir de la transcription d'un extrait, "
        + _HOOK_INSTRUCTION + ' Réponds UNIQUEMENT en JSON : {"score": <entier 0-100>, '
        '"hook": <entier 0-100>, "hook_line": "<phrase ou \\"\\">", '
        '"title": "<accroche, max 12 mots, pas la transcription brute>", '
        '"summary": "<une phrase: de quoi parle l\'extrait>", '
        '"reasons": ["<justification courte>", ...]}.'
        + _output_language_clause(language)
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


def _coerce_score(value: object, default: int) -> int:
    try:
        return max(0, min(100, int(float(value))))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _normalise_rating(raw: dict, fallback_text: str) -> dict:
    reasons = [str(r).strip() for r in raw.get("reasons", []) if str(r).strip()]
    title = str(raw.get("title") or "").strip()
    if _looks_raw(title, fallback_text):
        title = _short_label(fallback_text, 10)
    summary = str(raw.get("summary") or "").strip()
    if _looks_raw(summary, fallback_text) and summary.lower()[:20] == title.lower()[:20]:
        summary = _short_label(fallback_text, 26)
    raw_hook = raw.get("hook", raw.get("hook_score"))
    hook_line = str(raw.get("hook_line") or "").strip().strip('"«»')
    return {
        "score": _coerce_score(raw.get("score", 50), 50),
        "title": title[:110],
        "summary": summary or _short_label(fallback_text, 26),
        "reasons": reasons[:3],
        # -1 => le modèle n'a pas noté l'accroche, l'appelant retombe sur l'heuristique.
        "hook_score": _coerce_score(raw_hook, -1) if raw_hook is not None else -1,
        "hook_line": hook_line[:140],
    }


# Garde-fou contexte Ollama --------------------------------------------------
# Sans num_ctx explicite, Ollama retombe sur le défaut du modèle (souvent
# 2048-4096 tokens) et TRONQUE SILENCIEUSEMENT un prompt trop long — pas
# d'erreur, juste une notation faite sur un texte incomplet. Un lot de 4
# extraits de 6 min ≈ 3600-4000 mots de transcript avant même la consigne
# système : largement de quoi dépasser ce défaut. On dimensionne num_ctx sur
# le contenu réel envoyé (borné pour ne pas gaspiller de VRAM sur un extrait
# court).
_MIN_NUM_CTX = 2048
_MAX_NUM_CTX = 16384


def _estimate_tokens(text: str) -> int:
    """Estimation grossière mais prudente (~3 caractères/token en FR/EN) :
    sert à dimensionner num_ctx, pas un budget précis — mieux vaut
    surestimer un peu que tronquer silencieusement."""
    return max(1, len(text) // 3)


def _context_size(*texts: str, headroom: int) -> int:
    """Taille de contexte Ollama pour ces textes + la réponse JSON attendue
    (`headroom`, en tokens), arrondie au Ko supérieur et bornée."""
    needed = sum(_estimate_tokens(t) for t in texts) + headroom
    return min(_MAX_NUM_CTX, max(_MIN_NUM_CTX, ((needed // 1024) + 1) * 1024))


def _adaptive_batch_size(max_duration: float) -> int:
    """Moins d'extraits par lot Ollama quand les clips sont longs : chaque lot
    envoie la transcription COMPLÈTE de chaque extrait dans un seul prompt.
    num_ctx (ci-dessus) empêche la troncature silencieuse, mais un lot plus
    petit reste plus rapide à traiter et plus fiable qu'un contexte énorme."""
    if max_duration <= 90:
        return _BATCH_SIZE
    if max_duration <= 180:
        return 3
    if max_duration <= 300:
        return 2
    return 1


def _rate_batch_with_llm(texts: list[str], model: str, language: str | None = None) -> list[dict | None]:
    from src.llm import chat_json

    system = _system_batch(language)
    body = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(texts))
    # ~200 tokens de réponse JSON par extrait (titre + résumé + raisons) + marge.
    num_ctx = _context_size(system, body, headroom=200 * len(texts) + 300)
    data = chat_json(system, body, model=model, timeout=180.0, num_ctx=num_ctx)
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


def _rate_one_with_llm(text: str, model: str, language: str | None = None) -> dict:
    from src.llm import chat_json

    system = _system_one(language)
    user = f'Transcription :\n"""\n{text}\n"""'
    num_ctx = _context_size(system, user, headroom=300)
    data = chat_json(system, user, model=model, timeout=60.0, num_ctx=num_ctx)
    return _normalise_rating(data, text)


_HEURISTIC_REASONS = {
    "fr": (
        "Ouverture accrocheuse", "Contient une ou plusieurs questions",
        "Données chiffrées concrètes", "Densité de mots-clés et longueur adaptées",
    ),
    "en": (
        "Strong opening", "Contains one or more questions",
        "Concrete numbers or stats", "Good keyword density and length",
    ),
}


def _rate_heuristic(text: str, pre: float, language: str | None = None) -> dict:
    hook_r, question_r, number_r, default_r = (
        _HEURISTIC_REASONS["en"] if (language or "").lower().startswith("en")
        else _HEURISTIC_REASONS["fr"]
    )
    reasons: list[str] = []
    if _HOOK_RE.match(text):
        reasons.append(hook_r)
    if "?" in text:
        reasons.append(question_r)
    if _NUMBER_RE.search(text):
        reasons.append(number_r)
    if not reasons:
        reasons.append(default_r)
    return {
        "score": int(round(20 + pre * 70)),
        "title": _short_label(text, 10),
        "summary": _short_label(text, 26),
        "reasons": reasons[:3],
        "hook_score": -1,   # calculé par find_highlights à partir de l'ouverture
        "hook_line": "",
    }


def find_highlights(
    transcript: Transcript,
    *,
    target_count: int = 8,
    min_duration: float = 20.0,
    max_duration: float = 75.0,
    model: str | None = None,
    progress: ProgressCallback | None = None,
    source_window: tuple[float, float] | None = None,
    audio_curve: tuple[list[float], float, float] | None = None,
    chat_spikes: list[tuple[float, float]] | None = None,
    content_profile: str = DEFAULT_CONTENT_PROFILE,
) -> list[Highlight]:
    """Renvoie les meilleurs extraits, classés par score décroissant.

    `content_profile` (voir `CONTENT_PROFILES`) fixe la priorité des 4 signaux
    du score : `"gaming"` (chat & ambiance priorisés), `"podcast"` (dialogue &
    accroche priorisés) ou `"balanced"`.

    `source_window` restreint les extraits candidats à `(start, end)` de la source
    — pratique pour une rediff de live : on cadre sur la partie active.

    `audio_curve` `(valeurs, hop, offset)` (voir `src.audio_energy`) et
    `chat_spikes` `[(temps, intensité), …]` (voir `src.twitch_chat`) sont des
    signaux **non textuels** : ils créent des fenêtres autour des pics de chat et
    bonifient le score des passages intenses (rires, cris, chat qui s'emballe).
    """
    report = progress or (lambda _value, _message: None)
    language = transcript.language  # titres / résumés / hook dans la langue de la vidéo
    units = _sentence_units(transcript)
    if not units:
        return []

    report(0.1, "Repérage des phrases…")
    raw = _candidate_windows(units, min_dur=min_duration, max_dur=max_duration)
    if chat_spikes:
        raw = raw + _seed_windows_from_spikes(
            units, chat_spikes, min_dur=min_duration, max_dur=max_duration, existing=raw,
        )
    if source_window is not None:
        w0, w1 = source_window
        raw = [(s, e, t) for (s, e, t) in raw if s >= w0 - 0.01 and e <= w1 + 0.01]
        if not raw:
            return []

    scored: list[tuple[float, float, str, float]] = []
    for s, e, t in raw:
        energy, chat = _context_bonus(s, e, audio_curve, chat_spikes)
        pre = _pre_score(t, e - s) * (1.0 + 0.5 * energy + 0.7 * chat)
        if pre <= 0.0 and (chat > 0.25 or energy > 0.4):
            pre = 0.05 + 0.4 * chat + 0.3 * energy  # graine forte, texte plat
        if pre > 0.0:
            scored.append((s, e, t, pre))
    finalists = _dedupe(scored)[: max(target_count + 4, 10)]
    if not finalists:
        return []

    use_llm = model is not None
    batch_size = _adaptive_batch_size(max_duration)
    weights = _score_weights(chat_available=bool(chat_spikes), profile=content_profile)
    highlights: list[Highlight] = []
    for offset in range(0, len(finalists), batch_size):
        chunk = finalists[offset : offset + batch_size]
        ratings: list[dict | None] = [None] * len(chunk)
        if use_llm:
            try:
                ratings = _rate_batch_with_llm(
                    [text for (_s, _e, text, _p) in chunk], model, language,
                )
            except Exception:  # noqa: BLE001 - Ollama absent/incohérent -> heuristique
                use_llm = False
        for position, (start, end, text, pre) in enumerate(chunk):
            rated = ratings[position] if position < len(ratings) else None
            if rated is None and use_llm:
                # Le lot a sauté cet extrait : deuxième essai, un par un (fiable).
                try:
                    rated = _rate_one_with_llm(text, model, language)
                except Exception:  # noqa: BLE001 - Ollama tombé -> heuristique
                    use_llm = False
            if rated is None:
                rated = _rate_heuristic(text, pre, language)
            opening = _opening(text)
            hook_score = rated.get("hook_score", -1)
            if hook_score < 0:  # accroche non notée par le modèle -> heuristique
                hook_score = _hook_score(opening)
            # On ne met en avant une hook_line que pour les accroches qui valent le coup.
            if hook_score >= HOOK_STRONG:
                hook_line = (rated.get("hook_line") or "").strip() or opening
                hook_line = _short_label(hook_line, 16)  # phrase courte et lisible
            else:
                hook_line = ""
            energy, chat = _context_bonus(start, end, audio_curve, chat_spikes)
            reasons = list(rated["reasons"])
            # Le chat s'emballe -> badge dédié dans l'UI (chat_intensity), pas dans reasons.
            if energy >= 0.5:
                reasons.insert(0, "🔊 Pic d'intensité (rires / cris)")
            score = _weighted_score(
                dialogue_score=rated["score"], chat=chat, energy=energy,
                hook_score=hook_score, weights=weights,
            )
            highlights.append(
                Highlight(
                    start=round(max(0.0, start - _LEAD_IN), 2), end=end,
                    score=score, title=rated["title"],
                    summary=rated["summary"], reasons=reasons[:3], transcript=text,
                    hook_score=hook_score, hook_line=hook_line,
                    chat_intensity=_chat_peak(start, end, chat_spikes),
                )
            )
        report(
            0.15 + 0.8 * min(offset + batch_size, len(finalists)) / len(finalists),
            "Notation des extraits…",
        )

    # Deux extraits notés séparément peuvent être la suite quasi immédiate l'un
    # de l'autre (_dedupe n'écarte que les chevauchements > 50 %) : sans ça
    # l'utilisateur voit deux clips qui racontent la même histoire coupée en
    # deux. Fusionnés tant que ça tient dans max_duration.
    highlights = _merge_adjacent_highlights(highlights, max_duration=max_duration)

    # Classement par score viral uniquement : le hook n'est qu'une mise en avant.
    highlights.sort(key=lambda item: item.score, reverse=True)
    report(1.0, "Analyse terminée.")
    return highlights[:target_count]
