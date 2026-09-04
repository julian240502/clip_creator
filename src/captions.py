"""Génération de sous-titres animés au format ASS depuis une transcription mot-à-mot.

Les sous-titres sont ensuite incrustés (hardsub) par FFmpeg dans `resizer.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from src.transcribe import Transcript, Word

_PUNCT_ONLY = re.compile(r"^[^\w\s]+$", re.UNICODE)
_GLUE_PREFIXES = ("'", "’", "-", "&")

__all__ = [
    "CaptionStyle",
    "TEMPLATES",
    "CAPTION_FONTS",
    "CAPTION_MODES",
    "CAPTION_POSITIONS",
    "build_ass",
    "write_clip_captions",
]

# --- Options exposées à l'UI --------------------------------------------------
CAPTION_FONTS = [
    "Arial", "Arial Black", "Impact", "Verdana", "Trebuchet MS", "Tahoma", "Georgia",
    "Microsoft YaHei", "SimHei",       # chinois (Windows)
    "Malgun Gothic",                   # coréen — Hangul (Windows)
    "Yu Gothic",                       # japonais (Windows)
]
# Police imposée par langue de sous-titres pour avoir les bons glyphes.
LANG_FONT = {"zh": "Microsoft YaHei", "ko": "Malgun Gothic", "ja": "Yu Gothic"}


def font_for_language(code: str | None) -> str | None:
    return LANG_FONT.get((code or "").lower())
CAPTION_MODES = {
    "Mot actif": "active",
    "Karaoké": "karaoke",
    "Mot par mot": "word",
    "Ligne par ligne": "lines",
}
CAPTION_POSITIONS = {"Bas": "bottom", "Centre": "middle", "Haut": "top"}


@dataclass(frozen=True)
class CaptionStyle:
    font: str = "Arial Black"
    font_size: int = 64            # pixels à la résolution du clip
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFD400"
    outline_color: str = "#000000"
    outline: float = 3.5
    shadow: float = 0.0
    box: bool = False             # fond opaque derrière le texte
    box_color: str = "#000000"
    box_alpha: int = 160          # 0 = opaque, 255 = transparent
    position: str = "bottom"      # top | middle | bottom
    margin_v: int = 200           # pixels depuis le bord (haut ou bas)
    nudge_x: int = 0              # décalage fin en px (+ = vers la droite)
    nudge_y: int = 0              # décalage fin en px (+ = vers le haut)
    uppercase: bool = False
    mode: str = "active"          # lines | word | karaoke | active
    max_words: int = 5
    max_duration: float = 3.5
    fade_ms: int = 120


TEMPLATES: dict[str, CaptionStyle] = {
    "Mot actif": CaptionStyle(
        font="Arial Black", font_size=64, highlight_color="#FFD400",
        position="bottom", margin_v=230, mode="active", max_words=5,
    ),
    "Karaoké": CaptionStyle(
        font="Arial Black", font_size=60, highlight_color="#22D3EE",
        position="bottom", margin_v=230, mode="karaoke", max_words=6,
    ),
    "Fondu": CaptionStyle(
        font="Arial", font_size=54, outline=3.0, position="bottom", margin_v=210,
        mode="lines", fade_ms=260, max_words=7,
    ),
    "Gros titres": CaptionStyle(
        font="Impact", font_size=92, highlight_color="#FF2D55", outline=5.0,
        position="middle", mode="word", uppercase=True, max_words=3, fade_ms=80,
    ),
}

_ALIGNMENT = {"bottom": 2, "middle": 5, "top": 8}


def _pos_override(width: int, height: int, style: CaptionStyle) -> str:
    r"""`\pos(x,y)` quand l'utilisateur décale les sous-titres au pixel près.

    Chaîne vide sinon : on garde le placement par alignement + MarginV.
    """
    if not (style.nudge_x or style.nudge_y):
        return ""
    x = round(width / 2) + style.nudge_x
    if style.position == "top":
        y = style.margin_v
    elif style.position == "middle":
        y = round(height / 2)
    else:
        y = height - style.margin_v
    return f"\\pos({x},{y - style.nudge_y})"


# --- Helpers ASS -----------------------------------------------------------------
def _ass_colour(hex_rgb: str, alpha: int = 0) -> str:
    value = hex_rgb.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Couleur hex invalide : {hex_rgb!r}")
    red, green, blue = value[0:2], value[2:4], value[4:6]
    return f"&H{alpha:02X}{blue}{green}{red}".upper()


def _ass_time(seconds: float) -> str:
    centis = max(0, round(seconds * 100))
    hours, centis = divmod(centis, 360000)
    minutes, centis = divmod(centis, 6000)
    secs, centis = divmod(centis, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")


def _word_text(word: Word, style: CaptionStyle) -> str:
    text = word.text.strip()
    if style.uppercase:
        text = text.upper()
    return _escape(text)


# --- Découpage en lignes -------------------------------------------------------
@dataclass
class _Line:
    words: list[Word] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    text: str = ""          # texte prêt (sous-titres traduits, sans timing mot à mot)


def _merge_tokens(words: list[Word]) -> list[Word]:
    """Recolle ponctuation et clitiques (« ! », « 'est », « -ce », « &M's ») au mot précédent."""
    merged: list[Word] = []
    for word in words:
        token = word.text.strip()
        if not token:
            continue
        glue = bool(merged) and (
            _PUNCT_ONLY.match(token) is not None or token[0] in _GLUE_PREFIXES
        )
        if glue:
            previous = merged[-1]
            merged[-1] = Word(previous.start, max(previous.end, word.end), previous.text + token)
        else:
            merged.append(Word(word.start, word.end, token))
    return merged


def _clip_words(transcript: Transcript, clip_start: float, clip_end: float) -> list[Word]:
    kept: list[Word] = []
    for word in transcript.words:
        if word.end <= clip_start or word.start >= clip_end:
            continue
        kept.append(
            Word(
                start=max(0.0, word.start - clip_start),
                end=min(clip_end, word.end) - clip_start,
                text=word.text,
            )
        )
    return _merge_tokens(kept)


def _group_lines(words: list[Word], *, max_words: int, max_duration: float, gap: float = 0.6) -> list[_Line]:
    lines: list[_Line] = []
    current: list[Word] = []
    for word in words:
        if current:
            span = word.end - current[0].start
            pause = word.start - current[-1].end
            sentence_break = current[-1].text.strip().endswith((".", "!", "?", "…"))
            if len(current) >= max_words or span > max_duration or pause > gap or sentence_break:
                lines.append(_Line(current, current[0].start, current[-1].end))
                current = []
        current.append(word)
    if current:
        lines.append(_Line(current, current[0].start, current[-1].end))
    return lines


def _segment_lines(transcript: Transcript, clip_start: float, clip_end: float) -> list[_Line]:
    """Une ligne par segment (texte déjà prêt, ex. traduction) — calage au segment."""
    lines: list[_Line] = []
    for seg in transcript.segments:
        if seg.end <= clip_start or seg.start >= clip_end:
            continue
        text = seg.text.strip()
        if not text:
            continue
        lines.append(_Line(
            words=[], text=text,
            start=max(0.0, seg.start - clip_start),
            end=min(clip_end, seg.end) - clip_start,
        ))
    return lines


def _clamp_line_ends(lines: list[_Line], clip_duration: float, hold: float = 0.35) -> None:
    for index, line in enumerate(lines):
        ceiling = lines[index + 1].start - 0.02 if index + 1 < len(lines) else clip_duration
        line.end = min(line.end + hold, max(line.start + 0.2, ceiling))
        line.start = max(0.0, line.start)


# --- Génération des évènements par mode --------------------------------------
def _events_lines(lines: list[_Line], style: CaptionStyle, align: int, pos: str = ""):
    for line in lines:
        tags = f"\\an{align}{pos}"
        if style.fade_ms:
            tags += f"\\fad({style.fade_ms},{style.fade_ms})"
        if line.words:
            body = " ".join(_word_text(word, style) for word in line.words)
        else:
            body = _escape(line.text.upper() if style.uppercase else line.text)
        yield line.start, line.end, f"{{{tags}}}{body}"


def _events_active(lines: list[_Line], style: CaptionStyle, align: int, pos: str = ""):
    highlight = _ass_colour(style.highlight_color)
    for line in lines:
        count = len(line.words)
        for i, word in enumerate(line.words):
            seg_start = line.start if i == 0 else word.start
            seg_end = line.words[i + 1].start if i + 1 < count else line.end
            if seg_end <= seg_start:
                continue
            parts = []
            for j, other in enumerate(line.words):
                token = _word_text(other, style)
                if j == i:
                    token = f"{{\\c{highlight}\\fscx112\\fscy112}}{token}{{\\r}}"
                parts.append(token)
            tags = f"\\an{align}{pos}"
            if style.fade_ms and i == 0:
                tags += f"\\fad({style.fade_ms},0)"
            yield seg_start, seg_end, f"{{{tags}}}" + " ".join(parts)


def _events_word(lines: list[_Line], style: CaptionStyle, align: int, pos: str = ""):
    pop = "\\fscx62\\fscy62\\t(0,90,\\fscx100\\fscy100)"
    for line in lines:
        count = len(line.words)
        for i, word in enumerate(line.words):
            seg_start = line.start if i == 0 else word.start
            seg_end = line.words[i + 1].start if i + 1 < count else line.end
            if seg_end <= seg_start:
                continue
            tags = f"\\an{align}{pos}{pop}"
            if style.fade_ms:
                tags += f"\\fad({min(style.fade_ms, 80)},60)"
            yield seg_start, seg_end, f"{{{tags}}}{_word_text(word, style)}"


def _events_karaoke(lines: list[_Line], style: CaptionStyle, align: int, pos: str = ""):
    for line in lines:
        count = len(line.words)
        chunks = []
        for i, word in enumerate(line.words):
            end = line.words[i + 1].start if i + 1 < count else word.end
            duration_cs = max(1, round((end - word.start) * 100))
            chunks.append(f"{{\\kf{duration_cs}}}{_word_text(word, style)}")
        tags = f"\\an{align}{pos}"
        if style.fade_ms:
            tags += f"\\fad({style.fade_ms},{style.fade_ms})"
        yield line.start, line.end, f"{{{tags}}}" + " ".join(chunks)


_EMITTERS = {
    "lines": _events_lines,
    "active": _events_active,
    "word": _events_word,
    "karaoke": _events_karaoke,
}


# --- Document ASS ------------------------------------------------------------
def _style_line(style: CaptionStyle) -> str:
    border_style = 3 if style.box else 1
    if style.mode == "karaoke":
        primary = _ass_colour(style.highlight_color)
        secondary = _ass_colour(style.primary_color)
    else:
        primary = _ass_colour(style.primary_color)
        secondary = _ass_colour(style.primary_color)
    outline_c = _ass_colour(style.outline_color)
    back_c = _ass_colour(style.box_color, alpha=style.box_alpha)
    align = _ALIGNMENT.get(style.position, 2)
    return (
        f"Style: Caption,{style.font},{style.font_size},{primary},{secondary},"
        f"{outline_c},{back_c},1,0,0,0,100,100,0,0,{border_style},"
        f"{style.outline:g},{style.shadow:g},{align},60,60,{style.margin_v},1"
    )


def _document(width: int, height: int, style: CaptionStyle, events) -> str:
    head = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        _style_line(style),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    body = [
        f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,{text}"
        for start, end, text in events
    ]
    return "\n".join(head + body) + "\n"


def build_ass(
    transcript: Transcript,
    *,
    clip_start: float,
    clip_end: float,
    width: int,
    height: int,
    style: CaptionStyle,
) -> str:
    """Construit le contenu d'un fichier .ass pour la fenêtre [clip_start, clip_end]."""
    if style.mode not in _EMITTERS:
        raise ValueError(f"Mode de sous-titres inconnu : {style.mode!r}")
    if transcript.words:
        words = _clip_words(transcript, clip_start, clip_end)
        lines = _group_lines(words, max_words=style.max_words, max_duration=style.max_duration)
        emitter = _EMITTERS[style.mode]
    else:
        # Pas de timing mot à mot (ex. sous-titres traduits) → lignes par segment.
        lines = _segment_lines(transcript, clip_start, clip_end)
        emitter = _events_lines
    _clamp_line_ends(lines, clip_end - clip_start)
    align = _ALIGNMENT.get(style.position, 2)
    pos = _pos_override(width, height, style)
    events = list(emitter(lines, style, align, pos))
    return _document(width, height, style, events)


def write_clip_captions(
    transcript: Transcript,
    path: str | Path,
    *,
    clip_start: float,
    clip_end: float,
    width: int,
    height: int,
    style: CaptionStyle,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        build_ass(
            transcript, clip_start=clip_start, clip_end=clip_end,
            width=width, height=height, style=style,
        ),
        encoding="utf-8",
    )
    return destination


def style_from_options(template: str, **overrides) -> CaptionStyle:
    """Part d'un template et applique les réglages de l'UI."""
    base = TEMPLATES.get(template, TEMPLATES["Mot actif"])
    clean = {key: value for key, value in overrides.items() if value is not None}
    return replace(base, **clean)
