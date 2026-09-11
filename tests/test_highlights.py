import urllib.error

import pytest

from src import llm
from src.highlights import (
    _BATCH_SIZE,
    HOOK_STRONG,
    Highlight,
    _adaptive_batch_size,
    _candidate_windows,
    _context_size,
    _dedupe,
    _hook_score,
    _looks_raw,
    _merge_adjacent_highlights,
    _normalise_rating,
    _opening,
    _pre_score,
    _rate_batch_with_llm,
    _rate_heuristic,
    _sentence_units,
    _short_label,
    _system_batch,
    _system_one,
    find_highlights,
)
from src.transcribe import Transcript, TranscriptSegment, Word


def _seg(start: float, end: float, text: str) -> TranscriptSegment:
    step = (end - start) / max(len(text.split()), 1)
    words = [
        Word(start + i * step, start + (i + 1) * step, token)
        for i, token in enumerate(text.split())
    ]
    return TranscriptSegment(start, end, text, words)


def _transcript() -> Transcript:
    segments = [
        _seg(0.0, 6.0, "Pourquoi 90 pour cent des gens échouent à cet exercice tout simple ?"),
        _seg(6.0, 14.0, "La raison est contre-intuitive et je vais vous la montrer maintenant."),
        _seg(14.0, 24.0, "Prenez trois secondes, respirez, et regardez ce qui se passe vraiment."),
        _seg(24.0, 30.0, "Et donc voilà, c'est tout pour aujourd'hui, merci."),
        _seg(30.0, 42.0, "Deuxième point important : la méthode fonctionne en quatre étapes claires."),
        _seg(42.0, 55.0, "Étape une, étape deux, étape trois, étape quatre, on récapitule ensemble."),
        _seg(55.0, 68.0, "Le résultat surprend tout le monde, y compris les experts du domaine."),
    ]
    return Transcript(language="fr", duration=68.0, model="test", segments=segments)


def test_candidate_windows_respect_duration_bounds() -> None:
    windows = _candidate_windows(_sentence_units(_transcript()), min_dur=20.0, max_dur=45.0)
    assert windows
    for start, end, _text in windows:
        assert 15.0 <= end - start <= 45.5


def test_sentence_units_split_on_punctuation_and_pause() -> None:
    units = _sentence_units(_transcript())
    # Une unité par phrase ponctuée de la transcription de test.
    assert len(units) == 7
    assert units[0].text.startswith("Pourquoi")
    assert units[0].text.rstrip().endswith("?")
    assert units[3].text.startswith("Et donc")


def test_sentence_units_break_at_a_capitalised_segment_start() -> None:
    # Deux phrases sans ponctuation finale ; Whisper capitalise le début de la
    # seconde -> on coupe là plutôt que de tout agglomérer.
    segs = [
        _seg(0.0, 4.0, "on parle d'abord du contexte general sans transition nette"),
        _seg(4.0, 9.0, "Ensuite vient la partie vraiment interessante du sujet"),
    ]
    tr = Transcript(language="fr", duration=9.0, model="t", segments=segs)
    units = _sentence_units(tr)
    assert len(units) == 2
    assert units[1].text.startswith("Ensuite")


def test_sentence_units_from_word_gaps_without_punctuation() -> None:
    # Aucun signe de ponctuation : seuls les silences entre mots séparent.
    words = [
        Word(0.0, 0.4, "voici"), Word(0.4, 0.8, "le"), Word(0.8, 1.2, "plan"),
        Word(2.0, 2.4, "on"), Word(2.4, 2.8, "commence"), Word(2.8, 3.2, "maintenant"),
    ]
    tr = Transcript(
        language="fr", duration=3.2, model="t",
        segments=[TranscriptSegment(0.0, 3.2, "voici le plan on commence maintenant", words)],
    )
    units = _sentence_units(tr)
    assert [u.text for u in units] == ["Voici le plan", "On commence maintenant"]


def test_candidate_windows_never_start_on_a_dangling_connector() -> None:
    windows = _candidate_windows(_sentence_units(_transcript()), min_dur=18.0, max_dur=45.0)
    assert windows
    assert not any(text.lstrip().lower().startswith(("et donc", "du coup", "mais "))
                   for _s, _e, text in windows)


def test_no_window_starts_in_the_middle_of_a_sentence() -> None:
    # Whisper coupe souvent un segment en pleine phrase (suite en minuscule, sans
    # ponctuation ni pause) : aucune fenêtre candidate ne doit démarrer là.
    segs = [
        _seg(0.0, 6.0, "Le point vraiment important que je veux partager avec vous aujourd'hui"),
        _seg(6.0, 12.0, "et que presque personne ne comprend correctement au début de sa carrière"),
        _seg(12.0, 18.0, "c'est qu'il faut accepter de se tromper très souvent avant de progresser."),
        _seg(18.0, 26.0, "Voici la deuxième idée qui change tout quand on la met en pratique enfin."),
        _seg(26.0, 34.0, "Elle demande un peu de discipline mais les résultats arrivent vite ensuite."),
    ]
    tr = Transcript(language="fr", duration=34.0, model="t", segments=segs)
    units = _sentence_units(tr)
    unit_starts = {round(u.start, 2) for u in units}
    assert 6.0 not in unit_starts and 12.0 not in unit_starts  # continuations absorbées
    windows = _candidate_windows(units, min_dur=12.0, max_dur=30.0)
    assert windows
    for start, _end, text in windows:
        assert round(start, 2) in unit_starts
        assert text[:1].isupper() and not text.lower().startswith(("et ", "mais "))


def test_candidate_windows_end_on_a_sentence_boundary() -> None:
    units = _sentence_units(_transcript())
    unit_ends = {round(u.end, 2) for u in units}
    windows = _candidate_windows(units, min_dur=18.0, max_dur=45.0)
    assert windows
    assert all(round(end, 2) in unit_ends for _s, end, _t in windows)


def test_opening_takes_the_first_sentence() -> None:
    assert _opening("Tu fais ça mal. Voici pourquoi et comment corriger.") == "Tu fais ça mal."
    # Sans ponctuation : repli sur les premiers mots.
    long = " ".join(["mot"] * 40)
    assert len(_opening(long).split()) <= 22


def test_hook_score_rewards_a_question_opening() -> None:
    strong = _hook_score("Pourquoi est-ce que 90 % des gens abandonnent si vite ?")
    weak = _hook_score("Donc voilà en gros on va reparler un peu de tout ça tranquillement.")
    assert strong >= HOOK_STRONG
    assert weak < HOOK_STRONG
    assert strong > weak


def test_hook_score_penalises_dangling_and_filler_openings() -> None:
    assert _hook_score("Et du coup on continue sur le sujet précédent.") < HOOK_STRONG
    assert _hook_score("Euh bah je sais pas trop en fait.") < HOOK_STRONG


def test_find_highlights_attaches_hook_without_reordering() -> None:
    result = find_highlights(_transcript(), target_count=5, min_duration=18.0, max_duration=45.0)
    assert result
    # Le classement reste piloté par le score viral, pas par le hook.
    assert [h.score for h in result] == sorted((h.score for h in result), reverse=True)
    for h in result:
        assert 0 <= h.hook_score <= 100
        assert (h.hook_line != "") == (h.hook_score >= HOOK_STRONG)


def test_normalise_rating_reads_llm_hook_fields() -> None:
    out = _normalise_rating(
        {"score": 70, "hook": 88, "hook_line": "\"Personne ne te dit ça\"",
         "title": "Le conseil caché", "summary": "Un point clé."},
        "fallback",
    )
    assert out["hook_score"] == 88
    assert out["hook_line"] == "Personne ne te dit ça"


def test_normalise_rating_hook_absent_is_flagged_minus_one() -> None:
    out = _normalise_rating({"score": 60, "title": "T", "summary": "S"}, "fallback")
    assert out["hook_score"] == -1
    assert out["hook_line"] == ""


def test_system_prompts_target_the_transcript_language() -> None:
    assert "anglais" in _system_batch("en") and "anglais" in _system_one("en")
    assert "français" in _system_batch("fr")
    # Langue inconnue / absente -> consigne neutre, jamais un défaut FR figé.
    assert "même langue que la transcription" in _system_batch(None)
    assert "FR" not in _system_batch("en")  # plus de "accroche FR" en dur


def test_rate_heuristic_reasons_follow_the_language() -> None:
    en = _rate_heuristic("Why do 90% of people quit so fast?", 0.5, "en")
    fr = _rate_heuristic("Pourquoi 90 % des gens abandonnent si vite ?", 0.5, "fr")
    assert any(r[0].isupper() and "question" in r.lower() for r in en["reasons"])
    assert any("Contient" in r for r in fr["reasons"])


def test_pre_score_rewards_a_question_hook() -> None:
    hooky = "Pourquoi est-ce que personne n'en parle jamais ? " * 2 + "Voici la réponse claire."
    flat = "et donc on continue tranquillement sans rien de particulier à signaler ici voilà " * 2
    assert _pre_score(hooky, 35.0) > _pre_score(flat, 35.0)


def test_dedupe_drops_heavily_overlapping_windows() -> None:
    scored = [
        (0.0, 30.0, "a", 0.9),
        (2.0, 31.0, "b", 0.4),   # ~90% overlap with the first -> dropped
        (40.0, 70.0, "c", 0.5),
    ]
    kept = _dedupe(scored)
    assert {round(s) for s, _e, _t, _p in kept} == {0, 40}


def _hl(start: float, end: float, score: int, **kw) -> Highlight:
    return Highlight(
        start=start, end=end, score=score,
        title=kw.get("title", "Titre"), summary=kw.get("summary", "Résumé"),
        reasons=kw.get("reasons", []), transcript=kw.get("transcript", ""),
        hook_score=kw.get("hook_score", 0), hook_line=kw.get("hook_line", ""),
        chat_intensity=kw.get("chat_intensity", 0.0),
    )


def test_merge_adjacent_highlights_joins_a_continuation() -> None:
    a = _hl(0.0, 40.0, 60, title="A", transcript="première partie")
    b = _hl(42.0, 90.0, 55, title="B", transcript="suite de l'histoire")
    merged = _merge_adjacent_highlights([a, b], max_duration=120.0)
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (0.0, 90.0)
    assert merged[0].score == 60
    assert merged[0].title == "A"  # meilleur score garde le titre
    assert "première partie" in merged[0].transcript
    assert "suite de l'histoire" in merged[0].transcript


def test_merge_adjacent_highlights_leaves_far_apart_clips_alone() -> None:
    a = _hl(0.0, 30.0, 60)
    b = _hl(200.0, 240.0, 70)
    merged = _merge_adjacent_highlights([a, b], max_duration=120.0)
    assert len(merged) == 2


def test_merge_adjacent_highlights_respects_the_duration_cap() -> None:
    a = _hl(0.0, 100.0, 60)
    b = _hl(102.0, 200.0, 70)  # se touchent, mais l'union (200 s) dépasse max_duration
    merged = _merge_adjacent_highlights([a, b], max_duration=120.0)
    assert len(merged) == 2


def test_merge_adjacent_highlights_hook_comes_from_the_earlier_clip() -> None:
    # Le mieux noté est le second : le titre/résumé viennent de lui, mais le
    # hook doit rester celui du DÉBUT du clip fusionné (ce qui ouvre vraiment).
    a = _hl(0.0, 40.0, 50, hook_score=80, hook_line="Ouverture forte")
    b = _hl(41.0, 80.0, 90, hook_score=10, hook_line="")
    merged = _merge_adjacent_highlights([a, b], max_duration=120.0)
    assert merged[0].hook_score == 80
    assert merged[0].hook_line == "Ouverture forte"


def test_adaptive_batch_size_shrinks_for_long_clips() -> None:
    assert _adaptive_batch_size(60.0) == _BATCH_SIZE
    assert _adaptive_batch_size(120.0) < _BATCH_SIZE
    assert _adaptive_batch_size(180.0) < _adaptive_batch_size(60.0)
    assert _adaptive_batch_size(360.0) <= 2


def test_context_size_grows_with_content_and_stays_bounded() -> None:
    short = _context_size("sys", "un texte court", headroom=300)
    long_text = "mot " * 4000  # simule 4 extraits de 6 min dans un même lot
    long = _context_size("sys", long_text, headroom=300)
    assert short >= 2048
    assert long > short
    assert long <= 16384


def test_rate_batch_with_llm_passes_a_sized_num_ctx(monkeypatch) -> None:
    captured: dict = {}

    def _fake_chat_json(system, user, *, model, timeout=120.0, num_ctx=None):
        captured["num_ctx"] = num_ctx
        return {"clips": []}

    monkeypatch.setattr(llm, "chat_json", _fake_chat_json)
    _rate_batch_with_llm(["texte court"], "qwen2.5:3b")
    short_ctx = captured["num_ctx"]

    _rate_batch_with_llm([("mot " * 900)] * 4, "qwen2.5:3b")
    long_ctx = captured["num_ctx"]

    assert short_ctx and short_ctx >= 2048
    assert long_ctx > short_ctx


def test_find_highlights_heuristic_only_is_sorted_and_bounded() -> None:
    result = find_highlights(_transcript(), target_count=3, min_duration=18.0, max_duration=45.0)
    assert 1 <= len(result) <= 3
    assert [h.score for h in result] == sorted((h.score for h in result), reverse=True)
    for h in result:
        assert h.title and h.summary and h.reasons
        assert h.end > h.start


def test_find_highlights_source_window_keeps_extracts_inside_the_range() -> None:
    """Rediff de live : cadrer l'analyse sur une portion écarte tout extrait qui
    déborde des intros / pauses hors de cette portion."""
    full = find_highlights(_transcript(), target_count=8, min_duration=18.0, max_duration=45.0)
    assert any(h.start < 20.0 for h in full)  # sans fenêtre, des extraits démarrent tôt

    windowed = find_highlights(
        _transcript(), target_count=8, min_duration=18.0, max_duration=45.0,
        source_window=(20.0, 68.0),
    )
    assert windowed
    for h in windowed:
        assert h.start >= 20.0 - 0.01 and h.end <= 68.0 + 0.01


def test_find_highlights_source_window_too_tight_returns_nothing() -> None:
    assert find_highlights(
        _transcript(), min_duration=18.0, max_duration=45.0, source_window=(10.0, 15.0),
    ) == []


def test_find_highlights_seeds_a_window_from_a_chat_spike() -> None:
    """Un pic de chat crée un extrait autour de cet instant et le signale."""
    spike_t = 48.0
    result = find_highlights(
        _transcript(), target_count=6, min_duration=18.0, max_duration=45.0,
        chat_spikes=[(spike_t, 4.5)],
    )
    covering = [h for h in result if h.start <= spike_t <= h.end]
    assert covering, "le pic de chat doit être couvert par un extrait"
    # L'intensité du pic est portée par un champ dédié (badge UI), pas par reasons.
    assert any(h.chat_intensity >= 4.0 for h in covering)


def test_find_highlights_audio_intensity_lifts_the_hot_window() -> None:
    """Une fenêtre bruyante (rires / cris) est mieux notée qu'à texte égal sans."""
    hop = 0.5
    quiet = [0.1] * int(70 / hop)
    loud = list(quiet)
    for i in range(int(44 / hop), int(54 / hop)):     # ~10 s de fort vers 44-54 s
        loud[i] = 0.97

    plain = find_highlights(_transcript(), min_duration=18.0, max_duration=45.0)
    boosted = find_highlights(
        _transcript(), min_duration=18.0, max_duration=45.0,
        audio_curve=(loud, hop, 0.0),
    )
    hot = [h for h in boosted if h.start <= 48.0 <= h.end]
    assert hot
    assert any("intensité" in r.lower() or "🔊" in r for h in hot for r in h.reasons)
    assert max(h.score for h in boosted) >= max(h.score for h in plain)


_RAW = (
    "mal du coup avec du recul tu en as quels souvenirs mais franchement "
    "c'est quand même une belle période parce que quand tu as 15 ans tu te dis"
)


def test_short_label_is_capped_capitalised_and_not_mid_word() -> None:
    label = _short_label(_RAW, 10)
    assert label.split(" ")[0] == "Mal"          # capitalisé
    assert label.endswith("…")                    # tronqué proprement
    assert len(label.split(" ")) <= 11            # 10 mots + ellipse


def test_looks_raw_flags_a_transcript_dump_not_a_real_title() -> None:
    assert _looks_raw(_RAW, _RAW) is True
    assert _looks_raw("le titre en minuscule", _RAW) is True
    assert _looks_raw("Le vrai souvenir de ses 15 ans", _RAW) is False
    assert _looks_raw(_short_label(_RAW, 10), _RAW) is False  # le repli n'est pas "raw"


def test_normalise_rating_rescues_a_raw_llm_title() -> None:
    fixed = _normalise_rating({"score": 55, "title": _RAW, "summary": _RAW}, _RAW)
    assert fixed["title"] == _short_label(_RAW, 10)
    assert fixed["title"] != fixed["summary"]
    assert len(fixed["title"]) < len(_RAW)


def test_normalise_rating_keeps_a_good_llm_title() -> None:
    good = _normalise_rating(
        {"score": 72, "title": "Ses 15 ans, une belle période", "summary": "Elle en parle."}, _RAW,
    )
    assert good["title"] == "Ses 15 ans, une belle période"


def test_find_highlights_empty_transcript() -> None:
    empty = Transcript(language="fr", duration=0.0, model="t", segments=[])
    assert find_highlights(empty) == []


def test_ollama_available_is_false_when_nothing_listens() -> None:
    assert llm.ollama_available("http://127.0.0.1:1", timeout=0.3) is False
    assert llm.pick_model("http://127.0.0.1:1") is None


def test_pick_model_prefers_the_bigger_qwen_regardless_of_list_order(monkeypatch) -> None:
    monkeypatch.delenv("CLIP_CREATOR_LLM_MODEL", raising=False)
    # Ollama renvoie souvent le dernier modèle pull en tête -> le 3b avant le 7b.
    monkeypatch.setattr(llm, "list_models", lambda *a, **k: ["qwen2.5:3b", "qwen2.5:7b"])
    assert llm.pick_model() == "qwen2.5:7b"

    monkeypatch.setattr(llm, "list_models", lambda *a, **k: ["qwen2.5:3b", "mistral:latest"])
    assert llm.pick_model() == "qwen2.5:3b"                 # pas de 7b -> qwen2.5 générique

    monkeypatch.setenv("CLIP_CREATOR_LLM_MODEL", "mistral:latest")
    assert llm.pick_model() == "mistral:latest"             # forcé par env


def test_pick_rating_model_prefers_a_small_model_then_falls_back(monkeypatch) -> None:
    monkeypatch.delenv("CLIP_CREATOR_RATING_MODEL", raising=False)
    monkeypatch.delenv("CLIP_CREATOR_LLM_MODEL", raising=False)

    monkeypatch.setattr(llm, "list_models", lambda *a, **k: ["qwen2.5:7b", "qwen2.5:3b"])
    assert llm.pick_rating_model() == "qwen2.5:3b"          # petit modèle choisi

    monkeypatch.setattr(llm, "list_models", lambda *a, **k: ["qwen2.5:7b", "mistral:latest"])
    assert llm.pick_rating_model() == "qwen2.5:7b"          # aucun petit -> pick_model

    monkeypatch.setenv("CLIP_CREATOR_RATING_MODEL", "phi3:mini")
    assert llm.pick_rating_model() == "phi3:mini"           # forcé par env


def test_chat_json_parses_ollama_response(monkeypatch) -> None:
    class _Resp:
        status = 200

        def read(self) -> bytes:
            return b'{"message": {"content": "{\\"score\\": 77, \\"title\\": \\"ok\\"}"}}'

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *_a, **_k: _Resp())
    out = llm.chat_json("sys", "user", model="llama3")
    assert out == {"score": 77, "title": "ok"}


def test_chat_json_raises_when_ollama_unreachable(monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(llm.urllib.request, "urlopen", _boom)
    with pytest.raises(urllib.error.URLError):
        llm.chat_json("sys", "user", model="llama3")
