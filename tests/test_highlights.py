import urllib.error

import pytest

from src import llm
from src.highlights import (
    _candidate_windows,
    _dedupe,
    _pre_score,
    _sentences,
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
    windows = _candidate_windows(_sentences(_transcript()), min_dur=20.0, max_dur=45.0)
    assert windows
    for start, end, _text in windows:
        assert 15.0 <= end - start <= 45.0


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


def test_find_highlights_heuristic_only_is_sorted_and_bounded() -> None:
    result = find_highlights(_transcript(), target_count=3, min_duration=18.0, max_duration=45.0)
    assert 1 <= len(result) <= 3
    assert [h.score for h in result] == sorted((h.score for h in result), reverse=True)
    for h in result:
        assert h.title and h.summary and h.reasons
        assert h.end > h.start


def test_find_highlights_empty_transcript() -> None:
    empty = Transcript(language="fr", duration=0.0, model="t", segments=[])
    assert find_highlights(empty) == []


def test_ollama_available_is_false_when_nothing_listens() -> None:
    assert llm.ollama_available("http://127.0.0.1:1", timeout=0.3) is False
    assert llm.pick_model("http://127.0.0.1:1") is None


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
