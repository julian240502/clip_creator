from src import llm
from src.transcribe import Transcript, TranscriptSegment, Word
from src.translate import language_supported, translate_transcript


def _en_transcript() -> Transcript:
    segs = [
        TranscriptSegment(0.0, 2.0, "Hello everyone.", [Word(0.0, 2.0, "Hello everyone.")]),
        TranscriptSegment(2.0, 5.0, "This is a test.", [Word(2.0, 5.0, "This is a test.")]),
    ]
    return Transcript(language="en", duration=5.0, model="test", segments=segs)


def test_language_supported() -> None:
    assert language_supported("fr") and language_supported("ZH")
    assert not language_supported("") and not language_supported("xx")


def test_translate_transcript_replaces_text_as_a_block_per_segment(monkeypatch, tmp_path) -> None:
    """Sous-titres traduits = un bloc par segment (façon film), pas d'animation mot
    par mot : impossible d'aligner un vrai minutage sur un texte réécrit dans une
    autre langue, donc on n'invente pas un timing synthétique."""
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)
    calls: list[str] = []

    def fake_chat_json(system, user, **_kw):
        calls.append(system)
        return {"t": ["Bonjour à tous.", "Ceci est un test."]}

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)

    out = translate_transcript(_en_transcript(), "fr", model="llama3")
    assert out.language == "fr"
    assert [s.text for s in out.segments] == ["Bonjour à tous.", "Ceci est un test."]
    assert out.words == []  # pas de mots -> build_ass bascule en lignes par segment
    assert "français" in calls[0]

    # 2e appel : servi depuis le cache disque, pas de nouvel appel LLM.
    calls.clear()
    again = translate_transcript(_en_transcript(), "fr", model="llama3")
    assert [s.text for s in again.segments] == ["Bonjour à tous.", "Ceci est un test."]
    assert calls == []


def test_translate_transcript_noops_without_model_or_same_language() -> None:
    tr = _en_transcript()
    assert translate_transcript(tr, "en", model="llama3") is tr      # déjà anglais
    assert translate_transcript(tr, "fr", model=None) is tr          # pas d'IA locale


def test_translate_transcript_ignores_malformed_llm_reply(monkeypatch, tmp_path) -> None:
    """Régression : Ollama répond parfois {"t": <int>} au lieu d'une liste — ne doit
    pas planter (`len()` sur un int). Avec un seul segment, il n'y a rien à
    retenter en plus petit : le texte reste en VO."""
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"t": 2})

    tr = Transcript(
        language="en", duration=2.0, model="test",
        segments=[TranscriptSegment(0.0, 2.0, "Hello.", [Word(0.0, 2.0, "Hello.")])],
    )
    out = translate_transcript(tr, "fr", model="llama3")
    assert [s.text for s in out.segments] == ["Hello."]


def test_translate_transcript_retries_a_malformed_batch_in_smaller_halves(
    monkeypatch, tmp_path,
) -> None:
    """Un lot de plusieurs segments mal répondu est retenté par moitiés plus
    petites plutôt qu'abandonné en bloc — un modèle local se trompe plus souvent
    sur de gros lots que sur un seul segment à la fois."""
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)

    def fake_chat_json(system, user, **_kw):
        if len(user.strip().splitlines()) > 1:
            return {"t": 2}  # lot complet -> réponse mal formée (un nombre, pas une liste)
        return {"t": [f"Trad[{user.strip()}]"]}

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)

    out = translate_transcript(_en_transcript(), "fr", model="llama3")
    assert [s.text for s in out.segments] == [
        "Trad[[0] Hello everyone.]", "Trad[[0] This is a test.]",
    ]


def test_translate_transcript_only_translates_segments_within_windows(
    monkeypatch, tmp_path,
) -> None:
    """Optimisation perf : ne traduire que les segments qui chevauchent une fenêtre
    réellement exportée, pas tout le transcript d'une longue vidéo source."""
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)
    seen: list[str] = []

    def fake_chat_json(system, user, **_kw):
        seen.append(user)
        return {"t": ["Bonjour à tous."]}

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)

    out = translate_transcript(
        _en_transcript(), "fr", model="llama3", windows=[(0.0, 2.0)],
    )
    assert [s.text for s in out.segments] == ["Bonjour à tous.", "This is a test."]
    assert len(seen) == 1 and "Hello everyone." in seen[0]
    assert out.segments[1].words != []  # segment hors fenêtre : jamais envoyé, jamais touché
