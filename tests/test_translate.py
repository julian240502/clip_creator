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


def test_translate_transcript_replaces_text_and_drops_words(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "src.paths.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False,
    )
    monkeypatch.setattr(
        "src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False,
    )
    calls: list[str] = []

    def fake_chat_json(system, user, **_kw):
        calls.append(system)
        return {"t": ["Bonjour à tous.", "Ceci est un test."]}

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)

    out = translate_transcript(_en_transcript(), "fr", model="llama3")
    assert out.language == "fr"
    assert [s.text for s in out.segments] == ["Bonjour à tous.", "Ceci est un test."]
    assert out.words == []                      # plus de timing mot à mot
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
