from src import llm
from src.transcribe import Transcript, TranscriptSegment, Word
from src.translate import _synthetic_words, language_supported, translate_transcript


def _en_transcript() -> Transcript:
    segs = [
        TranscriptSegment(0.0, 2.0, "Hello everyone.", [Word(0.0, 2.0, "Hello everyone.")]),
        TranscriptSegment(2.0, 5.0, "This is a test.", [Word(2.0, 5.0, "This is a test.")]),
    ]
    return Transcript(language="en", duration=5.0, model="test", segments=segs)


def test_language_supported() -> None:
    assert language_supported("fr") and language_supported("ZH")
    assert not language_supported("") and not language_supported("xx")


def test_synthetic_words_span_the_segment_and_weight_by_length() -> None:
    words = _synthetic_words("Bonjour à tous", 10.0, 12.0, "fr")
    assert [w.text for w in words] == ["Bonjour", "à", "tous"]
    assert words[0].start == 10.0
    assert words[-1].end == 12.0                 # colle pile à la fin du segment
    assert all(w.end > w.start for w in words)    # chaque mot a une durée positive
    # "à" (1 lettre) doit recevoir moins de temps que "Bonjour" (7 lettres).
    assert (words[0].end - words[0].start) > (words[1].end - words[1].start)


def test_synthetic_words_split_chinese_by_character() -> None:
    words = _synthetic_words("你好世界", 0.0, 2.0, "zh")
    assert [w.text for w in words] == list("你好世界")
    assert words[-1].end == 2.0


def test_translate_transcript_replaces_text_and_synthesizes_word_timings(monkeypatch, tmp_path) -> None:
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
    assert out.words != []                       # timings synthétiques -> animation utilisable
    assert out.segments[0].words[0].start == 0.0
    assert out.segments[0].words[-1].end == 2.0   # calé sur la fin du segment d'origine
    assert "français" in calls[0]

    # 2e appel : servi depuis le cache disque, pas de nouvel appel LLM.
    calls.clear()
    again = translate_transcript(_en_transcript(), "fr", model="llama3")
    assert [s.text for s in again.segments] == ["Bonjour à tous.", "Ceci est un test."]
    assert again.words != []
    assert calls == []


def test_translate_transcript_noops_without_model_or_same_language() -> None:
    tr = _en_transcript()
    assert translate_transcript(tr, "en", model="llama3") is tr      # déjà anglais
    assert translate_transcript(tr, "fr", model=None) is tr          # pas d'IA locale


def test_translate_transcript_ignores_malformed_llm_reply(monkeypatch, tmp_path) -> None:
    """Régression : Ollama répond parfois {"t": <int>} au lieu d'une liste — ne doit
    pas planter (`len()` sur un int) mais garder le texte VO pour ce lot."""
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"t": 2})

    out = translate_transcript(_en_transcript(), "fr", model="llama3")
    assert [s.text for s in out.segments] == ["Hello everyone.", "This is a test."]
