from src import llm
from src.transcribe import Transcript, TranscriptSegment, Word
from src.translate import (
    _clean_unit_text,
    _looks_like_fragment,
    _merge_short_units,
    language_supported,
    translate_transcript,
)


def _en_transcript() -> Transcript:
    # Un vrai trou entre les deux phrases : elles ne doivent pas fusionner.
    segs = [
        TranscriptSegment(0.0, 2.0, "Hello everyone.", [Word(0.0, 2.0, "Hello everyone.")]),
        TranscriptSegment(2.9, 5.0, "This is a test.", [Word(2.9, 5.0, "This is a test.")]),
    ]
    return Transcript(language="en", duration=5.0, model="test", segments=segs)


def test_language_supported() -> None:
    assert language_supported("fr") and language_supported("ZH")
    assert not language_supported("") and not language_supported("xx")


def test_translate_transcript_replaces_text_as_a_block_per_sentence(monkeypatch, tmp_path) -> None:
    """Sous-titres traduits = un bloc par unité ~phrase (façon film), pas
    d'animation mot par mot."""
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)
    calls: list[str] = []

    def fake_chat_json(system, user, **_kw):
        calls.append(system)
        return {"t": ["Bonjour à tous.", "Ceci est un test."]}

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)

    out = translate_transcript(_en_transcript(), "fr", model="llama3")
    assert out.language == "fr"
    assert [s.text for s in out.segments] == ["Bonjour à tous.", "Ceci est un test."]
    assert out.words == []  # pas de mots -> build_ass bascule en lignes par bloc
    assert "français" in calls[0]

    # 2e appel : servi depuis le cache disque, pas de nouvel appel LLM.
    calls.clear()
    again = translate_transcript(_en_transcript(), "fr", model="llama3")
    assert [s.text for s in again.segments] == ["Bonjour à tous.", "Ceci est un test."]
    assert calls == []


def test_translate_transcript_splits_a_segment_by_real_word_timing(monkeypatch, tmp_path) -> None:
    """Un segment Whisper qui couvre plusieurs phrases est redécoupé en unités
    ~phrases, chacune portée sur sa fenêtre `[premier mot, dernier mot]` réelle —
    plus de bloc qui déborde sur les silences ni de fragment en avance."""
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)
    words = [
        Word(0.0, 0.4, "Hello"), Word(0.4, 0.9, "there."),
        # ~2 s de silence : la 2e phrase ne doit PAS commencer avant
        Word(3.0, 3.3, "This"), Word(3.3, 3.5, "is"), Word(3.5, 3.7, "a"),
        Word(3.7, 4.3, "longer"), Word(4.3, 5.0, "sentence."),
    ]
    seg = TranscriptSegment(0.0, 8.0, "Hello there. This is a longer sentence.", words)
    tr = Transcript(language="en", duration=8.0, model="t", segments=[seg])

    monkeypatch.setattr(
        llm, "chat_json",
        lambda *a, **k: {"t": ["Bonjour toi.", "Ceci est une phrase plus longue."]},
    )

    out = translate_transcript(tr, "fr", model="llama3")
    assert [s.text for s in out.segments] == ["Bonjour toi.", "Ceci est une phrase plus longue."]
    assert (out.segments[0].start, out.segments[0].end) == (0.0, 0.9)
    assert (out.segments[1].start, out.segments[1].end) == (3.0, 5.0)   # pas 0.9 -> 8.0


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
        if len([ln for ln in user.strip().splitlines() if ln.startswith("[")]) > 1:
            return {"t": 2}  # lot complet -> réponse mal formée (un nombre, pas une liste)
        return {"t": ["Trad-A" if "Hello" in user else "Trad-B"]}

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)

    out = translate_transcript(_en_transcript(), "fr", model="llama3")
    assert [s.text for s in out.segments] == ["Trad-A", "Trad-B"]


def test_translate_transcript_recovers_items_dropped_from_a_batch(monkeypatch, tmp_path) -> None:
    """Un lot renvoyé trop court (le modèle « oublie » le dernier élément) laisse
    des segments en VO : ils sont retentés un par un en passe finale."""
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)

    def fake_chat_json(system, user, **_kw):
        lines = user.strip().splitlines()
        if len(lines) > 1:                       # lot : renvoie un élément de moins
            return {"t": [f"FR{i}" for i in range(len(lines) - 1)]}
        return {"t": ["FR-solo"]}                # requête unitaire : fiable

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    out = translate_transcript(_en_transcript(), "fr", model="llama3")
    assert [s.text for s in out.segments] == ["FR0", "FR-solo"]


def test_clean_unit_text_strips_non_spoken_annotations() -> None:
    assert _clean_unit_text("[Music]") == ""
    assert _clean_unit_text("♪♪") == ""
    assert _clean_unit_text("[ Background noise ]") == ""
    assert _clean_unit_text("(laughter) c'est fou") == "c'est fou"
    assert _clean_unit_text("Bonjour tout le monde.") == "Bonjour tout le monde."


def test_looks_like_fragment_drops_lone_short_words_but_keeps_reactions() -> None:
    assert _looks_like_fragment("saint")           # bout de transcription
    assert _looks_like_fragment("the")
    assert _looks_like_fragment("insan")            # mot coupé
    assert not _looks_like_fragment("insane")       # vrai mot (6 lettres)
    assert not _looks_like_fragment("Quoi ?!")      # réaction ponctuée
    assert not _looks_like_fragment("Wow")          # interjection connue
    assert not _looks_like_fragment("Non.")
    assert not _looks_like_fragment("stop")         # utterance courante gardée
    assert not _looks_like_fragment("il court")     # 2 mots


def test_translate_transcript_drops_an_isolated_fragment(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)
    words = [
        Word(0.0, 0.4, "That"), Word(0.4, 0.7, "was"), Word(0.7, 1.2, "amazing."),
        Word(4.0, 4.3, "saint"),                       # fragment isolé (trou de 2.8 s)
        Word(8.0, 8.3, "Let's"), Word(8.3, 8.6, "go!"),
    ]
    tr = Transcript(
        language="en", duration=10.0, model="t",
        segments=[TranscriptSegment(0.0, 10.0, "That was amazing. saint Let's go!", words)],
    )
    monkeypatch.setattr(
        llm, "chat_json", lambda *a, **k: {"t": ["C'était incroyable.", "C'est parti !"]},
    )
    out = translate_transcript(tr, "fr", model="llama3")
    assert [s.text for s in out.segments] == ["C'était incroyable.", "C'est parti !"]


def test_clean_unit_text_keeps_real_words_inside_parentheses() -> None:
    """Un aparté du streamer entre parenthèses n'est PAS une annotation : on garde
    la parole, on enlève juste les crochets."""
    assert _clean_unit_text("he said (and I mean it) the game is rigged") == (
        "he said and I mean it the game is rigged"
    )
    assert _clean_unit_text("(my brother's name is Kevin)") == "my brother's name is Kevin"


def test_merge_short_units_glues_a_flash_to_the_next_line() -> None:
    from src.highlights import _Unit

    units = [
        _Unit(0.0, 0.4, "Ouais."),
        _Unit(0.6, 3.0, "et donc voilà ce que je voulais dire."),
        _Unit(9.0, 12.0, "Une autre phrase bien plus loin dans le temps."),
    ]
    merged = _merge_short_units(units)
    assert [u.text for u in merged] == [
        "Ouais. et donc voilà ce que je voulais dire.",
        "Une autre phrase bien plus loin dans le temps.",
    ]
    assert (merged[0].start, merged[0].end) == (0.0, 3.0)


def test_translate_transcript_drops_non_lexical_units(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)
    tr = Transcript(
        language="en", duration=6.0, model="t",
        segments=[TranscriptSegment(0.0, 6.0, "[Music] Hello there.", [
            Word(0.0, 1.5, "[Music]"), Word(3.0, 3.4, "Hello"), Word(3.4, 3.9, "there."),
        ])],
    )
    seen: list[str] = []

    def fake(system, user, **_kw):
        seen.append(user)
        return {"t": ["Bonjour toi."]}

    monkeypatch.setattr(llm, "chat_json", fake)
    out = translate_transcript(tr, "fr", model="llama3")
    assert [s.text for s in out.segments] == ["Bonjour toi."]
    assert len(seen) == 1 and "[Music]" not in seen[0] and "Hello there." in seen[0]


def test_translate_batch_prompt_carries_duration_and_context(monkeypatch, tmp_path) -> None:
    """Le corps envoyé au modèle porte la durée à l'écran de chaque réplique et,
    à partir du 2e lot, la réplique précédente en contexte."""
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr("src.translate._BATCH", 2, raising=False)
    bodies: list[str] = []

    def fake(system, user, **_kw):
        bodies.append(user)
        n = len([ln for ln in user.splitlines() if ln.startswith("[")])
        return {"t": [f"tr{i}" for i in range(n)]}

    monkeypatch.setattr(llm, "chat_json", fake)

    # phrases bien espacées (> 0.6 s) pour qu'elles ne fusionnent pas
    words = [Word(i * 2.0, i * 2.0 + 0.6, f"phrase{i}.") for i in range(6)]
    segs = [TranscriptSegment(w.start, w.end, w.text, [w]) for w in words]
    tr = Transcript(language="en", duration=12.0, model="t", segments=segs)

    translate_transcript(tr, "fr", model="llama3")
    # _sentence_units capitalise la 1re lettre -> "Phrase0."
    assert "(0.6s) Phrase0." in bodies[0]           # durée à l'écran dans le corps
    assert "Contexte déjà dit" not in bodies[0]     # 1er lot : pas de contexte
    assert "Contexte déjà dit" in bodies[1]         # 2e lot : réplique précédente


def test_translate_transcript_writes_a_debug_pairs_file(monkeypatch, tmp_path) -> None:
    """`debug_out` -> un fichier VO / traduction pour vérifier la fidélité."""
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(
        llm, "chat_json",
        lambda *a, **k: {"t": ["Bonjour à tous.", "Ceci est un test."]},
    )
    out = tmp_path / "translation.fr.txt"
    translate_transcript(_en_transcript(), "fr", model="llama3", debug_out=out)
    text = out.read_text(encoding="utf-8")
    assert "Hello everyone." in text and "-> Bonjour à tous." in text
    assert "[0:00]" in text


def test_translate_transcript_only_translates_units_within_windows(
    monkeypatch, tmp_path,
) -> None:
    """Perf : ne traiter que les segments qui chevauchent un clip exporté ; les
    autres n'apparaissent pas dans le transcript traduit (jamais rendus)."""
    monkeypatch.setattr("src.translate.TRANSCRIPTIONS_DIR", str(tmp_path), raising=False)
    seen: list[str] = []

    def fake_chat_json(system, user, **_kw):
        seen.append(user)
        return {"t": ["Bonjour à tous."]}

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)

    out = translate_transcript(
        _en_transcript(), "fr", model="llama3", windows=[(0.0, 2.0)],
    )
    assert [s.text for s in out.segments] == ["Bonjour à tous."]
    assert (out.segments[0].start, out.segments[0].end) == (0.0, 2.0)
    assert len(seen) == 1 and "Hello everyone." in seen[0]
