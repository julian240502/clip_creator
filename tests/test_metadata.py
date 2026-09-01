import pytest

from src import llm
from src.metadata import ClipMeta, generate_metadata, write_metadata

_TEXT = (
    "Le nouveau système de compétences change tout : on équipe trois artefacts par "
    "personnage et chaque artefact ouvre un arbre de talents différent. Les builds "
    "deviennent vraiment variés."
)


def test_heuristic_metadata_without_a_model() -> None:
    meta = generate_metadata(_TEXT)
    assert meta.title and len(meta.title.split()) <= 13
    assert meta.description
    assert 3 <= len(meta.hashtags) <= 8
    assert all(tag.startswith("#") for tag in meta.hashtags)


def test_hints_win_over_the_transcript() -> None:
    meta = generate_metadata(_TEXT, hint_title="Mon titre", hint_summary="Mon résumé")
    assert meta.title == "Mon titre"
    assert meta.description == "Mon résumé"


def test_as_text_has_three_blocks() -> None:
    meta = ClipMeta("Titre", "Une description.", ["#a", "#b"])
    assert meta.as_text() == "Titre\n\nUne description.\n\n#a #b\n"


def test_empty_text_returns_a_safe_meta() -> None:
    meta = generate_metadata("   ", hint_title="Secours")
    assert meta.title == "Secours"
    assert meta.hashtags == []


def test_write_metadata_roundtrip(tmp_path) -> None:
    path = write_metadata(ClipMeta("T", "D", ["#x"]), tmp_path / "clip_001.txt")
    assert path.read_text(encoding="utf-8") == "T\n\nD\n\n#x\n"


def test_generate_metadata_uses_the_llm_when_given_a_model(monkeypatch) -> None:
    monkeypatch.setattr(
        llm, "chat_json",
        lambda *_a, **_k: {
            "title": "Titre IA", "description": "Desc IA",
            "hashtags": ["ia", "#clip"],
        },
    )
    meta = generate_metadata(_TEXT, model="llama3")
    assert meta.title == "Titre IA"
    assert meta.hashtags == ["#ia", "#clip"]


def test_generate_metadata_falls_back_when_the_llm_raises(monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(llm, "chat_json", _boom)
    meta = generate_metadata(_TEXT, model="llama3")
    assert meta.title  # heuristic result, no exception
    assert meta.hashtags


def test_generate_metadata_prompt_targets_the_source_language(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def _capture(system, _user, **_k):
        seen["system"] = system
        return {"title": "T", "description": "D", "hashtags": ["#a"]}

    monkeypatch.setattr(llm, "chat_json", _capture)
    generate_metadata(_TEXT, model="llama3", language="en")
    assert "anglais" in seen["system"]
    generate_metadata(_TEXT, model="llama3", language=None)
    assert "même langue que la transcription" in seen["system"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
