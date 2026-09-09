import subprocess
from pathlib import Path

import pytest

from src import transcribe as transcribe_mod
from src.transcribe import (
    DEFAULT_MODEL,
    Transcript,
    TranscriptSegment,
    Word,
    _audio_clean_enabled,
    _extract_audio,
    _run_transcription,
    _source_key,
    dump_transcript,
    load_transcript,
    transcribe,
    transcription_available,
)


def _sample_transcript() -> Transcript:
    return Transcript(
        language="en",
        duration=3.0,
        model=DEFAULT_MODEL,
        segments=[
            TranscriptSegment(
                start=0.0, end=1.4, text="hello world",
                words=[Word(0.0, 0.6, "hello"), Word(0.7, 1.4, "world")],
            ),
        ],
    )


def test_transcription_available_returns_bool() -> None:
    assert isinstance(transcription_available(), bool)


def test_extract_audio_is_16k_mono_wav(sample_video: Path, tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    _extract_audio(sample_video, wav)
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", str(wav),
        ],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == "16000,1"


def test_audio_clean_toggle_via_env(monkeypatch) -> None:
    monkeypatch.delenv("CLIP_CREATOR_WHISPER_AUDIO_CLEAN", raising=False)
    assert _audio_clean_enabled() is True                     # actif par défaut
    monkeypatch.setenv("CLIP_CREATOR_WHISPER_AUDIO_CLEAN", "0")
    assert _audio_clean_enabled() is False


def test_source_key_depends_on_audio_clean(sample_video: Path, monkeypatch) -> None:
    monkeypatch.delenv("CLIP_CREATOR_WHISPER_AUDIO_CLEAN", raising=False)
    with_clean = _source_key(sample_video, "tiny", None)
    monkeypatch.setenv("CLIP_CREATOR_WHISPER_AUDIO_CLEAN", "0")
    assert _source_key(sample_video, "tiny", None) != with_clean


def test_extract_audio_applies_clean_filter_when_enabled(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, list[str]] = {}

    def fake_run(cmd, *a, **k):
        seen["cmd"] = cmd
        class R:  # noqa: D401 - stub minimal
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(transcribe_mod.subprocess, "run", fake_run)
    monkeypatch.setenv("CLIP_CREATOR_WHISPER_AUDIO_CLEAN", "1")
    _extract_audio(Path("x.mp4"), tmp_path / "o.wav")
    assert "-af" in seen["cmd"]
    monkeypatch.setenv("CLIP_CREATOR_WHISPER_AUDIO_CLEAN", "0")
    _extract_audio(Path("x.mp4"), tmp_path / "o.wav")
    assert "-af" not in seen["cmd"]


def test_run_transcription_prefers_the_batched_pipeline(monkeypatch) -> None:
    calls: list[str] = []

    class FakeBatched:
        def __init__(self, model=None):
            calls.append("construct")

        def transcribe(self, wav, **kw):
            calls.append("batched")
            assert kw["condition_on_previous_text"] is False
            assert kw["word_timestamps"] is True
            return iter(()), object()

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "BatchedInferencePipeline", FakeBatched)
    _run_transcription(object(), "a.wav", None)
    assert calls == ["construct", "batched"]


def test_run_transcription_falls_back_to_sequential(monkeypatch) -> None:
    import faster_whisper

    def boom(model=None):
        raise RuntimeError("pas de VRAM pour le batché")

    monkeypatch.setattr(faster_whisper, "BatchedInferencePipeline", boom)

    used: dict[str, dict] = {}

    class FakeModel:
        def transcribe(self, wav, **kw):
            used["kw"] = kw
            return iter(()), object()

    _run_transcription(FakeModel(), "a.wav", "en")
    assert used["kw"]["language"] == "en"
    assert used["kw"]["condition_on_previous_text"] is False


def test_transcript_json_roundtrip(tmp_path: Path) -> None:
    original = _sample_transcript()
    path = dump_transcript(original, tmp_path / "t.json")
    restored = load_transcript(path)
    assert restored == original
    assert [word.text for word in restored.words] == ["hello", "world"]
    assert restored.text == "hello world"


def test_transcribe_reads_cache_without_faster_whisper(sample_video: Path, tmp_path: Path) -> None:
    key = _source_key(sample_video, DEFAULT_MODEL, None)
    dump_transcript(_sample_transcript(), tmp_path / f"{key}.json")
    # Aucun modèle chargé : le résultat doit venir du cache.
    result = transcribe(sample_video, cache_dir=tmp_path)
    assert result.language == "en"
    assert result.words[0].text == "hello"


def test_transcribe_without_cache_and_without_dependency_raises(
    sample_video: Path, tmp_path: Path,
) -> None:
    if transcription_available():
        pytest.skip("faster-whisper est installé : le chemin d'erreur ne s'applique pas.")
    with pytest.raises(RuntimeError, match="faster-whisper"):
        transcribe(sample_video, cache_dir=tmp_path, cache=False)


@pytest.mark.skipif(not transcription_available(), reason="faster-whisper non installé")
def test_transcribe_real_tiny_model(sample_video: Path, tmp_path: Path) -> None:
    result = transcribe(sample_video, model="tiny", cache_dir=tmp_path)
    assert isinstance(result.language, str) and result.language
    assert result.duration == pytest.approx(3.0, abs=0.5)
    assert (tmp_path / f"{_source_key(sample_video, 'tiny', None)}.json").is_file()
