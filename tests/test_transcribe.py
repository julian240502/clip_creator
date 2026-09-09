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


def test_unload_models_clears_the_cache() -> None:
    transcribe_mod._models[("x", "cpu", "int8")] = object()
    transcribe_mod.unload_models()
    assert transcribe_mod._models == {}


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


def test_source_key_depends_on_clip_range(sample_video: Path) -> None:
    whole = _source_key(sample_video, "tiny", None)
    part = _source_key(sample_video, "tiny", None, (100.0, 130.0))
    other = _source_key(sample_video, "tiny", None, (200.0, 230.0))
    assert whole != part != other and part != other


def test_extract_audio_seeks_and_limits_for_a_clip_range(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, list[str]] = {}

    def fake_run(cmd, *a, **k):
        seen["cmd"] = cmd
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(transcribe_mod.subprocess, "run", fake_run)
    _extract_audio(Path("v.mp4"), tmp_path / "o.wav", start=3600.0, end=3630.0)
    cmd = seen["cmd"]
    assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "3600.000"
    assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "30.000"


def test_transcribe_with_clip_range_offsets_timestamps_to_absolute(
    sample_video: Path, tmp_path: Path, monkeypatch,
) -> None:
    """Sur une portion analysée, on ne transcrit que l'extrait mais les
    horodatages renvoyés sont recalés dans le temps absolu de la source."""
    if not transcription_available():
        pytest.skip("faster-whisper non installé")

    monkeypatch.setattr(transcribe_mod, "_extract_audio", lambda *a, **k: None)
    monkeypatch.setattr(transcribe_mod, "_resolve_backend", lambda: ("cpu", "int8"))
    monkeypatch.setattr(transcribe_mod, "_load_model", lambda *a, **k: object())

    class W:
        def __init__(self, s, e, txt):
            self.start, self.end, self.word = s, e, txt

    class Seg:
        def __init__(self):
            self.start, self.end, self.text = 1.0, 2.5, "salut"
            self.words = [W(1.0, 1.4, "sa"), W(1.4, 2.5, "lut")]

    class Info:
        language = "fr"
        duration = 30.0

    monkeypatch.setattr(
        transcribe_mod, "_run_transcription", lambda *a, **k: (iter([Seg()]), Info()),
    )

    tr = transcribe(
        sample_video, model="tiny", cache_dir=tmp_path, cache=False,
        clip_range=(3600.0, 3630.0),
    )
    assert tr.segments[0].start == pytest.approx(3601.0)
    assert tr.segments[0].end == pytest.approx(3602.5)
    assert tr.segments[0].words[0].start == pytest.approx(3601.0)
    assert tr.segments[0].words[-1].end == pytest.approx(3602.5)
    assert tr.duration == pytest.approx(3630.0)


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
