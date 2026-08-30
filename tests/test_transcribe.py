import subprocess
from pathlib import Path

import pytest

from src.transcribe import (
    DEFAULT_MODEL,
    Transcript,
    TranscriptSegment,
    Word,
    _extract_audio,
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
