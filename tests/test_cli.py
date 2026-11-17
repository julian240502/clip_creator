import json
import subprocess
from pathlib import Path

import pytest

import cli
from src.transcribe import Transcript, TranscriptSegment, Word


@pytest.fixture()
def long_video(tmp_path: Path) -> Path:
    """Vidéo 640x360 de 60 s avec un son, assez longue pour plusieurs extraits."""
    output = tmp_path / "long.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=60",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(output),
    ], check=True)
    return output


def _fake_transcript(duration: float = 60.0) -> Transcript:
    sentences = [
        "Tu ne vas jamais croire ce qui vient de se passer.",
        "Le boss était à un point de vie et là, énorme retournement !",
        "Franchement c'est le meilleur moment du stream.",
        "Bon, on continue, il faut finir la mission.",
        "Attends, attends, regarde ce qui arrive derrière toi !",
        "Incroyable, personne ne s'y attendait, le chat explose.",
    ]
    segments, t = [], 0.0
    step = duration / len(sentences)
    for sentence in sentences:
        tokens = sentence.split()
        dt = (step - 0.6) / len(tokens)
        words = [Word(t + k * dt, t + (k + 1) * dt, " " + token) for k, token in enumerate(tokens)]
        segments.append(TranscriptSegment(t, t + step - 0.6, sentence, words))
        t += step
    return Transcript(language="fr", duration=duration, model="fake", segments=segments)


def _run(capsys, argv: list[str]) -> dict:
    assert cli.main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_parse_time_and_window():
    assert cli.parse_time("95") == 95.0
    assert cli.parse_time("1:35") == 95.0
    assert cli.parse_time("1:01:35.5") == 3695.5
    assert cli.parse_window("12:30-13:05") == (750.0, 785.0)
    with pytest.raises(cli.CliError):
        cli.parse_window("13:05-12:30")
    with pytest.raises(cli.CliError):
        cli.parse_time("abc")


def test_render_windows_with_texts(long_video, tmp_path, capsys):
    clips_file = tmp_path / "clips.json"
    clips_file.write_text(json.dumps([
        {"start": "0:40", "end": "0:46", "title": "Second", "description": "Desc 2", "hashtags": ["gta6", "#gaming"]},
        {"start": 5, "end": 11, "title": "Premier", "description": "Desc 1", "hashtags": "#a #b"},
    ]), encoding="utf-8")
    out = _run(capsys, [
        "render", str(long_video), "--clips", str(clips_file), "--window", "20-24",
        "--quality", "720p", "--encoder", "cpu", "--speed", "fast", "--publish", str(tmp_path / "drive"),
        "--label", "Test",
    ])
    clips = out["clips"]
    assert [c["start"] for c in clips] == [5.0, 20.0, 40.0]  # triés comme process_video
    assert all(Path(c["path"]).is_file() for c in clips)
    assert Path(clips[0]["text_file"]).read_text(encoding="utf-8").startswith("Premier")
    assert "#gta6 #gaming" in Path(clips[2]["text_file"]).read_text(encoding="utf-8")
    assert clips[1]["text_file"] is None  # --window sans texte
    assert len(list(Path(out["published_to"]).glob("clips/*.mp4"))) == 3


def test_render_regular_mode(long_video, capsys):
    out = _run(capsys, [
        "render", str(long_video), "--start", "10", "--end", "30", "--length", "10",
        "--quality", "720p", "--encoder", "cpu", "--speed", "fast",
    ])
    assert len(out["clips"]) == 2


def test_analyze_then_pick_with_captions(long_video, tmp_path, capsys, monkeypatch):
    import src.transcribe as transcribe_mod

    calls = {"n": 0}

    def fake_transcribe(*_args, **_kwargs):
        calls["n"] += 1
        return _fake_transcript()

    monkeypatch.setattr(transcribe_mod, "transcribe", fake_transcribe)
    monkeypatch.setattr(transcribe_mod, "transcription_available", lambda: True)
    analysis = _run(capsys, [
        "analyze", str(long_video), "--no-llm", "--count", "3", "--min", "8", "--max", "25",
        "--out", str(tmp_path / "analysis"),
    ])
    assert analysis["highlights"], "au moins un extrait attendu"
    assert Path(analysis["transcript_txt"]).read_text(encoding="utf-8").startswith("[0:00.0")
    first = analysis["highlights"][0]
    assert first["id"] == 1 and first["end"] > first["start"]

    out = _run(capsys, [
        "render", "--analysis", analysis["analysis"], "--pick", "1", "--captions", "active",
        "--quality", "720p", "--encoder", "cpu", "--speed", "fast",
    ])
    assert len(out["clips"]) == 1 and Path(out["clips"][0]["path"]).is_file()
    assert Path(out["clips"][0]["path"]).with_suffix(".ass").is_file()  # sous-titres gravés
    assert calls["n"] == 1  # le transcript de l'analyse est réutilisé, pas de 2e transcription


def test_render_meta_without_captions(long_video, capsys, monkeypatch):
    import src.transcribe as transcribe_mod

    monkeypatch.setattr(transcribe_mod, "transcribe", lambda *_a, **_k: _fake_transcript())
    monkeypatch.setattr(transcribe_mod, "transcription_available", lambda: True)
    out = _run(capsys, [
        "render", str(long_video), "--window", "0:10-0:20", "--meta",
        "--quality", "720p", "--encoder", "cpu", "--speed", "fast",
    ])
    text_file = out["clips"][0]["text_file"]
    assert text_file and Path(text_file).read_text(encoding="utf-8").strip()


def test_frames(long_video, tmp_path, capsys):
    out = _run(capsys, ["frames", str(long_video), "--window", "10-20", "--count", "3",
                        "--out", str(tmp_path / "frames")])
    assert len(out["frames"]) == 3 and all(Path(f["path"]).is_file() for f in out["frames"])


def test_usage_errors(capsys, long_video):
    assert cli.main(["render", str(long_video), "--pick", "1"]) == 2
    assert cli.main(["render", str(long_video), "--background", "split"]) == 2
    assert "Erreur" in capsys.readouterr().err
