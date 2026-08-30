import subprocess
from pathlib import Path

import pytest

from src.downloader import _format_selector, _validate_url
from src.encoder import encoder_label, resolve_video_encoder, video_encoder_args
from src.resizer import resize_clip_for_vertical
from src.video_splitter import get_video_duration, split_video


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    output = tmp_path / "sample.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(output),
    ]
    subprocess.run(command, check=True)
    return output


def test_split_video_is_precise(sample_video: Path, tmp_path: Path) -> None:
    clips = split_video(sample_video, 2, tmp_path / "clips", encoder="cpu")
    assert len(clips) == 2
    assert get_video_duration(clips[0]) == pytest.approx(2, abs=0.15)
    assert get_video_duration(clips[1]) == pytest.approx(1, abs=0.15)


@pytest.mark.parametrize(
    ("mode", "quality", "expected_dimensions"),
    [("crop", "1080p", "1080,1920"), ("fit", "720p", "720,1280")],
)
def test_vertical_export(
    sample_video: Path, tmp_path: Path, mode: str,
    quality: str, expected_dimensions: str,
) -> None:
    output = resize_clip_for_vertical(
        sample_video, tmp_path / f"{mode}.mp4", mode,
        encoder="cpu", quality=quality,
    )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == expected_dimensions


def test_invalid_mode(sample_video: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resize_clip_for_vertical(sample_video, tmp_path / "bad.mp4", "stretch")


def test_vertical_segment_is_cut_in_one_pass(sample_video: Path, tmp_path: Path) -> None:
    output = resize_clip_for_vertical(
        sample_video, tmp_path / "segment.mp4", "crop",
        encoder="cpu", start=1, duration=1,
    )
    assert get_video_duration(output) == pytest.approx(1, abs=0.15)


def test_4k_vertical_export(sample_video: Path, tmp_path: Path) -> None:
    output = resize_clip_for_vertical(
        sample_video, tmp_path / "4k.mp4", "crop", encoder="cpu",
        quality="4k", start=0, duration=0.25,
    )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == "2160,3840"


def test_url_validation() -> None:
    assert _validate_url(" https://youtu.be/example ") == "https://youtu.be/example"
    with pytest.raises(ValueError):
        _validate_url("not-a-url")


def test_4k_download_selector() -> None:
    assert _format_selector(2160) == "bv*[height<=2160]+ba/b[height<=2160]/b"


def test_cpu_encoder_fallback() -> None:
    encoder = resolve_video_encoder("cpu")
    assert encoder == "libx264"
    assert "veryfast" in video_encoder_args(encoder)
    assert encoder_label(encoder) == "CPU · x264"
