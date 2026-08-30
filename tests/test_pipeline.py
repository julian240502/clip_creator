import subprocess
from pathlib import Path

import pytest

from src.downloader import _format_selector, _validate_url
from src.encoder import (
    PROBE_HEIGHT,
    PROBE_SOURCE,
    PROBE_WIDTH,
    encoder_label,
    resolve_video_encoder,
    video_encoder_args,
)
from src.resizer import (
    _black_background_filter,
    _blur_background_filter,
    resize_clip_for_vertical,
)
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


def test_vertical_export_preserves_landscape_video(
    sample_video: Path, tmp_path: Path,
) -> None:
    output = resize_clip_for_vertical(
        sample_video, tmp_path / "vertical.mp4", encoder="cpu", quality="720p",
        background="blur",
    )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == "720,1280"


def test_vertical_segment_is_cut_in_one_pass(sample_video: Path, tmp_path: Path) -> None:
    output = resize_clip_for_vertical(
        sample_video, tmp_path / "segment.mp4", encoder="cpu", start=1, duration=1,
    )
    assert get_video_duration(output) == pytest.approx(1, abs=0.15)


def test_4k_vertical_export(sample_video: Path, tmp_path: Path) -> None:
    output = resize_clip_for_vertical(
        sample_video, tmp_path / "4k.mp4", encoder="cpu",
        quality="4k", start=0, duration=0.25,
    )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == "2160,3840"


def test_cuda_black_filter_scales_then_adds_bands() -> None:
    video_filter = _black_background_filter(1080, 1920, cuda=True)
    assert "scale_cuda=" in video_filter
    assert "force_original_aspect_ratio=decrease" in video_filter
    assert "hwdownload" in video_filter
    assert "pad=1080:1920" in video_filter


def test_blur_filter_uses_compatible_software_composition() -> None:
    video_filter = _blur_background_filter(1080, 1920)
    assert "split=2" in video_filter
    assert "boxblur=9:2" in video_filter
    assert "overlay=" in video_filter
    assert "overlay_cuda" not in video_filter
    assert "scale_cuda" not in video_filter
    assert "[foreground]" in video_filter


def test_invalid_background_is_rejected(sample_video: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resize_clip_for_vertical(
            sample_video, tmp_path / "invalid.mp4",
            encoder="cpu", background="transparent",
        )


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


def test_fast_nvenc_uses_fastest_preset() -> None:
    arguments = video_encoder_args("h264_nvenc", "fast")
    assert arguments[arguments.index("-preset") + 1] == "p1"
    assert arguments[arguments.index("-cq") + 1] == "23"


def test_fast_cpu_profile() -> None:
    arguments = video_encoder_args("libx264", "fast")
    assert "ultrafast" in arguments
    assert "23" in arguments


def test_gpu_probe_uses_nvenc_compatible_dimensions() -> None:
    assert (PROBE_WIDTH, PROBE_HEIGHT) == (320, 180)
    assert "size=320x180" in PROBE_SOURCE
