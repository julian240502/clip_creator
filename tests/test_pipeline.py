from pathlib import Path

import pytest

from src import downloader
from src.downloader import _format_selector, _validate_url, download_source
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
    segment_vertical,
)
from src.video_splitter import (
    get_video_duration,
    get_video_resolution,
    resolve_source_window,
    split_video,
)


def test_download_source_caches_by_url_and_quality(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_download_video(video_url: str, output_dir, max_height: int = 1080) -> str:
        calls.append((video_url, max_height))
        target = Path(output_dir) / "video.mp4"
        target.write_bytes(b"data" * 50)
        return str(target)

    monkeypatch.setattr(downloader, "download_video", fake_download_video)
    first = download_source("https://host.test/watch?v=abc", tmp_path, max_height=720)
    second = download_source("https://host.test/watch?v=abc", tmp_path, max_height=720)
    assert first == second
    assert len(calls) == 1  # 2e appel servi depuis le cache
    # Une autre qualité = un autre bucket = un nouveau téléchargement.
    download_source("https://host.test/watch?v=abc", tmp_path, max_height=1080)
    assert len(calls) == 2


def test_split_video_is_precise(sample_video: Path, tmp_path: Path) -> None:
    clips = split_video(sample_video, 2, tmp_path / "clips", encoder="cpu")
    assert len(clips) == 2
    assert get_video_duration(clips[0]) == pytest.approx(2, abs=0.15)
    assert get_video_duration(clips[1]) == pytest.approx(1, abs=0.15)


def test_get_video_resolution_reads_stream_dimensions(sample_video: Path) -> None:
    assert get_video_resolution(sample_video) == (640, 360)


def test_resolve_source_window_clamps_and_rejects_empty() -> None:
    assert resolve_source_window(10.0, None, None) == (0.0, 10.0)
    assert resolve_source_window(10.0, 2.0, 99.0) == (2.0, 10.0)
    with pytest.raises(ValueError):
        resolve_source_window(10.0, 8.0, 5.0)


def test_split_video_respects_source_window_and_reports_each_clip(
    sample_video: Path, tmp_path: Path,
) -> None:
    seen: list[Path] = []
    clips = split_video(
        sample_video, 1, tmp_path / "clips", encoder="cpu",
        source_start=1.0, source_end=3.0, on_clip=seen.append,
    )
    assert len(clips) == 2
    assert [Path(clip) for clip in clips] == seen
    assert get_video_duration(clips[0]) == pytest.approx(1, abs=0.15)


def test_process_video_renders_only_the_selected_windows(
    sample_video: Path, tmp_path: Path, monkeypatch,
) -> None:
    from src import pipeline

    monkeypatch.setattr(pipeline, "DATA_DIR", str(tmp_path))
    seen: list[Path] = []
    project_dir, clips = pipeline.process_video(
        uploaded_path=sample_video, vertical=True, encoder="cpu",
        export_quality="720p", encoding_speed="fast",
        clips_windows=[(0.0, 1.0), (2.0, 3.0)], on_clip=seen.append,
    )
    assert len(clips) == 2
    assert [Path(c) for c in clips] == seen
    assert get_video_duration(clips[0]) == pytest.approx(1, abs=0.25)
    assert sorted(p.name for p in (Path(project_dir) / "vertical").glob("*.mp4")) == [
        "clip_001.mp4", "clip_002.mp4",
    ]


def test_safe_folder_sanitises_and_falls_back() -> None:
    from src.pipeline import _safe_folder

    assert _safe_folder('Lex Fridman: clips/2024') == "Lex Fridman clips 2024"
    assert _safe_folder("   ") == "Clips"
    assert _safe_folder("A" * 200) == "A" * 80


def test_publish_to_folder_splits_clips_and_texts(tmp_path: Path) -> None:
    from src.pipeline import _publish_to_folder

    src = tmp_path / "vertical"
    src.mkdir()
    (src / "clip_001.mp4").write_bytes(b"v")
    (src / "clip_001.txt").write_text("Titre\n\nDescription\n\n#a #b", encoding="utf-8")
    (src / "clip_002.mp4").write_bytes(b"v")  # pas de .txt

    drive = tmp_path / "drive"
    root = _publish_to_folder(
        [src / "clip_001.mp4", src / "clip_002.mp4"], drive, "fr", "Cool Streamer", "2026-09-01 12h00",
    )
    assert root == drive / "FR" / "Cool Streamer" / "2026-09-01 12h00"
    assert (root / "clips" / "clip_001.mp4").is_file()
    assert (root / "clips" / "clip_002.mp4").is_file()
    assert (root / "textes" / "clip_001.txt").is_file()   # même nom que le clip
    assert not (root / "textes" / "clip_002.txt").exists()


def test_publish_to_folder_copies_only_the_given_subset(tmp_path: Path) -> None:
    """La copie vers le dossier se fait sur la sélection, plus automatiquement."""
    from src.pipeline import _publish_to_folder

    src = tmp_path / "vertical"
    src.mkdir()
    for name in ("clip_001", "clip_002", "clip_003"):
        (src / f"{name}.mp4").write_bytes(b"v")
        (src / f"{name}.txt").write_text("t", encoding="utf-8")

    drive = tmp_path / "drive"
    root = _publish_to_folder(
        [src / "clip_001.mp4", src / "clip_003.mp4"], drive, "EN", "Streamer", "2026-09-03 10h10",
    )
    assert root.parent.parent.name == "EN"
    assert sorted(p.name for p in (root / "clips").glob("*.mp4")) == ["clip_001.mp4", "clip_003.mp4"]
    assert not (root / "clips" / "clip_002.mp4").exists()


def test_segment_vertical_cuts_and_reframes_in_one_pass(
    sample_video: Path, tmp_path: Path,
) -> None:
    seen: list[Path] = []
    clips = segment_vertical(
        sample_video, tmp_path / "vertical",
        clip_length=1, window_start=0.0, window_end=3.0,
        encoder="cpu", quality="720p", background="blur", on_clip=seen.append,
    )
    assert len(clips) == 3
    assert [Path(clip) for clip in clips] == seen
    assert get_video_resolution(clips[0]) == (720, 1280)
    assert get_video_duration(clips[1]) == pytest.approx(1, abs=0.25)


def test_frame_size_matches_the_requested_aspect() -> None:
    from src.quality import frame_size

    assert frame_size("1080p", "9:16") == (1080, 1920)
    assert frame_size("1080p", "1:1") == (1080, 1080)
    assert frame_size("1080p", "4:5") == (1080, 1350)
    assert frame_size("1080p", "16:9") == (1920, 1080)
    assert frame_size("720p", "16:9") == (1280, 720)


def test_resize_clip_honours_the_aspect(sample_video: Path, tmp_path: Path) -> None:
    output = resize_clip_for_vertical(
        sample_video, tmp_path / "sq.mp4", encoder="cpu", quality="720p",
        aspect="1:1", background="black", start=0.0, duration=1.0,
    )
    assert get_video_resolution(output) == (720, 720)


def test_vertical_export_preserves_landscape_video(
    sample_video: Path, tmp_path: Path,
) -> None:
    output = resize_clip_for_vertical(
        sample_video, tmp_path / "vertical.mp4", encoder="cpu", quality="720p",
        background="blur",
    )
    assert get_video_resolution(output) == (720, 1280)


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
    assert get_video_resolution(output) == (2160, 3840)


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


def test_cuda_blur_filter_scales_on_gpu_but_composites_on_cpu() -> None:
    video_filter = _blur_background_filter(1080, 1920, cuda=True)
    assert video_filter.count("scale_cuda=") == 2   # fond réduit + premier plan
    assert "hwupload_cuda" in video_filter and "hwdownload" in video_filter
    assert "boxblur=9:2" in video_filter            # flou toujours logiciel
    assert "overlay=" in video_filter
    assert "overlay_cuda" not in video_filter       # jamais : zones vertes


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


def test_client_opts_reads_cookie_env(monkeypatch) -> None:
    from src.downloader import _client_opts

    for var in (
        "CLIP_CREATOR_YTDLP_COOKIES_BROWSER",
        "CLIP_CREATOR_YTDLP_COOKIES_FILE",
        "CLIP_CREATOR_YTDLP_PLAYER_CLIENT",
    ):
        monkeypatch.delenv(var, raising=False)
    assert _client_opts() == {}

    monkeypatch.setenv("CLIP_CREATOR_YTDLP_COOKIES_BROWSER", "Firefox")
    assert _client_opts()["cookiesfrombrowser"] == ("firefox", None, None, None)

    monkeypatch.setenv("CLIP_CREATOR_YTDLP_COOKIES_BROWSER", "chrome:Profile 1")
    monkeypatch.setenv("CLIP_CREATOR_YTDLP_PLAYER_CLIENT", "android, web")
    opts = _client_opts()
    assert opts["cookiesfrombrowser"][:2] == ("chrome", "Profile 1")
    assert opts["extractor_args"] == {"youtube": {"player_client": ["android", "web"]}}


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
