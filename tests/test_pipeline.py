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
    SplitLayout,
    _black_background_filter,
    _blur_background_filter,
    _split_layout_filter,
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


def test_download_source_clears_stale_fragments_before_retry(tmp_path: Path, monkeypatch) -> None:
    """Un téléchargement interrompu laisse des `.part` / `.part-FragNNNN` : on les
    purge avant de relancer, sinon yt-dlp reprend sur un état cassé."""
    import hashlib

    key = hashlib.sha1(b"https://host.test/v/1|720").hexdigest()[:16]
    bucket = tmp_path / key
    bucket.mkdir()
    (bucket / "vid.mp4.part").write_bytes(b"x")
    (bucket / "vid.mp4.part-Frag2189").write_bytes(b"x")

    def fake_download_video(video_url, output_dir, max_height=1080):
        # à ce stade les restes doivent avoir disparu
        assert not list(Path(output_dir).glob("*.part*"))
        target = Path(output_dir) / "vid.mp4"
        target.write_bytes(b"data" * 50)
        return str(target)

    monkeypatch.setattr(downloader, "download_video", fake_download_video)
    out = download_source("https://host.test/v/1", tmp_path, max_height=720)
    assert Path(out).name == "vid.mp4"


def test_download_video_outtmpl_avoids_the_title(monkeypatch, tmp_path: Path) -> None:
    """Le nom de fichier ne doit plus dépendre du titre (emoji / longueur → yt-dlp
    perd ses fragments sous Windows)."""
    captured: dict[str, dict] = {}

    class FakeYDL:
        def __init__(self, options):
            captured["options"] = options

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            (tmp_path / "v123.mp4").write_bytes(b"data" * 50)
            return {"id": "v123"}

    monkeypatch.setattr(downloader, "YoutubeDL", FakeYDL)
    downloader.download_video("https://host.test/v/1", tmp_path, max_height=720)
    tmpl = captured["options"]["outtmpl"]
    assert "%(title" not in tmpl and "%(id)s" in tmpl
    assert captured["options"]["restrictfilenames"] is True


def test_parse_hls_segments_extracts_init_and_segment_list() -> None:
    from src.downloader import _parse_hls_segments

    text = (
        "#EXTM3U\n"
        '#EXT-X-MAP:URI="init-0.mp4"\n'
        "#EXTINF:10.000,\n"
        "0-muted.mp4\n"
        "#EXTINF:10.000,\n"
        "1-muted.mp4\n"
        "#EXT-X-ENDLIST\n"
    )
    map_uri, segments = _parse_hls_segments(text, "https://cdn.test/path/")
    assert map_uri == "https://cdn.test/path/init-0.mp4"
    assert segments == [
        (10.0, "https://cdn.test/path/0-muted.mp4"),
        (10.0, "https://cdn.test/path/1-muted.mp4"),
    ]


def test_select_segments_pads_only_the_start_not_the_end() -> None:
    from src.downloader import _select_segments

    segments = [(10.0, f"seg{i}.mp4") for i in range(10)]  # couvre 0-100 s, 10 s/segment
    chosen, first_start = _select_segments(segments, start=70.0, end=76.0, pad=12.0)
    # start - pad = 58 -> segment [50,60) ; end = 76 -> segment [70,80), pas plus loin.
    assert [uri for _dur, uri in chosen] == ["seg5.mp4", "seg6.mp4", "seg7.mp4"]
    assert first_start == 50.0


def test_download_source_range_fetches_only_the_needed_hls_segments(
    monkeypatch, tmp_path: Path,
) -> None:
    """Manifeste fMP4 réaliste (#EXT-X-MAP) : seuls les quelques segments qui
    couvrent la fenêtre sont demandés, jamais le reste de la rediff — même
    quand la fenêtre est loin dedans."""
    from src.downloader import download_source_range

    playlist_text = (
        "#EXTM3U\n"
        '#EXT-X-MAP:URI="init-0.mp4"\n'
        + "".join(f"#EXTINF:10.000,\n{i}.mp4\n" for i in range(50))
        + "#EXT-X-ENDLIST\n"
    )

    class FakeProbeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            assert download is False  # sonde de métadonnées seulement
            return {"formats": [{"url": "https://cdn.test/vod/index-muted-X.m3u8", "height": 480}]}

    fetched_urls: list[str] = []

    def fake_fetch_text(url, timeout=20.0):
        fetched_urls.append(url)
        return playlist_text

    written_playlists: list[str] = []

    class FakeResult:
        returncode = 0
        stderr = ""

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        local_playlist = Path(cmd[cmd.index("-i") + 1])
        written_playlists.append(local_playlist.read_text(encoding="utf-8"))
        Path(cmd[-1]).write_bytes(b"x" * 150_000)
        return FakeResult()

    monkeypatch.setattr(downloader, "YoutubeDL", FakeProbeYDL)
    monkeypatch.setattr(downloader, "_fetch_text", fake_fetch_text)
    monkeypatch.setattr(downloader.subprocess, "run", fake_run)
    monkeypatch.setattr("src.video_splitter.get_video_duration", lambda path: 100.0)

    # Fenêtre loin dans une "rediff" de 500 s (50 segments x 10 s) : 300-306 s.
    media = download_source_range(
        "https://www.twitch.tv/videos/1", tmp_path, 300.0, 306.0, max_height=480,
    )
    assert Path(media).stat().st_size >= 100_000
    assert len(fetched_urls) == 1  # 1 seule lecture de la playlist distante

    listed = [
        line for line in written_playlists[0].splitlines()
        if line.startswith("https://cdn.test/vod/") and line.endswith(".mp4")
    ]
    assert listed == [
        "https://cdn.test/vod/28.mp4",
        "https://cdn.test/vod/29.mp4",
        "https://cdn.test/vod/30.mp4",
    ]  # pas les 50 segments de la rediff
    assert '#EXT-X-MAP:URI="https://cdn.test/vod/init-0.mp4"' in written_playlists[0]


def test_download_source_range_trims_hls_segments_to_the_exact_window(
    monkeypatch, tmp_path: Path,
) -> None:
    """Sans découpe, la playlist locale rend des segments ENTIERS (jusqu'à
    ~1 segment de trop à chaque bord) — get_video_duration dépasse alors la
    fenêtre demandée de bien plus que ce que le recalage lead/media_t0 des
    appelants (pipeline.py) tolère, d'où un décalage audio/sous-titres dans le
    clip final. Le résultat doit correspondre EXACTEMENT à [start, end]."""
    from src.downloader import download_source_range

    playlist_text = (
        "#EXTM3U\n" + "".join(f"#EXTINF:10.000,\n{i}.mp4\n" for i in range(50)) + "#EXT-X-ENDLIST\n"
    )

    class FakeProbeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            return {"formats": [{"url": "https://cdn.test/vod/index-X.m3u8", "height": 480}]}

    calls: list[list[str]] = []

    class FakeResult:
        returncode = 0
        stderr = ""

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"x" * 150_000)
        return FakeResult()

    monkeypatch.setattr(downloader, "YoutubeDL", FakeProbeYDL)
    monkeypatch.setattr(downloader, "_fetch_text", lambda url, timeout=20.0: playlist_text)
    monkeypatch.setattr(downloader.subprocess, "run", fake_run)
    monkeypatch.setattr("src.video_splitter.get_video_duration", lambda path: 100.0)

    # start=304 -> 1er segment choisi [290,300) (pad=12 -> lo=292) : first_start=290.
    download_source_range("https://www.twitch.tv/videos/1", tmp_path, 304.0, 308.0, max_height=480)
    assert len(calls) == 2  # 1 concat playlist (segments entiers) + 1 découpe locale exacte

    trim_cmd = calls[1]
    assert float(trim_cmd[trim_cmd.index("-ss") + 1]) == pytest.approx(304.0 - 290.0)
    assert float(trim_cmd[trim_cmd.index("-t") + 1]) == pytest.approx(4.0)


def test_download_clip_uses_segments_then_trims_to_the_exact_window(
    monkeypatch, tmp_path: Path,
) -> None:
    """Même bug de seek fMP4 pour l'aperçu ("Aperçu impossible : ... Output file
    does not contain any stream", plus de piste audio pour Whisper). Fetch par
    segments (large, à la granularité du segment près) PUIS découpe locale pour
    retomber sur EXACTEMENT [start, end] — l'aperçu en a besoin (~4 s précis)."""
    from src.downloader import download_clip

    playlist_text = (
        "#EXTM3U\n" + "".join(f"#EXTINF:10.000,\n{i}.mp4\n" for i in range(50)) + "#EXT-X-ENDLIST\n"
    )

    class FakeProbeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            return {"formats": [{"url": "https://cdn.test/vod/index-X.m3u8", "height": 480}]}

    calls: list[list[str]] = []

    class FakeResult:
        returncode = 0
        stderr = ""

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"x" * 150_000)
        return FakeResult()

    monkeypatch.setattr(downloader, "YoutubeDL", FakeProbeYDL)
    monkeypatch.setattr(downloader, "_fetch_text", lambda url, timeout=20.0: playlist_text)
    monkeypatch.setattr(downloader.subprocess, "run", fake_run)
    monkeypatch.setattr("src.video_splitter.get_video_duration", lambda path: 100.0)

    # start=304 -> 1er segment choisi est [290,300) (pad=12 -> lo=292) : first_start=290.
    media = download_clip("https://www.twitch.tv/videos/1", tmp_path, 304.0, 308.0, max_height=480)
    assert Path(media).name == "preview_source.mp4"
    assert Path(media).stat().st_size >= 100_000
    assert len(calls) == 2  # 1 concat playlist + 1 découpe locale précise

    trim_cmd = calls[1]
    assert float(trim_cmd[trim_cmd.index("-ss") + 1]) == pytest.approx(304.0 - 290.0)
    assert float(trim_cmd[trim_cmd.index("-t") + 1]) == pytest.approx(4.0)


def test_download_source_range_downloads_only_the_window(monkeypatch, tmp_path: Path) -> None:
    """Fenêtre analysée d'une rediff de 5 h : yt-dlp reçoit un download_ranges, et
    le résultat est mis en cache par URL + qualité + fenêtre."""
    from src.downloader import download_source_range

    captured: dict[str, dict] = {}

    class FakeYDL:
        def __init__(self, options):
            captured["options"] = options

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            out = Path(captured["options"]["outtmpl"].replace("%(id)s.%(ext)s", "v1.mp4"))
            out.write_bytes(b"data" * 30_000)  # au-dessus du seuil anti-flux-vide (100 Ko)
            return {"id": "v1"}

    monkeypatch.setattr(downloader, "YoutubeDL", FakeYDL)
    monkeypatch.setattr("src.video_splitter.get_video_duration", lambda path: 100.0)
    a = download_source_range("https://host.test/v/1", tmp_path, 12780.0, 14220.0, max_height=720)
    ranges = captured["options"]["download_ranges"](None, None)
    assert ranges == [{"start_time": 12780.0, "end_time": 14220.0}]
    # copie de flux, pas de ré-encodage forcé de toute la section
    assert "force_keyframes_at_cuts" not in captured["options"]
    # 2e appel, même fenêtre -> cache (pas de nouveau téléchargement)
    captured.clear()
    b = download_source_range("https://host.test/v/1", tmp_path, 12780.0, 14220.0, max_height=720)
    assert a == b and "options" not in captured
    # fenêtre différente -> autre bucket
    download_source_range("https://host.test/v/1", tmp_path, 0.0, 60.0, max_height=720)
    assert captured["options"]["download_ranges"](None, None)[0]["end_time"] == 60.0


def test_download_source_range_retries_with_a_smaller_window_on_a_near_empty_result(
    monkeypatch, tmp_path: Path,
) -> None:
    """La fin demandée déborde la fin réelle du flux (VOD tronqué en bout de
    rediff) : le 1er essai ne renvoie presque rien -> on recule la fin et on
    retente plutôt que de renvoyer un fichier inexploitable."""
    from src.downloader import download_source_range

    calls: list[dict] = []

    class FakeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            calls.append(self.options["download_ranges"](None, None)[0])
            out = Path(self.options["outtmpl"].replace("%(id)s.%(ext)s", "v1.mp4"))
            out.write_bytes(b"x" * (10 if len(calls) == 1 else 150_000))
            return {"id": "v1"}

    monkeypatch.setattr(downloader, "YoutubeDL", FakeYDL)
    monkeypatch.setattr("src.video_splitter.get_video_duration", lambda path: 100.0)
    media = download_source_range("https://host.test/v/2", tmp_path, 100.0, 200.0, max_height=480)
    assert Path(media).stat().st_size >= 100_000
    assert len(calls) == 2
    assert calls[0]["end_time"] == 200.0
    assert calls[1]["end_time"] == pytest.approx(192.0)  # -8 s, 2e essai


def test_download_source_range_raises_a_clear_error_when_the_stream_stays_truncated(
    monkeypatch, tmp_path: Path,
) -> None:
    from src.downloader import download_source_range

    class FakeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            out = Path(self.options["outtmpl"].replace("%(id)s.%(ext)s", "v1.mp4"))
            out.write_bytes(b"x" * 10)  # toujours quasi vide, quel que soit l'essai
            return {"id": "v1"}

    monkeypatch.setattr(downloader, "YoutubeDL", FakeYDL)
    with pytest.raises(RuntimeError, match="dépasse probablement"):
        download_source_range("https://host.test/v/3", tmp_path, 100.0, 130.0, max_height=480)


def test_download_source_range_falls_back_to_start_zero_on_a_muted_vod(
    monkeypatch, tmp_path: Path,
) -> None:
    """Playlist Twitch « muted » (musique sous droits) : ffmpeg ne peut sauter
    qu'à start=0 dans le flux distant, jamais à un instant non nul — quel que
    soit cet instant. Repli : télécharger [0, end] puis découper localement."""
    from src.downloader import download_source_range

    class FakeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            rng = self.options["download_ranges"](None, None)[0]
            out = Path(self.options["outtmpl"].replace("%(id)s.%(ext)s", "v1.mp4"))
            out.write_bytes(b"x" * (200_000 if rng["start_time"] == 0.0 else 10))
            return {"id": "v1"}

    class FakeResult:
        returncode = 0
        stderr = ""

    def fake_run(cmd, capture_output=True, text=True):
        Path(cmd[-1]).write_bytes(b"x" * 150_000)  # simule la découpe locale ffmpeg
        return FakeResult()

    monkeypatch.setattr(downloader, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(downloader.subprocess, "run", fake_run)
    monkeypatch.setattr("src.video_splitter.get_video_duration", lambda path: 100.0)

    media = download_source_range("https://host.test/v/5", tmp_path, 500.0, 560.0, max_height=480)
    assert Path(media).stat().st_size >= 100_000


def test_download_source_range_ignores_a_stale_near_empty_cached_file(
    monkeypatch, tmp_path: Path,
) -> None:
    """Un essai précédent raté avait laissé un fichier quasi vide en cache :
    il ne doit pas être renvoyé tel quel, ni bloquer un nouveau téléchargement."""
    from src.downloader import download_source_range

    url = "https://host.test/v/4"
    key = downloader.hashlib.sha1(f"{url}|480|100.0|200.0".encode()).hexdigest()[:16]
    bucket = tmp_path / f"range_{key}"
    bucket.mkdir(parents=True)
    (bucket / "v1.mp4").write_bytes(b"x" * 10)

    class FakeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            out = Path(self.options["outtmpl"].replace("%(id)s.%(ext)s", "v1.mp4"))
            out.write_bytes(b"x" * 150_000)
            return {"id": "v1"}

    monkeypatch.setattr(downloader, "YoutubeDL", FakeYDL)
    monkeypatch.setattr("src.video_splitter.get_video_duration", lambda path: 100.0)
    media = download_source_range(url, tmp_path, 100.0, 200.0, max_height=480)
    assert Path(media).stat().st_size >= 100_000


def test_is_valid_media_rejects_a_large_but_corrupt_file(tmp_path: Path) -> None:
    """Reported crash: "moov atom not found" on a file that had passed a
    size-only check. A truncated/corrupt file can easily clear the byte
    threshold while still being unreadable — the structural check must catch
    that, not just the size."""
    from src.downloader import _is_valid_media

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"x" * 150_000)  # bien au-dessus du seuil de taille
    assert not _is_valid_media(corrupt)


def test_fetch_range_via_segments_defers_to_direct_seek_for_a_large_plain_manifest(
    monkeypatch, tmp_path: Path,
) -> None:
    """Sans #EXT-X-MAP (segments .ts classiques, pas de bug de seek à
    contourner), une fenêtre qui demanderait trop de segments doit laisser la
    main au téléchargement direct (plus rapide, pas de risque de timeout)
    plutôt que de tout concaténer nous-mêmes."""
    from src.downloader import _fetch_range_via_segments

    # Playlist sans #EXT-X-MAP, fenêtre qui couvre plus de segments que la limite.
    n = 100
    playlist_text = "#EXTM3U\n" + "".join(f"#EXTINF:10.000,\n{i}.ts\n" for i in range(n)) + "#EXT-X-ENDLIST\n"

    class FakeProbeYDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            return {"formats": [{"url": "https://cdn.test/vod/index.m3u8", "height": 480}]}

    monkeypatch.setattr(downloader, "YoutubeDL", FakeProbeYDL)
    monkeypatch.setattr(downloader, "_fetch_text", lambda url, timeout=20.0: playlist_text)

    def boom(*_a, **_k):
        raise AssertionError("ne doit pas tenter de concaténer sans #EXT-X-MAP sur une grosse fenêtre")

    monkeypatch.setattr(downloader.subprocess, "run", boom)

    result = _fetch_range_via_segments(
        "https://www.twitch.tv/videos/1", tmp_path, 0.0, float(n * 10), 480,
    )
    assert result is None


def test_fetch_range_via_segments_cleans_up_the_output_on_timeout(
    monkeypatch, tmp_path: Path,
) -> None:
    """Un process tué par le timeout laisse un fichier tronqué — il ne doit
    jamais rester en cache pour tromper un appel suivant."""
    import subprocess as real_subprocess

    from src.downloader import _fetch_range_via_segments

    playlist_text = "#EXTM3U\n#EXTINF:10.000,\n0.ts\n#EXT-X-ENDLIST\n"

    class FakeProbeYDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            return {"formats": [{"url": "https://cdn.test/vod/index.m3u8", "height": 480}]}

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        Path(cmd[-1]).write_bytes(b"x" * 150_000)  # écriture partielle avant d'être "tué"
        raise real_subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(downloader, "YoutubeDL", FakeProbeYDL)
    monkeypatch.setattr(downloader, "_fetch_text", lambda url, timeout=20.0: playlist_text)
    monkeypatch.setattr(downloader.subprocess, "run", fake_run)

    with pytest.raises(real_subprocess.TimeoutExpired):
        _fetch_range_via_segments("https://www.twitch.tv/videos/1", tmp_path, 0.0, 10.0, 480)
    assert list(tmp_path.glob("*.mp4")) == []  # rien laissé derrière


def test_process_video_ranged_download_rebases_clip_windows(
    sample_video: Path, tmp_path: Path, monkeypatch,
) -> None:
    """URL + portion 1h00–1h00m10 : on télécharge la fenêtre (fichier 0-basé) et
    les clips_windows absolus sont ramenés dans ce référentiel local."""
    from src import pipeline

    seen: dict[str, object] = {}

    def fake_range(url, root, start, end, max_height=1080):
        seen["range"] = (start, end)
        return str(sample_video)

    def fake_full(url, root, max_height=1080):
        seen["full"] = True
        return str(sample_video)

    monkeypatch.setattr(pipeline, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr("src.downloader.download_source_range", fake_range)
    monkeypatch.setattr(pipeline, "download_source", fake_full)

    rendered: list[tuple[float, float]] = []
    real_resize = pipeline.resize_clip_for_vertical

    def spy_resize(*a, **k):
        rendered.append((k["start"], k["duration"]))
        return real_resize(*a, **k)

    monkeypatch.setattr(pipeline, "resize_clip_for_vertical", spy_resize)

    pipeline.process_video(
        url="https://host.test/v/1", vertical=True, encoder="cpu",
        export_quality="720p", encoding_speed="fast",
        source_start=3600.0, source_end=3610.0, source_duration=7200.0,
        clips_windows=[(3601.0, 3603.0)],
    )
    assert seen.get("range") == (3600.0, 3610.0) and "full" not in seen
    assert rendered and rendered[0][0] == pytest.approx(1.0, abs=0.05)  # 3601 - 3600


def test_process_video_smart_downloads_each_clip_window_not_the_whole_portion(
    tmp_path: Path, monkeypatch,
) -> None:
    """Sélection intelligente + transcript de l'analyse : un téléchargement par
    fenêtre de clip (± marge), pas toute l'heure analysée."""
    from src import pipeline
    from src.transcribe import Transcript, TranscriptSegment, Word

    monkeypatch.setattr(pipeline, "DATA_DIR", str(tmp_path))

    ranges: list[tuple[float, float]] = []

    def fake_range(url, root, start, end, max_height=1080):
        ranges.append((start, end))
        return str(tmp_path / "seg.mp4")

    monkeypatch.setattr("src.downloader.download_source_range", fake_range)
    monkeypatch.setattr(pipeline, "download_source", lambda *a, **k: pytest.fail("full DL"))
    monkeypatch.setattr(pipeline, "get_video_duration", lambda _p: 40.0)

    renders: list[tuple[float, float]] = []

    def fake_resize(media, output, **k):
        renders.append((k["start"], k["duration"]))
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"x")
        return Path(output)

    monkeypatch.setattr(pipeline, "resize_clip_for_vertical", fake_resize)

    tr = Transcript(
        language="en", duration=3700.0, model="x",
        segments=[TranscriptSegment(100.0, 101.0, "hi", [Word(100.0, 101.0, "hi")])],
    )
    _pd, clips = pipeline.process_video(
        url="https://host.test/v/1", vertical=True, encoder="cpu",
        export_quality="720p", encoding_speed="fast",
        source_start=600.0, source_end=4200.0, source_duration=46800.0,
        clips_windows=[(700.0, 730.0), (3000.0, 3040.0)],
        pretranscript=tr,
    )
    assert len(clips) == 2
    assert ranges == [(694.0, 736.0), (2994.0, 3046.0)]     # ± _CLIP_DL_PAD (6 s)
    assert len(renders) == 2


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


def test_split_layout_filter_stacks_two_cropped_panels() -> None:
    layout = SplitLayout(top=(0.30, 0.05, 0.35, 0.45), bottom=(0.0, 0.0, 1.0, 1.0), top_frac=0.40)
    f = _split_layout_filter(1080, 1920, layout)
    assert f.count("crop=") == 4          # 2 crops source (fractions) + 2 recadrages panneau
    assert "crop=iw*" in f and "ih*" in f  # fractions -> indépendant de la résolution
    assert "vstack" in f and f.endswith("[vout]")
    assert "scale=1080:768" in f          # panneau haut : _even(1920*0.40)
    assert "scale=1080:1152" in f         # panneau bas : 1920 - 768
    assert f.count("flags=lanczos") == 2  # agrandissement net sur les deux panneaux
    assert f.count("unsharp=") == 1       # léger renforcement, facecam uniquement


def test_split_background_needs_a_layout(sample_video: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resize_clip_for_vertical(
            sample_video, tmp_path / "split.mp4",
            encoder="cpu", background="split", start=0.0, duration=1.0,
        )


def test_process_video_split_layout_renders_vertical(
    sample_video: Path, tmp_path: Path, monkeypatch,
) -> None:
    from src import pipeline

    monkeypatch.setattr(pipeline, "DATA_DIR", str(tmp_path))
    layout = SplitLayout(top=(0.0, 0.0, 1.0, 0.5), top_frac=0.5)
    _project_dir, clips = pipeline.process_video(
        uploaded_path=sample_video, vertical=True, encoder="cpu",
        export_quality="720p", encoding_speed="fast",
        vertical_background="split", split_layout=layout,
        clips_windows=[(0.0, 1.0), (2.0, 3.0)],
    )
    assert len(clips) == 2
    assert get_video_resolution(clips[0]) == (720, 1280)


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
