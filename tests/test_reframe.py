from src.reframe import (
    CropPath,
    centred_crop_path,
    crop_box,
    crop_path,
    reframe_available,
    write_sendcmd,
)


def test_crop_box_keeps_target_ratio_inside_source() -> None:
    # source 1920x1080 -> 9:16 crop is as tall as the source, narrower.
    assert crop_box(1920, 1080, 1080, 1920) == (608, 1080)
    # source 1080x1920 -> 16:9 crop is as wide as the source, shorter (~608, even).
    assert crop_box(1080, 1920, 1920, 1080) == (1080, 608)
    # already the right ratio -> whole frame.
    assert crop_box(1080, 1920, 1080, 1920) == (1080, 1920)


def test_reframe_available_is_bool() -> None:
    assert isinstance(reframe_available(), bool)


def test_centred_crop_path_is_a_single_centred_event() -> None:
    path = centred_crop_path(1920, 1080, 600, 1080)
    assert path.events == [(0.0, 660, 0)]
    assert (path.x0, path.y0) == (660, 0)


def test_crop_path_follows_the_face_and_stays_in_bounds() -> None:
    # Face drifts from left to right across the clip.
    track = [(t / 2, 0.2 + 0.06 * t) for t in range(0, 12)]
    path = crop_path(track, 1920, 1080, 608, 1080, t_start=0.0, t_end=6.0)
    xs = [x for _t, x, _y in path.events]
    assert path.events[0][0] == 0.0
    assert all(0 <= x <= 1920 - 608 for x in xs)
    assert xs[-1] > xs[0]  # a suivi le visage vers la droite
    assert all(y == 0 for _t, _x, y in path.events)


def test_crop_path_without_detections_stays_centred() -> None:
    track = [(t / 2, None) for t in range(6)]
    path = crop_path(track, 1920, 1080, 608, 1080)
    assert {x for _t, x, _y in path.events} == {(1920 - 608) // 2}


def test_write_sendcmd_format(tmp_path) -> None:
    path = CropPath(608, 1080, [(0.0, 100, 0), (0.2, 104, 0)])
    out = write_sendcmd(tmp_path / "reframe.cmd", path)
    assert out.read_text(encoding="utf-8") == (
        "0.000 crop x 100, crop y 0;\n0.200 crop x 104, crop y 0;\n"
    )
