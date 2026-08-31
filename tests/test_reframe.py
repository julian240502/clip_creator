from src.reframe import (
    centred_crop_path,
    crop_box,
    crop_path,
    reframe_available,
    write_sendcmd,
)


def test_crop_box_keeps_target_ratio_inside_source() -> None:
    assert crop_box(1920, 1080, 1080, 1920) == (608, 1080)
    assert crop_box(1080, 1920, 1920, 1080) == (1080, 608)
    assert crop_box(1080, 1920, 1080, 1920) == (1080, 1920)


def test_reframe_available_is_bool() -> None:
    assert isinstance(reframe_available(), bool)


def test_centred_crop_path_is_one_static_event() -> None:
    path = centred_crop_path(1920, 1080, 600, 1080)
    assert path.events == [(0.0, 660)]
    assert (path.x0, path.y0) == (660, 0)


def test_crop_path_follows_the_face_within_bounds() -> None:
    # Le visage dérive de gauche (0.25) vers la droite (0.75) sur 12 s.
    track = [(t / 2.0, 0.25 + 0.5 * (t / 22)) for t in range(0, 23)]
    path = crop_path(track, 1920, 1080, 608, 1080, t_start=0.0, t_end=11.0)
    xs = [x for _t, x in path.events]
    assert xs[-1] > xs[0]                          # a suivi vers la droite
    assert all(0 <= x <= 1920 - 608 for x in xs)
    assert path.events[0][0] == 0.0               # rebasé à t=0
    assert path.events[-1][0] >= 10.0             # couvre la fenêtre


def test_crop_path_is_dense_and_has_no_visible_jumps() -> None:
    noisy = [(t / 1.5, 0.5 + (0.06 if t % 2 else -0.06)) for t in range(0, 30)]
    path = crop_path(noisy, 1920, 1080, 608, 1080, t_start=0.0, t_end=18.0)
    xs = [x for _t, x in path.events]
    assert len(path.events) > 100                              # commandes denses
    assert max(abs(b - a) for a, b in zip(xs, xs[1:], strict=False)) <= 3  # pas d'escalier


def test_crop_path_without_detections_is_centred() -> None:
    track = [(t / 1.5, None) for t in range(8)]
    path = crop_path(track, 1920, 1080, 608, 1080)
    assert path.events == [(0.0, (1920 - 608) // 2)]


def test_write_sendcmd_format(tmp_path) -> None:
    path = centred_crop_path(1920, 1080, 608, 1080)
    out = write_sendcmd(tmp_path / "reframe.cmd", path)
    assert out.read_text(encoding="utf-8") == "0.000 crop x 656;\n"
