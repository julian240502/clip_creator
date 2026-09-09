from pathlib import Path

import pytest

from src.audio_energy import excitement, loudness_curve

_HOP = 0.5


def test_excitement_is_low_on_a_flat_quiet_curve() -> None:
    curve = [0.1] * 200
    assert excitement(curve, _HOP, 10.0, 30.0) < 0.15


def test_excitement_spikes_on_a_loud_burst_inside_the_window() -> None:
    curve = [0.1] * 200
    for i in range(60, 68):          # ~3 s de fort à t≈30 s
        curve[i] = 0.95
    hot = excitement(curve, _HOP, 28.0, 40.0)
    cold = excitement(curve, _HOP, 5.0, 17.0)
    assert hot > 0.4 and hot > cold + 0.3


def test_excitement_rewards_a_sudden_silence_to_loud_jump() -> None:
    curve = [0.05] * 100 + [0.9] * 100          # saut net à t = 50 s
    assert excitement(curve, _HOP, 45.0, 55.0) > 0.4


def test_excitement_handles_empty_or_degenerate_input() -> None:
    assert excitement([], _HOP, 0.0, 10.0) == 0.0
    assert excitement([0.3] * 10, _HOP, 20.0, 10.0) == 0.0    # end <= start


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("numpy") is None, reason="numpy absent",
)
def test_loudness_curve_from_a_real_clip(sample_video: Path) -> None:
    out = loudness_curve(str(sample_video), hop=0.5)
    assert out is not None
    values, hop = out
    assert hop == 0.5
    assert 4 <= len(values) <= 8                 # ~3 s / 0.5
    assert all(0.0 <= v <= 1.0 for v in values)
