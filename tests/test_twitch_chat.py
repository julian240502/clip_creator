import json

from src.twitch_chat import _LAUGH_RE, chat_spikes, download_chat, is_twitch_vod


def test_is_twitch_vod() -> None:
    assert is_twitch_vod("https://www.twitch.tv/videos/2868116605")
    assert is_twitch_vod("https://twitch.tv/videos/12345?t=1h2m3s")
    assert not is_twitch_vod("https://www.twitch.tv/somestreamer")       # live, pas VOD
    assert not is_twitch_vod("https://www.youtube.com/watch?v=abc")


def test_laugh_regex_catches_common_emotes() -> None:
    assert _LAUGH_RE.search("OMEGALUL that was insane")
    assert _LAUGH_RE.search("kekw")
    assert _LAUGH_RE.search("clip it")
    assert _LAUGH_RE.search("mdrrr")
    assert not _LAUGH_RE.search("that was a good play")


def _steady(rate: int, minutes: int) -> list[tuple[float, str]]:
    return [
        (m * 60 + s * (60 / rate), "nice play")
        for m in range(minutes) for s in range(rate)
    ]


def test_chat_spikes_ignores_a_steady_stream() -> None:
    assert chat_spikes(_steady(6, 20)) == []


def test_chat_spikes_flags_a_burst_and_corrects_for_lag() -> None:
    msgs = _steady(6, 20)
    # salve de ~40 messages autour de t = 605 s, dont beaucoup de KEKW
    msgs += [(603.0 + i * 0.05, "KEKW" if i % 2 else "LMAO") for i in range(40)]
    msgs.sort()

    spikes = chat_spikes(msgs, bucket=5.0, lag=3.5)
    assert spikes, "la salve doit produire un pic"
    t, intensity = spikes[0]
    # bucket 600-605 -> temps recalé de 3.5 s de délai chat
    assert 594.0 <= t <= 602.0
    assert intensity > 3.0                       # bien au-dessus de la base


def test_chat_spikes_weights_laughter_higher_than_plain_chatter() -> None:
    base = _steady(6, 20)
    burst_plain = base + [(603.0 + i * 0.1, "wow") for i in range(30)]
    burst_laugh = base + [(603.0 + i * 0.1, "KEKW") for i in range(30)]
    plain = chat_spikes(sorted(burst_plain))
    laugh = chat_spikes(sorted(burst_laugh))
    assert plain and laugh
    assert laugh[0][1] > plain[0][1]


def test_download_chat_reads_a_cached_file(tmp_path) -> None:
    url = "https://www.twitch.tv/videos/999"
    (tmp_path / "chat_999.json").write_text(
        json.dumps([[12.0, "hello"], [13.5, "KEKW"]]), encoding="utf-8",
    )
    out = download_chat(url, tmp_path)
    assert out == [(12.0, "hello"), (13.5, "KEKW")]
