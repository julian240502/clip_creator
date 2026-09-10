import json

import src.twitch_chat as twitch_chat
from src.twitch_chat import _LAUGH_RE, chat_spikes, download_chat, is_twitch_vod


def _gql_page(rows: list[tuple[float, str]], *, has_next: bool) -> list:
    """Réponse GraphQL Twitch factice pour `VideoCommentsByOffsetOrCursor`."""
    return [{
        "data": {"video": {"comments": {
            "edges": [
                {"node": {
                    "contentOffsetSeconds": t,
                    "message": {"fragments": [{"text": text}]},
                }}
                for t, text in rows
            ],
            "pageInfo": {"hasNextPage": has_next},
        }}},
    }]


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


def test_download_chat_ignores_non_twitch_urls(tmp_path, monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError("l'API ne doit pas être appelée")

    monkeypatch.setattr(twitch_chat, "_gql", _boom)
    assert download_chat("https://youtube.com/watch?v=abc", tmp_path) is None


def test_download_chat_paginates_by_offset_and_dedups(tmp_path, monkeypatch) -> None:
    pages = [
        _gql_page([(10.0, "a"), (20.0, "b"), (30.0, "c")], has_next=True),
        _gql_page([(30.0, "c"), (40.0, "d"), (50.0, "e")], has_next=True),  # chevauche
        _gql_page([(50.0, "e"), (60.0, "f")], has_next=False),
    ]
    seen_vars: list[dict] = []

    def _fake_gql(body, *, timeout=20.0):
        seen_vars.append(body[0]["variables"])
        return pages[len(seen_vars) - 1]

    monkeypatch.setattr(twitch_chat, "_gql", _fake_gql)
    out = download_chat("https://www.twitch.tv/videos/123", tmp_path)

    assert out == [
        (10.0, "a"), (20.0, "b"), (30.0, "c"),
        (40.0, "d"), (50.0, "e"), (60.0, "f"),
    ]
    # pagination par offset croissant, jamais par cursor (déclenche l'anti-bot)
    assert all("cursor" not in v for v in seen_vars)
    assert [v["contentOffsetSeconds"] for v in seen_vars] == [0, 30, 50]
    assert (tmp_path / "chat_123.json").is_file()  # résultat complet -> mis en cache


def test_download_chat_clips_to_the_requested_window(tmp_path, monkeypatch) -> None:
    page = _gql_page(
        [(5.0, "avant"), (15.0, "dans1"), (25.0, "dans2"), (60.0, "apres")],
        has_next=False,
    )
    monkeypatch.setattr(twitch_chat, "_gql", lambda body, *, timeout=20.0: page)

    out = download_chat("https://www.twitch.tv/videos/123", tmp_path, start=10, end=30)
    assert out == [(15.0, "dans1"), (25.0, "dans2")]
    # collecte complète sur la fenêtre -> cache sous un nom qui porte la fenêtre
    assert (tmp_path / "chat_123_10-30.json").is_file()
    assert not (tmp_path / "chat_123.json").is_file()


def test_download_chat_returns_none_on_api_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        twitch_chat, "_gql",
        lambda body, *, timeout=20.0: [{"errors": [{"message": "PersistedQueryNotFound"}]}],
    )
    assert download_chat("https://www.twitch.tv/videos/123", tmp_path) is None


def test_download_chat_keeps_partial_sample_on_mid_stream_error(tmp_path, monkeypatch) -> None:
    pages = [
        _gql_page([(10.0, "a"), (20.0, "b")], has_next=True),
        [{"errors": [{"message": "IntegrityCheckFailed"}]}],
    ]
    calls: list[int] = []

    def _fake_gql(body, *, timeout=20.0):
        calls.append(1)
        return pages[len(calls) - 1]

    monkeypatch.setattr(twitch_chat, "_gql", _fake_gql)
    out = download_chat("https://www.twitch.tv/videos/123", tmp_path)
    assert out == [(10.0, "a"), (20.0, "b")]
    assert not (tmp_path / "chat_123.json").is_file()  # tronqué -> pas de cache
