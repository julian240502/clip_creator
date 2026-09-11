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


def test_spikes_from_probes_flags_the_outlier_against_the_median() -> None:
    probes = [{"t": i * 100.0, "rate": 5.0, "laugh_frac": 0.0} for i in range(10)]
    probes[4] = {"t": 400.0, "rate": 40.0, "laugh_frac": 0.5}
    spikes = twitch_chat._spikes_from_probes(probes)
    assert len(spikes) == 1
    assert abs(spikes[0][0] - (400.0 - 3.5)) < 0.01  # recalé du délai (lag)


def test_spikes_from_probes_needs_at_least_five_probes() -> None:
    probes = [{"t": 0.0, "rate": 100.0, "laugh_frac": 1.0}] * 4
    assert twitch_chat._spikes_from_probes(probes) == []


def test_find_chat_spikes_uses_the_sequential_path_on_a_light_window(
    tmp_path, monkeypatch,
) -> None:
    """Petite fenêtre, chat calme : marche séquentielle habituelle, même résultat
    qu'appeler chat_spikes() à la main — pas de bascule en sondes inutile."""
    msgs = _steady(6, 20)
    msgs += [(603.0 + i * 0.05, "KEKW" if i % 2 else "LMAO") for i in range(40)]
    msgs.sort()
    page = _gql_page(msgs, has_next=False)
    calls: list[dict] = []

    def _fake_gql(body, *, timeout=20.0):
        calls.append(body[0]["variables"])
        return page

    monkeypatch.setattr(twitch_chat, "_gql", _fake_gql)
    spikes = twitch_chat.find_chat_spikes(
        "https://www.twitch.tv/videos/321", tmp_path, start=0, end=1200,
    )
    assert spikes == chat_spikes(msgs)
    assert all("cursor" not in v for v in calls)
    assert {v["contentOffsetSeconds"] for v in calls} == {0}  # un seul offset visité
    assert (tmp_path / "spikes_chat_321_0-1200.json").is_file()


def test_find_chat_spikes_samples_the_whole_window_when_chat_is_too_dense(
    tmp_path, monkeypatch,
) -> None:
    """Fenêtre de 8h, chat dense (mega-streamer) : marcher depuis le début
    n'atteindrait jamais la fin -> des sondes réparties sur TOUTE la fenêtre
    doivent quand même trouver le pic, loin du début."""
    window_end = 28800.0  # 8h
    target = 14400.0  # 4h, au milieu de la fenêtre
    calls: list[float] = []

    def _fake_gql(body, *, timeout=20.0):
        off = body[0]["variables"]["contentOffsetSeconds"]
        calls.append(off)
        near_target = abs(off - target) < 300
        n = 200 if near_target else 40
        step = 4.0 / n  # même durée de page (~4s) quel que soit le débit
        rows = [
            (off + i * step, "KEKW" if near_target and i % 3 == 0 else "hey")
            for i in range(n)
        ]
        return _gql_page(rows, has_next=True)

    monkeypatch.setattr(twitch_chat, "_gql", _fake_gql)
    spikes = twitch_chat.find_chat_spikes(
        "https://www.twitch.tv/videos/555", tmp_path, start=0, end=window_end,
    )
    assert spikes, "un pic net doit ressortir malgré un chat très dense"
    assert any(abs(t - target) < 600 for t, _ in spikes)
    # les sondes couvrent toute la fenêtre, pas seulement les premières minutes
    assert max(calls) > window_end * 0.5
    assert len(calls) < 200  # budget respecté (pas une marche séquentielle)


def test_find_chat_spikes_caches_and_skips_refetching(tmp_path, monkeypatch) -> None:
    page = _gql_page([(5.0, "hey"), (6.0, "hey")], has_next=False)
    calls = {"n": 0}

    def _fake_gql(body, *, timeout=20.0):
        calls["n"] += 1
        return page

    monkeypatch.setattr(twitch_chat, "_gql", _fake_gql)
    first = twitch_chat.find_chat_spikes(
        "https://www.twitch.tv/videos/777", tmp_path, start=0, end=100,
    )
    made_on_first_call = calls["n"]
    assert made_on_first_call > 0

    second = twitch_chat.find_chat_spikes(
        "https://www.twitch.tv/videos/777", tmp_path, start=0, end=100,
    )
    assert calls["n"] == made_on_first_call  # relu depuis le cache, pas de re-fetch
    assert second == first


def test_find_chat_spikes_ignores_non_twitch_urls(tmp_path, monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError("l'API ne doit pas être appelée")

    monkeypatch.setattr(twitch_chat, "_gql", _boom)
    assert twitch_chat.find_chat_spikes("https://youtube.com/watch?v=abc", tmp_path) is None
