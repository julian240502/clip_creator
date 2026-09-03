from pathlib import Path

from src.captions import (
    TEMPLATES,
    CaptionStyle,
    _ass_colour,
    _merge_tokens,
    build_ass,
    write_clip_captions,
)
from src.resizer import resize_clip_for_vertical, segment_vertical
from src.transcribe import Transcript, TranscriptSegment, Word
from src.video_splitter import get_video_resolution


def _transcript() -> Transcript:
    words = [
        Word(0.0, 0.4, " Les"),
        Word(0.4, 0.8, " M"),
        Word(0.8, 1.2, "&M's"),
        Word(1.2, 1.7, " Crispy"),
        Word(1.7, 2.0, " sont"),
        Word(2.0, 2.4, " là"),
        Word(2.4, 2.6, " !"),
        Word(2.9, 3.3, " Trop"),
        Word(3.3, 3.9, " bon"),
        Word(3.9, 4.1, " ?"),
    ]
    return Transcript(
        language="fr", duration=5.0, model="test",
        segments=[TranscriptSegment(0.0, 4.1, "Les M&M's Crispy sont là ! Trop bon ?", words)],
    )


def test_ass_colour_is_abgr_with_alpha() -> None:
    assert _ass_colour("#FFD400") == "&H0000D4FF"
    assert _ass_colour("#000000", alpha=160) == "&HA0000000"


def test_merge_tokens_glues_punctuation_and_clitics() -> None:
    merged = _merge_tokens([
        Word(0.0, 0.2, " M"), Word(0.2, 0.5, "&M's"), Word(0.5, 0.7, " sont"),
        Word(0.7, 0.9, " là"), Word(0.9, 1.0, " !"),
    ])
    assert [w.text for w in merged] == ["M&M's", "sont", "là!"]
    assert merged[0].end == 0.5


def test_build_ass_has_style_block_and_dialogue_events() -> None:
    ass = build_ass(
        _transcript(), clip_start=0.0, clip_end=5.0, width=1080, height=1920,
        style=TEMPLATES["Mot actif"],
    )
    assert "[V4+ Styles]" in ass
    assert "Style: Caption,Arial Black," in ass
    assert ass.count("Dialogue:") >= 5


def test_active_mode_scales_and_recolours_the_current_word() -> None:
    ass = build_ass(
        _transcript(), clip_start=0.0, clip_end=5.0, width=1080, height=1920,
        style=CaptionStyle(mode="active", highlight_color="#FF0000"),
    )
    assert "\\fscx112\\fscy112" in ass
    assert "\\c&H000000FF" in ass  # rouge en ABGR


def test_karaoke_mode_emits_kf_and_swaps_primary_colour() -> None:
    style = CaptionStyle(mode="karaoke", primary_color="#FFFFFF", highlight_color="#22D3EE")
    ass = build_ass(
        _transcript(), clip_start=0.0, clip_end=5.0, width=1080, height=1920, style=style,
    )
    assert "\\kf" in ass
    style_line = next(line for line in ass.splitlines() if line.startswith("Style: Caption"))
    fields = style_line.split(",")
    # En karaoké, la couleur de remplissage (PrimaryColour) = couleur du mot actif.
    assert fields[3] == _ass_colour("#22D3EE")
    assert fields[4] == _ass_colour("#FFFFFF")


def test_nudge_emits_a_pos_override_from_the_preset_anchor() -> None:
    ass = build_ass(
        _transcript(), clip_start=0.0, clip_end=5.0, width=1080, height=1920,
        style=CaptionStyle(mode="lines", position="bottom", margin_v=200, nudge_x=30, nudge_y=80),
    )
    # x = 1080/2 + 30 ; y = (1920 - 200) - 80
    assert "\\pos(570,1640)" in ass


def test_no_nudge_keeps_alignment_only() -> None:
    ass = build_ass(
        _transcript(), clip_start=0.0, clip_end=5.0, width=1080, height=1920,
        style=CaptionStyle(mode="lines", position="bottom"),
    )
    assert "\\pos(" not in ass
    assert "\\an2" in ass


def test_position_top_uses_alignment_8() -> None:
    ass = build_ass(
        _transcript(), clip_start=0.0, clip_end=5.0, width=1080, height=1920,
        style=CaptionStyle(mode="lines", position="top"),
    )
    style_line = next(line for line in ass.splitlines() if line.startswith("Style: Caption"))
    assert style_line.split(",")[18] == "8"
    assert "\\an8" in ass


def test_uppercase_transforms_text() -> None:
    ass = build_ass(
        _transcript(), clip_start=0.0, clip_end=5.0, width=1080, height=1920,
        style=CaptionStyle(mode="lines", uppercase=True),
    )
    assert "CRISPY" in ass and "Crispy" not in ass


def test_window_slice_rebases_timestamps() -> None:
    ass = build_ass(
        _transcript(), clip_start=2.9, clip_end=5.0, width=1080, height=1920,
        style=CaptionStyle(mode="lines"),
    )
    first_event = next(line for line in ass.splitlines() if line.startswith("Dialogue:"))
    assert first_event.split(",")[1] == "0:00:00.00"
    assert "Trop" in ass and "Crispy" not in ass


def test_build_ass_with_no_speech_has_no_dialogue() -> None:
    empty = Transcript(language="nn", duration=5.0, model="test", segments=[])
    ass = build_ass(
        empty, clip_start=0.0, clip_end=5.0, width=1080, height=1920,
        style=TEMPLATES["Mot actif"],
    )
    assert "[V4+ Styles]" in ass
    assert "Dialogue:" not in ass


def test_pipeline_skips_captions_when_no_speech(
    sample_video: Path, tmp_path: Path, monkeypatch,
) -> None:
    from src import pipeline
    from src import transcribe as transcribe_mod

    monkeypatch.setattr(
        transcribe_mod, "transcribe",
        lambda *a, **k: Transcript(language="nn", duration=3.0, model="x", segments=[]),
    )
    monkeypatch.setattr(pipeline, "DATA_DIR", str(tmp_path))
    project_dir, clips = pipeline.process_video(
        uploaded_path=sample_video, clip_length=2, vertical=True, encoder="cpu",
        export_quality="720p", encoding_speed="fast", captions_style=TEMPLATES["Mot actif"],
    )
    assert len(clips) >= 1
    assert (Path(project_dir) / "transcript.json").is_file()
    assert not list(Path(project_dir).rglob("*.ass"))


def test_segment_vertical_burns_one_ass_across_all_clips(
    sample_video: Path, tmp_path: Path,
) -> None:
    ass = write_clip_captions(
        _transcript(), tmp_path / "captions.ass",
        clip_start=0.0, clip_end=3.0, width=720, height=1280,
        style=TEMPLATES["Fondu"],
    )
    clips = segment_vertical(
        sample_video, tmp_path / "v",
        clip_length=1, window_start=0.0, window_end=3.0,
        encoder="cpu", quality="720p", background="black", captions_file=ass,
    )
    assert len(clips) == 3
    assert all(Path(clip).is_file() for clip in clips)


def test_captions_are_burned_into_the_vertical_export(sample_video: Path, tmp_path: Path) -> None:
    ass = write_clip_captions(
        _transcript(), tmp_path / "clip_001.ass",
        clip_start=0.0, clip_end=3.0, width=720, height=1280,
        style=TEMPLATES["Mot actif"],
    )
    output = resize_clip_for_vertical(
        sample_video, tmp_path / "clip_001.mp4", encoder="cpu", quality="720p",
        background="blur", start=0.0, duration=3.0, captions_file=ass,
    )
    assert output.is_file()
    assert get_video_resolution(output) == (720, 1280)
