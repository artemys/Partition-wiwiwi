from pathlib import Path

from server.pipeline import (
    NoteEvent,
    _determine_written_octave_shift,
    post_process_notes,
    write_score_musicxml,
)


def test_post_process_notes_respects_tempo_grid():
    note_events = [NoteEvent(start=0.14, duration=0.2, pitch=64, velocity=80)]

    quantified_slow, _ = post_process_notes(note_events, "fast", 60)
    quantified_default, _ = post_process_notes(note_events, "fast", None)

    assert quantified_slow and quantified_default
    assert quantified_slow[0].start == 0.0
    assert quantified_default[0].start > quantified_slow[0].start


def test_score_written_octave_shift_applies_thresholds():
    low_notes = [NoteEvent(start=0, duration=0.2, pitch=45, velocity=70)]
    high_notes = [NoteEvent(start=0, duration=0.2, pitch=85, velocity=70)]

    assert _determine_written_octave_shift(low_notes) > 0
    assert _determine_written_octave_shift(high_notes) < 0


def test_write_score_musicxml_omits_technical_elements(tmp_path: Path):
    score_json = {
        "metadata": {
            "tempo": 120,
            "tempoSource": "midi",
            "timeSignature": "4/4",
            "divisions": 4,
            "quality": "fast",
            "scoreWrittenOctaveShift": 12,
        },
        "measures": [
            {
                "number": 1,
                "events": [
                    {"startTick": 0, "durationTick": 4, "tieStart": False, "tieStop": False, "notes": [{"midi": 64}]}
                ],
            }
        ],
    }
    destination = tmp_path / "test_score.musicxml"
    write_score_musicxml(
        score_json=score_json,
        tempo_bpm=120,
        title="Test",
        artist="Artist",
        annotations=None,
        output_path=str(destination),
        logger=None,
    )
    xml_content = destination.read_text(encoding="utf-8")
    assert "<technical>" not in xml_content
    assert "<string>" not in xml_content
    assert "<fret>" not in xml_content
