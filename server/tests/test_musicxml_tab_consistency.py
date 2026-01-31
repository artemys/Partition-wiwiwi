from pathlib import Path

from server.pipeline import (
    TabNote,
    build_tab_json,
    compare_tab_json_and_musicxml,
    write_tab_musicxml,
)


def test_tab_json_and_generated_musicxml_have_same_note_tuples(tmp_path: Path):
    tab_notes = [
        TabNote(timeStart=0.0, duration=0.5, pitch=64, string=2, fret=5),
        TabNote(timeStart=0.5, duration=0.5, pitch=67, string=1, fret=3),
    ]
    tab_json, _ = build_tab_json(
        tab_notes=tab_notes,
        tuning="EADGBE",
        capo=0,
        quality="fast",
        tempo_bpm=120,
        tempo_source="midi",
        warnings=[],
        divisions=4,
    )

    destination = tmp_path / "tab.musicxml"
    write_tab_musicxml(
        tab_json=tab_json,
        tuning="EADGBE",
        capo=0,
        tempo_bpm=120,
        title="Test",
        artist="Test",
        annotations=None,
        output_path=str(destination),
        logger=None,
    )

    tab_count, xml_count, diffs = compare_tab_json_and_musicxml(tab_json, str(destination))
    assert tab_count == xml_count
    assert diffs == []

