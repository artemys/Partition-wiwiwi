from typing import Dict, List

from server.pipeline import render_tab_txt_from_json


def _make_tab_json(events: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "metadata": {
            "divisions": 4,
            "tempo": 120,
            "tuning": ["E4", "B3", "G3", "D3", "A2", "E2"],
        },
        "measures": [{"number": 1, "events": events}],
    }


def _chunk_line(value: str) -> List[str]:
    return [value[i : i + 3] for i in range(0, len(value), 3)]


def test_render_tab_txt_from_json_places_notes_in_expected_columns():
    events = [
        {"startTick": 0, "durationTick": 1, "notes": [{"string": 1, "fret": 3}]},
        {"startTick": 4, "durationTick": 1, "notes": [{"string": 3, "fret": 5}]},
        {"startTick": 8, "durationTick": 1, "notes": [{"string": 6, "fret": 7}]},
    ]
    tab_json = _make_tab_json(events)
    tab_txt, note_count, warnings = render_tab_txt_from_json(
        tab_json, quality="standard", measures_per_system=1
    )

    assert note_count == 3
    assert warnings == []
    prefixes = ("e|", "B|", "G|", "D|", "A|", "E|")
    lines = {
        line.split("|")[0]: line
        for line in tab_txt.splitlines()
        if line.startswith(prefixes)
    }

    e_cells = _chunk_line(lines["e"].split("|")[1])
    g_cells = _chunk_line(lines["G"].split("|")[1])
    E_cells = _chunk_line(lines["E"].split("|")[1])

    assert e_cells[0] == "3--"
    assert g_cells[4] == "5--"
    assert E_cells[8] == "7--"


def test_render_tab_txt_from_json_records_collision_warning():
    events = [
        {"startTick": 2, "durationTick": 1, "notes": [{"string": 1, "fret": 3}]},
        {"startTick": 2, "durationTick": 1, "notes": [{"string": 1, "fret": 5}]},
    ]
    tab_json = _make_tab_json(events)
    _, note_count, warnings = render_tab_txt_from_json(
        tab_json, quality="standard", measures_per_system=1
    )

    assert note_count == 1
    assert any("Collision" in warning for warning in warnings)
