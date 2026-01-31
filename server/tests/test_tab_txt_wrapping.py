from server.pipeline import render_tab_txt_from_json


def test_tab_txt_wraps_wide_measures_with_divisions_8():
    # divisions=8 => 32 ticks/mesure, token_width=3 => 96 chars de payload (trop large pour 80)
    tab_json = {
        "metadata": {
            "divisions": 8,
            "tempo": 120,
            "tuning": ["E4", "B3", "G3", "D3", "A2", "E2"],
        },
        "measures": [
            {
                "number": 1,
                "events": [
                    {"startTick": 0, "durationTick": 1, "notes": [{"string": 1, "fret": 10}]},
                    {"startTick": 8, "durationTick": 1, "notes": [{"string": 2, "fret": 12}]},
                    {"startTick": 16, "durationTick": 1, "notes": [{"string": 3, "fret": 15}]},
                ],
            }
        ],
    }
    tab_txt, note_count, warnings = render_tab_txt_from_json(
        tab_json,
        quality="standard",
        measures_per_system=1,
        wrap_columns=80,
        token_width=3,
    )
    assert note_count == 3
    assert warnings == []
    for line in tab_txt.splitlines():
        if line.startswith(("e|", "B|", "G|", "D|", "A|", "E|")):
            assert len(line) <= 80

