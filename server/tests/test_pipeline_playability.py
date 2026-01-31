from typing import List

from server.pipeline import (
    NoteEvent,
    assign_chord,
    map_notes_to_tab,
    parse_tuning,
)


def _make_note(pitch: int, start: float = 0.0) -> NoteEvent:
    return NoteEvent(start=start, duration=0.5, pitch=pitch, velocity=80)


def test_identical_notes_keep_same_position():
    notes = [
        _make_note(64, start=0.0),
        _make_note(64, start=0.5),
        _make_note(64, start=1.0),
    ]
    tab_notes, _, _ = map_notes_to_tab(notes, "EADGBE", 0, hand_span=4)
    positions = {(note.string, note.fret) for note in tab_notes}
    assert len(positions) == 1, "La position doit rester identique pour les mêmes notes successives."


def test_riff_avoids_large_leaps():
    notes = [
        _make_note(55, start=0.0),
        _make_note(59, start=0.5),
        _make_note(62, start=1.0),
        _make_note(65, start=1.5),
    ]
    tab_notes, _, _ = map_notes_to_tab(notes, "EADGBE", 0, hand_span=4)
    frets = [note.fret for note in tab_notes]
    diffs = [abs(a - b) for a, b in zip(frets, frets[1:])]
    assert diffs and max(diffs) <= 7, "Le solveur ne devrait pas engager de sauts > 7 frettes."


def test_chord_span_stays_within_hand():
    notes = [
        _make_note(64),
        _make_note(67),
        _make_note(71),
        _make_note(76),
    ]
    open_strings, _ = parse_tuning("EADGBE", 0)
    assignment, _, _ = assign_chord(notes, open_strings, {}, hand_span=4)
    assert assignment, "L'accord devrait être assignable."
    frets = [assignment[idx][1] for idx in range(len(notes))]
    assert max(frets) - min(frets) <= 4, "La span des accords doit rester dans la main normale."


def test_fingering_debug_exports_candidates_and_handpos():
    notes = [
        _make_note(64, start=0.0),
        _make_note(67, start=0.5),
    ]
    _, _, playability = map_notes_to_tab(notes, "EADGBE", 0)
    events = playability.get("events") or []
    assert events, "Le debug doit contenir les événements."
    for event in events:
        assert "handPosBefore" in event
        assert "handPosAfter" in event
        assert isinstance(event.get("candidates"), list), "Chaque événement doit lister les candidats."
