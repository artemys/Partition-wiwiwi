from server.pipeline import (
    build_score_json,
    build_tab_json,
    map_notes_to_tab,
    post_process_basic_pitch_notes,
    quantize_basic_pitch_notes,
)


def _event_keys(measures):
    keys = set()
    for measure in measures:
        number = int(measure.get("number", 0))
        for ev in measure.get("events", []):
            keys.add((number, int(ev.get("startTick", 0))))
    return keys


def test_best_free_merges_notes_gap_lt_40ms():
    notes = [
        {"start": 0.00, "end": 0.20, "pitch": 60, "confidence": 0.9},
        {"start": 0.23, "end": 0.40, "pitch": 60, "confidence": 0.8},
    ]
    processed, stats = post_process_basic_pitch_notes(notes, confidence_threshold=0.35, lead_mode=False)
    assert stats["counts"]["afterMerge"] == 1
    assert len(processed) == 1
    assert processed[0]["pitch"] == 60
    assert processed[0]["start"] == 0.0
    assert abs(processed[0]["end"] - 0.40) < 1e-6


def test_best_free_drops_short_octave_harmonic_note():
    notes = [
        {"start": 0.00, "end": 0.20, "pitch": 60, "confidence": 0.95},
        {"start": 0.00, "end": 0.05, "pitch": 72, "confidence": 0.40},  # +12, très courte
    ]
    processed, _ = post_process_basic_pitch_notes(
        notes,
        confidence_threshold=0.35,
        min_duration_seconds=0.01,  # pour que la note courte ne soit pas filtrée avant l'anti-harmonique
        lead_mode=False,
    )
    pitches = sorted({int(n["pitch"]) for n in processed})
    assert pitches == [60]


def test_best_free_quantization_uses_provided_tempo():
    cleaned = [{"start": 0.12, "end": 0.62, "pitch": 60, "confidence": 0.9}]
    quantized = quantize_basic_pitch_notes(cleaned, tempo_bpm=60.0, divisions=8, grid_ticks=4)
    assert len(quantized) == 1
    assert abs(quantized[0].start - 0.0) < 1e-9
    assert abs(quantized[0].duration - 0.5) < 1e-9


def test_best_free_score_and_tab_event_alignment():
    # Notes déjà quantifiées (tempo=120, divisions=8 => spb=0.5, tick=0.0625s)
    note_dicts = [
        {"start": 0.00, "end": 0.50, "pitch": 64, "confidence": 0.9},
        {"start": 0.00, "end": 0.50, "pitch": 60, "confidence": 0.9},
        {"start": 0.50, "end": 1.00, "pitch": 62, "confidence": 0.9},
    ]
    notes = quantize_basic_pitch_notes(note_dicts, tempo_bpm=120.0, divisions=8, grid_ticks=2)

    score_json, _ = build_score_json(notes, quality="fast", tempo_bpm=120.0, tempo_source="beat_track", divisions=8)
    tab_notes, _, _ = map_notes_to_tab(notes, tuning="EADGBE", capo=0)
    tab_json, _ = build_tab_json(
        tab_notes,
        tuning="EADGBE",
        capo=0,
        quality="fast",
        tempo_bpm=120.0,
        tempo_source="beat_track",
        warnings=[],
        divisions=8,
    )

    assert _event_keys(score_json["measures"]) == _event_keys(tab_json["measures"])

