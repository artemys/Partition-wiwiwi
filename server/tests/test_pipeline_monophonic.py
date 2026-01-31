import numpy as np

from server.pipeline import NoteEvent, PitchTrackerResult, f0_to_note_events, post_process_notes


def _build_tracker(f0_values: np.ndarray) -> PitchTrackerResult:
    length = len(f0_values)
    return PitchTrackerResult(
        audio=np.zeros(length * 512, dtype=np.float32),
        sr=44100,
        hop_length=512,
        f0=np.array(f0_values, dtype=np.float32),
        voiced=np.ones(length, dtype=bool),
        confidence=np.ones(length, dtype=np.float32),
        rms=np.ones(length, dtype=np.float32) * 0.1,
    )


def test_f0_to_note_events_with_constant_pitch_emits_single_note():
    frames = 12
    tracker = _build_tracker(np.full(frames, 440.0, dtype=np.float32))
    notes, stats = f0_to_note_events(tracker)

    assert len(notes) == 1
    first = notes[0]
    assert abs(first.pitch - 69) <= 1
    assert stats.notes_count == 1
    assert stats.voiced_ratio > 0.8


def test_f0_to_note_events_detects_two_notes_on_pitch_change():
    frames = 16
    values = np.array([440.0] * frames + [494.0] * frames, dtype=np.float32)
    tracker = _build_tracker(values)
    notes, stats = f0_to_note_events(tracker)

    assert len(notes) == 2
    assert notes[1].start > notes[0].start
    assert notes[0].pitch != notes[1].pitch
    assert stats.notes_count == 2


def test_post_process_notes_quantizes_to_tempo_grid():
    events = [
        NoteEvent(start=0.1, duration=0.11, pitch=64, velocity=70),
        NoteEvent(start=0.52, duration=0.14, pitch=65, velocity=65),
    ]
    processed, _ = post_process_notes(events, "accurate", 120)

    grid = (60.0 / 120.0) / 4.0
    for note in processed:
        normalized_start = note.start / grid
        normalized_duration = note.duration / grid
        assert abs(normalized_start - round(normalized_start)) < 1e-6
        assert abs(normalized_duration - round(normalized_duration)) < 1e-6
