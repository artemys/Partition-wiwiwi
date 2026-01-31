import numpy as np
from typing import Optional

from server.pipeline import NoteEvent, PitchTrackerResult, f0_to_note_events, post_process_notes


def _build_tracker(
    f0_values: np.ndarray,
    low_band_rms: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
) -> PitchTrackerResult:
    length = len(f0_values)
    low_band = (
        np.array(low_band_rms, dtype=np.float32)
        if low_band_rms is not None
        else np.ones(length, dtype=np.float32) * 0.1
    )
    conf = (
        np.array(confidence, dtype=np.float32)
        if confidence is not None
        else np.ones(length, dtype=np.float32)
    )
    return PitchTrackerResult(
        audio=np.zeros(length * 512, dtype=np.float32),
        sr=44100,
        hop_length=512,
        f0=np.array(f0_values, dtype=np.float32),
        voiced=np.ones(length, dtype=bool),
        confidence=conf,
        rms=np.ones(length, dtype=np.float32) * 0.1,
        low_band_rms=low_band,
    )


def test_f0_to_note_events_with_constant_pitch_emits_single_note():
    frames = 12
    tracker = _build_tracker(np.full(frames, 440.0, dtype=np.float32))
    notes, stats, _ = f0_to_note_events(tracker)

    assert len(notes) == 1
    first = notes[0]
    assert abs(first.pitch - 69) <= 1
    assert stats.notes_count == 1
    assert stats.voiced_ratio > 0.8


def test_f0_to_note_events_detects_two_notes_on_pitch_change():
    frames = 16
    values = np.array([440.0] * frames + [494.0] * frames, dtype=np.float32)
    tracker = _build_tracker(values)
    notes, stats, _ = f0_to_note_events(tracker)

    assert len(notes) == 2
    assert notes[1].start > notes[0].start
    assert notes[0].pitch != notes[1].pitch
    assert stats.notes_count == 2


def test_harmonic_dominance_prefers_lower_fundamental():
    frames = 20
    fundamental = np.array([440.0] * frames, dtype=np.float32)
    harmonic = np.array([880.0] * frames, dtype=np.float32)
    values = np.concatenate((fundamental, harmonic))
    low_energy = np.ones(len(values), dtype=np.float32) * 0.5
    tracker = _build_tracker(values, low_band_rms=low_energy)
    notes, stats, _ = f0_to_note_events(tracker)
    assert len(notes) == 1
    assert stats.notes_count == 1
    assert stats.polyphony_score < 0.5


def test_polyphonic_signal_increases_polyphony_score():
    frames = 24
    pattern = np.array([440.0, 554.0, 659.0, 880.0] * (frames // 4), dtype=np.float32)
    tracker = _build_tracker(pattern)
    _, stats, _ = f0_to_note_events(tracker)
    assert stats.polyphony_score > 0.5
    assert stats.note_change_rate > 0.0


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
