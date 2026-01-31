import os
import bisect
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import mido
import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import wavfile

from .config import SETTINGS
from .utils import clamp, run_cmd, read_json, write_json


@dataclass
class NoteEvent:
    start: float
    duration: float
    pitch: int
    velocity: int


@dataclass
class TabNote:
    timeStart: float
    duration: float
    pitch: int
    string: int
    fret: int


@dataclass
class PitchTrackerResult:
    audio: np.ndarray
    sr: int
    hop_length: int
    f0: np.ndarray
    voiced: np.ndarray
    confidence: np.ndarray
    rms: np.ndarray
    low_band_rms: np.ndarray


@dataclass
class NoteExtractionStats:
    voiced_ratio: float
    instability_ratio: float
    pitch_median: Optional[float]
    pitch_min: Optional[float]
    pitch_max: Optional[float]
    notes_count: int
    note_change_rate: float
    jump_rate: float
    polyphony_score: float
    low_energy_ratio: float


@dataclass
class PitchTrackingDiagnostics:
    midi_smoothed: np.ndarray
    times: np.ndarray
    note_changes: int
    jump_events: int
    note_change_rate: float
    jump_rate: float
    low_energy_ratio: float


DEFAULT_DIVISIONS = 4
DEFAULT_TIME_SIGNATURE = "4/4"
GUITAR_WRITTEN_SHIFT = 12
LOW_PITCH_THRESHOLD = 55
HIGH_PITCH_THRESHOLD = 79

MAX_HAND_POS = 20
DEFAULT_SPAN_FRETS = 4
SHIFT_PENALTY = 2.0
OUTSIDE_SPAN_PENALTY = 5.0
STRING_JUMP_PENALTY = 0.5
HIGH_FRET_PENALTY = 0.35
HIGH_FRET_THRESHOLD = 15
LOW_FRET_BONUS = 0.3
LOW_FRET_THRESHOLD = 7
CHORD_SPAN_OVER_PENALTY = 8.0
CHORD_DISPERSION_PENALTY = 0.15
PLAYABILITY_COST_SCALE = 12.0


def _butter_bandpass(lowcut: float, highcut: float, sr: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyquist = 0.5 * sr
    low = max(lowcut / nyquist, 1e-6)
    high = min(highcut / nyquist, 0.999)
    if low >= high:
        high = min(low + 0.01, 0.999)
    return butter(order, [low, high], btype="band")


def _butter_lowpass(cutoff: float, sr: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyquist = 0.5 * sr
    value = min(max(cutoff / nyquist, 1e-6), 0.999)
    return butter(order, value, btype="low")


def _butter_highpass(cutoff: float, sr: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyquist = 0.5 * sr
    value = min(max(cutoff / nyquist, 1e-6), 0.999)
    return butter(order, value, btype="high")


def _apply_guitar_focus_filter(
    y: np.ndarray, sr: int, hp_hz: float = 80.0, lp_hz: float = 7000.0
) -> np.ndarray:
    """Filtre simple 'focus guitare' (HPF + LPF)."""
    if y.size == 0:
        return y
    try:
        hp_b, hp_a = _butter_highpass(hp_hz, sr)
        out = lfilter(hp_b, hp_a, y)
        lp_b, lp_a = _butter_lowpass(min(lp_hz, 0.499 * sr), sr)
        out = lfilter(lp_b, lp_a, out)
        return out
    except ValueError:
        return y


def _apply_guitar_bandpass(y: np.ndarray, sr: int) -> np.ndarray:
    """Filtre 'guitare focus'.

    Objectif: réduire le bas inutile et couper les très hautes fréquences tout en gardant
    les fondamentales (E2–E6) exploitables pour le pitch tracking.
    """
    if y.size == 0:
        return y
    try:
        # Bandpass large pour conserver les fondamentales et une partie des harmoniques.
        band_b, band_a = _butter_bandpass(80.0, min(0.499 * sr, 8000.0), sr)
        filtered = lfilter(band_b, band_a, y)
        # Lowpass de "propreté" (souvent redondant si la coupure bandpass est < cutoff).
        lp_b, lp_a = _butter_lowpass(min(0.499 * sr, 8000.0), sr)
        return lfilter(lp_b, lp_a, filtered)
    except ValueError:
        return y


def _rms_dbfs(y: np.ndarray) -> Optional[float]:
    if y.size == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    if not np.isfinite(rms) or rms <= 1e-12:
        return None
    return float(20.0 * np.log10(rms))


def _target_rms_from_dbfs(dbfs: float) -> float:
    return float(10 ** (float(dbfs) / 20.0))


def _write_wav_mono_int16(path: str, y: np.ndarray, sr: int) -> None:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        wavfile.write(path, sr, np.zeros((0,), dtype=np.int16))
        return
    y = np.clip(y, -1.0, 1.0)
    data = (y * 32767.0).astype(np.int16)
    wavfile.write(path, int(sr), data)


def preprocess_guitar_stem(
    input_wav: str,
    output_wav: str,
    logger=None,
    hp_hz: float = 80.0,
    lp_hz: float = 7000.0,
    target_rms_dbfs: float = -18.0,
    apply_gate: bool = False,
) -> Dict[str, Any]:
    """Pré-traitement simple pour 'focus guitare'.

    - HPF 80Hz
    - LPF 7kHz
    - normalisation RMS vers ~-18 dBFS
    - gate léger optionnel
    """
    import librosa

    y, sr = librosa.load(input_wav, sr=SETTINGS.sample_rate, mono=True)
    duration = float(y.size) / float(sr) if sr else 0.0
    rms_before = _rms_dbfs(y)
    y_f = _apply_guitar_focus_filter(y, sr, hp_hz=hp_hz, lp_hz=lp_hz)
    if apply_gate:
        y_f = _apply_noise_gate(y_f, threshold=0.01)
    y_f = _normalize_rms(y_f, target_rms=_target_rms_from_dbfs(target_rms_dbfs))
    rms_after = _rms_dbfs(y_f)
    _write_wav_mono_int16(output_wav, y_f, sr)
    info = {
        "durationSeconds": duration,
        "sampleRate": int(sr),
        "rmsBeforeDbfs": rms_before,
        "rmsAfterDbfs": rms_after,
        "highpassHz": float(hp_hz),
        "lowpassHz": float(lp_hz),
        "targetRmsDbfs": float(target_rms_dbfs),
        "gateApplied": bool(apply_gate),
    }
    if logger:
        logger.info(
            "Stem preprocess: duration=%.2fs sr=%s rmsBefore=%s rmsAfter=%s gate=%s",
            duration,
            sr,
            f"{rms_before:.2f}dBFS" if rms_before is not None else "n/a",
            f"{rms_after:.2f}dBFS" if rms_after is not None else "n/a",
            apply_gate,
        )
    return info


def _apply_onset_band(y: np.ndarray, sr: int) -> np.ndarray:
    """Bande centrée guitare (attaque/rythme) pour onset/beat tracking."""
    if y.size == 0:
        return y
    try:
        b, a = _butter_bandpass(80.0, min(0.499 * sr, 1200.0), sr)
        return lfilter(b, a, y)
    except ValueError:
        return y


def _apply_noise_gate(y: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    if y.size == 0:
        return y
    level = max(threshold, float(np.percentile(np.abs(y), 15)))
    mask = np.abs(y) >= level
    return y * mask


def _normalize_rms(y: np.ndarray, target_rms: float = 0.12, max_gain: float = 12.0) -> np.ndarray:
    """Normalisation RMS légère (utile pour onset + pyin)."""
    if y.size == 0:
        return y
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    if not np.isfinite(rms) or rms <= 1e-8:
        return y
    gain = float(target_rms / rms)
    gain = float(clamp(gain, 1.0 / max_gain, max_gain))
    out = y * gain
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out


def _velocity_from_rms(values: List[float], max_rms: float) -> int:
    if not values:
        return 64
    avg = float(np.mean(values))
    if max_rms <= 0:
        return 64
    scaled = avg / max_rms
    return int(clamp(round(scaled * 127), 24, 127))


def _hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def run_pitch_tracker(input_wav: str, logger=None) -> PitchTrackerResult:
    import librosa

    frame_length = 4096
    hop_length = 512
    y, sr = librosa.load(input_wav, sr=SETTINGS.sample_rate, mono=True)
    filtered = _apply_guitar_bandpass(y, sr)
    filtered = _apply_noise_gate(filtered)
    filtered = _normalize_rms(filtered)
    try:
        f0, voiced, voiced_confidence = librosa.pyin(
            filtered,
            fmin=82,
            fmax=1319,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Pitch tracking échoué: {exc}") from exc
    if f0 is None:
        raise RuntimeError("Pitch tracking n'a retourné aucune fréquence.")

    smoothed = np.copy(f0)
    median_window = 7
    half = median_window // 2
    for idx in range(len(smoothed)):
        if not voiced[idx] or not np.isfinite(f0[idx]):
            continue
        start = max(0, idx - half)
        end = min(len(smoothed), idx + half + 1)
        block = f0[start:end]
        block = block[np.isfinite(block)]
        if block.size:
            smoothed[idx] = np.median(block)
    np.clip(smoothed, 82.0, 1319.0, out=smoothed)

    ema_alpha = 0.25
    ema_smoothed = np.full_like(smoothed, np.nan)
    last_value = None
    for idx in range(len(smoothed)):
        if not voiced[idx] or not np.isfinite(smoothed[idx]):
            last_value = None
            continue
        if last_value is None:
            last_value = smoothed[idx]
        else:
            last_value = ema_alpha * smoothed[idx] + (1 - ema_alpha) * last_value
        ema_smoothed[idx] = last_value
    np.clip(ema_smoothed, 82.0, 1319.0, out=ema_smoothed)
    smoothed = ema_smoothed

    rms = librosa.feature.rms(y=filtered, frame_length=frame_length, hop_length=hop_length)[
        0
    ]
    try:
        low_b, low_a = _butter_bandpass(80.0, min(200.0, 0.499 * sr), sr)
        low_filtered = lfilter(low_b, low_a, filtered)
        low_band_rms = librosa.feature.rms(
            y=low_filtered, frame_length=frame_length, hop_length=hop_length
        )[0]
    except ValueError:
        low_band_rms = np.zeros_like(rms)

    if logger:
        voiced_ratio = float(np.count_nonzero(voiced)) / max(1, len(voiced))
        logger.info("Pitch tracking frames: %s, voiced ratio: %.2f", len(voiced), voiced_ratio)
    return PitchTrackerResult(
        audio=filtered,
        sr=sr,
        hop_length=hop_length,
        f0=smoothed,
        voiced=voiced.astype(bool),
        confidence=np.nan_to_num(
            np.asarray(voiced_confidence) if voiced_confidence is not None else np.zeros_like(voiced), nan=0.0
        ),
        rms=rms,
        low_band_rms=low_band_rms,
    )


def _track_midi_sequence(
    midi_series: np.ndarray,
    valid: np.ndarray,
    frame_duration: float,
    confidences: np.ndarray,
    low_energy_ratio: np.ndarray,
    change_threshold: float,
    stable_frames: int,
    jump_semitones: float = 7.0,
    jump_cooldown_seconds: float = 0.12,
    high_note_threshold: float = 88.0,
    high_note_confidence: float = 0.85,
    low_energy_strength: float = 0.25,
) -> Tuple[np.ndarray, PitchTrackingDiagnostics]:
    n_frames = len(midi_series)
    tracked = np.full_like(midi_series, np.nan)
    current_midi: Optional[float] = None
    pending_midi: Optional[float] = None
    pending_frames = 0
    note_changes = 0
    jump_events = 0
    jump_cooldown_frames = max(1, int(np.ceil(jump_cooldown_seconds / frame_duration)))
    jump_cooldown = 0
    harmonic_intervals = (12.0, 19.0)
    voiced_frames = int(np.count_nonzero(valid))
    for idx in range(n_frames):
        if not valid[idx] or not np.isfinite(midi_series[idx]):
            tracked[idx] = np.nan
            pending_midi = None
            pending_frames = 0
            continue
        candidate = midi_series[idx]
        if current_midi is None:
            current_midi = candidate
            tracked[idx] = candidate
            continue
        if jump_cooldown > 0:
            jump_cooldown -= 1
        is_high = candidate > high_note_threshold
        if is_high and confidences[idx] < high_note_confidence:
            candidate = current_midi
        is_harmonic = any(
            abs(candidate - (current_midi + interval)) <= 1.0 for interval in harmonic_intervals
        )
        has_strong_low = idx < len(low_energy_ratio) and low_energy_ratio[idx] >= low_energy_strength
        # Anti-harmoniques: si on "monte" vers une harmonique (souvent +12) et que le bas est présent,
        # on préfère conserver la fondamentale.
        if is_harmonic and has_strong_low:
            tracked[idx] = current_midi
            pending_midi = None
            pending_frames = 0
            continue
        if pending_midi is None or abs(candidate - pending_midi) > 0.5:
            pending_midi = candidate
            pending_frames = 1
        else:
            pending_frames += 1
        if pending_frames < max(stable_frames, 6):
            tracked[idx] = current_midi
            continue
        proposed = pending_midi
        if abs(proposed - current_midi) > jump_semitones:
            jump_events += 1
            jump_cooldown = jump_cooldown_frames
            pending_midi = None
            pending_frames = 0
            tracked[idx] = current_midi
            continue
        if proposed > high_note_threshold and confidences[idx] < high_note_confidence:
            tracked[idx] = current_midi
            pending_midi = None
            pending_frames = 0
            continue
        if abs(proposed - current_midi) > change_threshold:
            note_changes += 1
        current_midi = proposed
        pending_midi = None
        pending_frames = 0
        tracked[idx] = current_midi

    total_seconds = max(1e-6, float(voiced_frames) * frame_duration)
    note_change_rate = float(note_changes) / total_seconds
    jump_rate = float(jump_events) / max(1, note_changes) if note_changes else 0.0
    ratio_slice = low_energy_ratio[:n_frames] if low_energy_ratio.size >= n_frames else low_energy_ratio
    ratio_valid = ratio_slice[valid[: ratio_slice.size]]
    low_energy_val = float(np.nanmean(ratio_valid)) if ratio_valid.size else 0.0
    diagnostics = PitchTrackingDiagnostics(
        midi_smoothed=tracked,
        times=np.arange(n_frames, dtype=float) * frame_duration,
        note_changes=note_changes,
        jump_events=jump_events,
        note_change_rate=note_change_rate,
        jump_rate=jump_rate,
        low_energy_ratio=low_energy_val,
    )
    return tracked, diagnostics


def f0_to_note_events(
    tracker: PitchTrackerResult,
    min_duration_sec: float = 0.08,
    pitch_change_threshold: float = 0.5,
    stable_frames: int = 3,
    onset_frames: Optional[Iterable[int]] = None,
) -> Tuple[List[NoteEvent], NoteExtractionStats, PitchTrackingDiagnostics]:
    frame_duration = tracker.hop_length / tracker.sr if tracker.sr else 0.0
    frame_duration = max(frame_duration, 1e-4)
    min_frames = max(1, int(np.ceil(min_duration_sec / frame_duration)))
    valid = tracker.voiced & np.isfinite(tracker.f0)
    midi_series = np.full_like(tracker.f0, np.nan)
    midi_series[valid] = _hz_to_midi(tracker.f0[valid])

    low_ratio = np.zeros_like(tracker.f0)
    ratio_len = min(len(tracker.low_band_rms), len(tracker.rms), len(tracker.f0))
    if ratio_len > 0:
        energy = np.maximum(np.square(tracker.rms[:ratio_len]), 1e-8)
        low_energy = np.square(tracker.low_band_rms[:ratio_len])
        low_ratio[:ratio_len] = np.clip(low_energy / energy, 0.0, 1.0)

    tracked_midi, diagnostics = _track_midi_sequence(
        midi_series=midi_series,
        valid=valid,
        frame_duration=frame_duration,
        confidences=tracker.confidence,
        low_energy_ratio=low_ratio,
        change_threshold=pitch_change_threshold,
        stable_frames=stable_frames,
    )

    notes: List[NoteEvent] = []
    instability = 0
    max_rms = float(np.max(tracker.rms)) if tracker.rms.size else 0.0

    # Segmentation onset-based.
    # Pour les tests unitaires / déterminisme, on peut injecter explicitement les onsets.
    boundaries: List[int] = []
    if onset_frames is not None:
        try:
            boundaries = sorted({int(x) for x in onset_frames if x is not None})
        except TypeError:
            boundaries = []
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries
    end_frame = len(tracker.f0)
    boundaries = [b for b in boundaries if 0 <= b < end_frame]
    if not boundaries or boundaries[-1] != end_frame:
        boundaries.append(end_frame)

    from collections import Counter as _Counter

    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
        if end_idx <= start_idx:
            continue
        segment_valid = valid[start_idx:end_idx]
        if not np.any(segment_valid):
            continue
        segment_pitches = tracked_midi[start_idx:end_idx][segment_valid]
        segment_pitches = segment_pitches[np.isfinite(segment_pitches)]
        if segment_pitches.size == 0:
            continue
        rounded = [int(round(float(val))) for val in segment_pitches]
        counter = _Counter(rounded)
        pitch_value, _ = counter.most_common(1)[0]
        # Instabilité: on mesure sur la série "brute" (avant hysteresis) pour détecter
        # les signaux polyphoniques/instables même si la stabilisation bloque les changements.
        raw_slice = midi_series[start_idx:end_idx][segment_valid]
        raw_slice = raw_slice[np.isfinite(raw_slice)]
        if raw_slice.size:
            instability += int(np.count_nonzero(np.abs(raw_slice - float(pitch_value)) > 1.0))
        duration = (end_idx - start_idx) * frame_duration
        rms_slice = tracker.rms[start_idx:end_idx] if tracker.rms.size else np.array([], dtype=np.float32)
        velocity = _velocity_from_rms([float(x) for x in rms_slice.tolist()], max_rms)
        notes.append(
            NoteEvent(
                start=start_idx * frame_duration,
                duration=duration,
                pitch=int(clamp(pitch_value, 0, 127)),
                velocity=velocity,
            )
        )

    # Fusion de segments adjacents (onsets “trop fins”) si même pitch / gap négligeable.
    merged: List[NoteEvent] = []
    for note in sorted(notes, key=lambda n: n.start):
        if not merged:
            merged.append(note)
            continue
        prev = merged[-1]
        gap = note.start - (prev.start + prev.duration)
        if abs(note.pitch - prev.pitch) <= 0 and gap <= max(1e-6, frame_duration * 0.5):
            prev.duration += note.duration + max(0.0, gap)
            prev.velocity = int(round((prev.velocity + note.velocity) / 2))
        else:
            merged.append(note)
    notes = merged

    valid_midi = tracked_midi[np.isfinite(tracked_midi)]
    pitch_median = float(np.median(valid_midi)) if valid_midi.size else None
    pitch_min = float(np.min(valid_midi)) if valid_midi.size else None
    pitch_max = float(np.max(valid_midi)) if valid_midi.size else None
    voiced_frames = float(np.count_nonzero(valid))
    total_frames = float(max(1, len(tracker.f0)))
    voiced_ratio = voiced_frames / total_frames
    instability_ratio = instability / max(1, int(voiced_frames))
    non_voiced_ratio = 1.0 - voiced_ratio
    note_change_norm = min(1.0, diagnostics.note_change_rate / 4.0)
    # Score polyphonique: on donne plus de poids à l'instabilité "brute" (variations de pitch)
    # pour capturer les cas où l'hysteresis stabilise mais le signal reste polyphonique.
    polyphony_score = float(
        np.clip(
            0.75 * instability_ratio
            + 0.15 * note_change_norm
            + 0.10 * diagnostics.jump_rate
            + 0.05 * non_voiced_ratio,
            0.0,
            1.0,
        )
    )
    stats = NoteExtractionStats(
        voiced_ratio=voiced_ratio,
        instability_ratio=instability_ratio,
        pitch_median=pitch_median,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        notes_count=len(notes),
        note_change_rate=diagnostics.note_change_rate,
        jump_rate=diagnostics.jump_rate,
        polyphony_score=polyphony_score,
        low_energy_ratio=diagnostics.low_energy_ratio,
    )
    return notes, stats, diagnostics


def detect_onsets(tracker: PitchTrackerResult, logger=None) -> np.ndarray:
    """Détecte les onsets (frames) sur la piste guitare filtrée."""
    import librosa

    if tracker.audio.size == 0 or tracker.sr <= 0:
        return np.array([], dtype=int)
    y = _apply_onset_band(tracker.audio, tracker.sr)
    try:
        frames = librosa.onset.onset_detect(
            y=y,
            sr=tracker.sr,
            hop_length=tracker.hop_length,
            backtrack=False,
            units="frames",
        )
    except Exception:  # noqa: BLE001
        frames = np.array([], dtype=int)
    frames = np.asarray(frames, dtype=int)
    frames = frames[np.isfinite(frames)]
    frames = frames[(frames >= 0) & (frames < len(tracker.f0))]
    frames = np.unique(frames)
    if logger:
        logger.info("Onsets détectés: %d", int(frames.size))
    return frames


def detect_onsets_from_wav(
    wav_path: str,
    *,
    sr: int = SETTINGS.sample_rate,
    hop_length: int = 512,
    logger=None,
) -> List[float]:
    """Détecte les onsets (en secondes) sur un fichier wav.

    Utilisé pour le mode best_free (stem guitare), afin de faire un onset gating des notes BasicPitch.
    """
    import librosa

    try:
        y, loaded_sr = librosa.load(wav_path, sr=sr, mono=True)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("detect_onsets_from_wav: load failed: %s", exc)
        return []
    if y.size == 0 or loaded_sr <= 0:
        return []
    y = _apply_onset_band(y, loaded_sr)
    y = _apply_noise_gate(y, threshold=0.01)
    y = _normalize_rms(y)
    try:
        onset_times = librosa.onset.onset_detect(
            y=y,
            sr=loaded_sr,
            hop_length=hop_length,
            backtrack=False,
            units="time",
        )
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("detect_onsets_from_wav: onset_detect failed: %s", exc)
        return []
    times = [float(x) for x in np.asarray(onset_times, dtype=float).tolist() if np.isfinite(x)]
    times = sorted(set(t for t in times if t >= 0.0))
    if logger:
        logger.info("Onsets détectés (wav): %d", len(times))
    return times


def _nearest_onset_distance_seconds(onsets: List[float], t: float) -> Optional[float]:
    if not onsets:
        return None
    idx = bisect.bisect_left(onsets, t)
    best = None
    for j in (idx - 1, idx):
        if 0 <= j < len(onsets):
            dist = abs(float(onsets[j]) - float(t))
            if best is None or dist < best:
                best = dist
    return best


def _apply_onset_gating_to_note_dicts(
    notes: List[Dict[str, Any]],
    onsets: List[float],
    *,
    onset_window_ms: int,
    min_short_seconds: float = 0.12,
    confidence_soft_threshold: float = 0.45,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Bloque les nouveaux starts hors attaque.

    - Si le start n'est pas proche d'un onset (±window), on tente de fusionner avec la note précédente
      (si pitch proche), sinon on ignore si courte / faible confidence.
    """
    if not notes or not onsets:
        return notes, {"dropped": 0, "merged": 0, "kept": len(notes)}
    window_s = max(0.0, float(onset_window_ms)) / 1000.0
    window_s = min(window_s, 0.25)
    dropped = 0
    merged = 0
    kept: List[Dict[str, Any]] = []
    for n in sorted(notes, key=lambda x: (float(x.get("start", 0.0)), int(x.get("pitch", 0)))):
        start = float(n["start"])
        end = float(n["end"])
        pitch = int(n["pitch"])
        conf = float(n.get("confidence", 0.0))
        dur = max(0.0, end - start)
        dist = _nearest_onset_distance_seconds(onsets, start)
        near = dist is not None and dist <= window_s
        if near:
            kept.append(n)
            continue
        # Hors attaque: on limite les changements.
        if kept:
            prev = kept[-1]
            prev_pitch = int(prev["pitch"])
            if abs(pitch - prev_pitch) <= 1:
                prev["end"] = max(float(prev["end"]), end)
                prev["confidence"] = float(max(float(prev.get("confidence", 0.0)), conf))
                merged += 1
                continue
        if dur < min_short_seconds or conf < confidence_soft_threshold:
            dropped += 1
            continue
        # Cas “fort” mais off-onset: on préfère ignorer plutôt que créer du rythme illisible.
        dropped += 1
    return kept, {"dropped": dropped, "merged": merged, "kept": len(kept)}


def _extract_lead_from_note_dicts(
    notes: List[Dict[str, Any]],
    *,
    bin_ms: int = 30,
    max_jump_semitones: int = 7,
    max_jump_window_seconds: float = 0.12,
    medium_center_midi: int = 64,
) -> List[Dict[str, Any]]:
    """Sélectionne une seule note par bin temporel (monodie stable).

    Tie-breakers:
    - confidence max
    - pitch proche de la note précédente (minimise les sauts)
    - bonus registre médium (éviter les harmoniques très aigües)
    """
    if not notes:
        return []
    bin_s = max(0.02, min(0.05, float(bin_ms) / 1000.0))
    sorted_notes = sorted(notes, key=lambda x: (float(x["start"]), -float(x.get("confidence", 0.0))))
    # Group by time bin.
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for n in sorted_notes:
        key = int(round(float(n["start"]) / bin_s))
        groups.setdefault(key, []).append(n)
    lead: List[Dict[str, Any]] = []
    prev_pitch: Optional[int] = None
    prev_start: Optional[float] = None
    for key in sorted(groups.keys()):
        candidates = groups[key]
        if not candidates:
            continue
        best = None
        best_score = -1e9
        for cand in candidates:
            pitch = int(cand["pitch"])
            conf = float(cand.get("confidence", 0.0))
            # Base score: confidence.
            score = conf * 10.0
            # Bonus registre médium + pénalité aigu.
            score -= 0.10 * abs(pitch - medium_center_midi)
            if pitch >= 84:
                score -= 0.25 * (pitch - 83)
            # Continuité (voice-leading).
            if prev_pitch is not None and prev_start is not None:
                jump = abs(pitch - prev_pitch)
                score -= 0.25 * jump
                dt = float(cand["start"]) - float(prev_start)
                if dt < max_jump_window_seconds and jump > max_jump_semitones and conf < 0.85:
                    score -= 50.0  # quasi impossible
            if score > best_score:
                best_score = score
                best = cand
        if best is None:
            continue
        lead.append(best)
        prev_pitch = int(best["pitch"])
        prev_start = float(best["start"])
    # Resort and de-duplicate overlaps (garde la plus confiante si collision).
    lead.sort(key=lambda x: float(x["start"]))
    out: List[Dict[str, Any]] = []
    for n in lead:
        if not out:
            out.append(n)
            continue
        prev = out[-1]
        if abs(float(n["start"]) - float(prev["start"])) <= bin_s * 0.5:
            if float(n.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                out[-1] = n
        else:
            out.append(n)
    return out


def estimate_tempo(audio: np.ndarray, sr: int, logger=None) -> Tuple[float, str]:
    import librosa

    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, start_bpm=SETTINGS.default_tempo_bpm)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("Tempo detection failed: %s", exc)
        tempo = None
    if tempo and tempo > 0:
        if logger:
            logger.info("Beat tracking tempo: %.2f BPM", tempo)
        return float(tempo), "beat_track"
    if logger:
        logger.info("Beat tracking fallback to default tempo: %.2f BPM", SETTINGS.default_tempo_bpm)
    return float(SETTINGS.default_tempo_bpm), "default"


def estimate_tempo_from_wav(wav_path: str, logger=None) -> Tuple[float, str]:
    import librosa

    y, sr = librosa.load(wav_path, sr=SETTINGS.sample_rate, mono=True)
    return estimate_tempo(y, sr, logger)


@dataclass
class _MonophonicChoice:
    event_index: int
    string_idx: int
    fret: int
    hand_pos_before: int
    last_string_before: Optional[int]
    hand_pos_after: int
    candidate_cost: float


@dataclass
class _DPEntry:
    cost: float
    prev_state: Optional[Tuple[int, Optional[int]]]
    choice: Optional[_MonophonicChoice]


def parse_tuning(tuning: str, capo: int) -> Tuple[List[int], Optional[str]]:
    standard = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4
    tuning = (tuning or "").strip().upper()
    if tuning == "EADGBE":
        return [p + capo for p in standard], None
    note_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    notes: List[str] = []
    idx = 0
    while idx < len(tuning):
        char = tuning[idx]
        if char not in note_map:
            return [p + capo for p in standard], "Tuning invalide, retour à EADGBE."
        if idx + 1 < len(tuning) and tuning[idx + 1] in ("#", "B"):
            notes.append(char + tuning[idx + 1])
            idx += 2
        else:
            notes.append(char)
            idx += 1
    if len(notes) != 6:
        return [p + capo for p in standard], "Tuning invalide, retour à EADGBE."
    octaves = [2, 2, 3, 3, 3, 4]
    midi: List[int] = []
    for note, octave in zip(notes, octaves):
        semitone = note_map[note[0]]
        if len(note) == 2 and note[1] == "#":
            semitone += 1
        elif len(note) == 2 and note[1] == "B":
            semitone -= 1
        midi.append((octave + 1) * 12 + semitone + capo)
    return midi, None


def ensure_dependencies(require_youtube: bool = False) -> None:
    missing = []
    for tool in (
        SETTINGS.ffmpeg_path,
        SETTINGS.ffprobe_path,
        SETTINGS.basic_pitch_path,
        SETTINGS.demucs_path,
        SETTINGS.musescore_path,
    ):
        from shutil import which

        if which(tool) is None:
            missing.append(tool)
    if require_youtube:
        from shutil import which

        if which(SETTINGS.yt_dlp_path) is None:
            missing.append(SETTINGS.yt_dlp_path)
    if missing:
        raise RuntimeError(
            "Dépendances manquantes: "
            + ", ".join(missing)
            + ". Installez-les (ffmpeg, demucs, basic-pitch, yt-dlp si YouTube)."
        )


def ffprobe_duration(path: str) -> float:
    import subprocess

    result = subprocess.run(
        [
            SETTINGS.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Impossible de lire la durée audio.")
    return float(result.stdout.strip())


def convert_to_wav(input_path: str, output_path: str, logger) -> None:
    run_cmd(
        [
            SETTINGS.ffmpeg_path,
            "-y",
            "-nostdin",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            str(SETTINGS.sample_rate),
            "-sample_fmt",
            "s16",
            output_path,
        ],
        logger,
    )


def trim_wav(
    input_path: str,
    output_path: str,
    start_seconds: float,
    end_seconds: Optional[float],
    logger,
) -> None:
    args = [
        SETTINGS.ffmpeg_path,
        "-y",
        "-nostdin",
        "-ss",
        str(start_seconds),
    ]
    if end_seconds is not None:
        args += ["-to", str(end_seconds)]
    args += [
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        str(SETTINGS.sample_rate),
        "-sample_fmt",
        "s16",
        output_path,
    ]
    run_cmd(args, logger)


def download_youtube_audio(url: str, output_dir: str, logger) -> str:
    output_template = os.path.join(output_dir, "youtube.%(ext)s")
    run_cmd(
        [
            SETTINGS.yt_dlp_path,
            "-f",
            "bestaudio/best",
            "--no-playlist",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--output",
            output_template,
            url,
        ],
        logger,
    )
    wav_path = os.path.join(output_dir, "youtube.wav")
    if not os.path.exists(wav_path):
        raise RuntimeError("Téléchargement YouTube échoué.")
    return wav_path


def run_demucs(input_wav: str, output_dir: str, logger) -> str:
    run_cmd(
        [
            SETTINGS.demucs_path,
            "-n",
            "htdemucs",
            "-o",
            output_dir,
            input_wav,
        ],
        logger,
        timeout_seconds=SETTINGS.subprocess_timeout_seconds,
    )
    basename = os.path.splitext(os.path.basename(input_wav))[0]
    other = os.path.join(output_dir, "htdemucs", basename, "other.wav")
    if not os.path.exists(other):
        raise RuntimeError("Demucs n'a pas produit other.wav.")
    return other


def run_demucs_pick_stem(input_wav: str, output_dir: str, logger, preferred: str = "other") -> Tuple[str, str]:
    """Lance Demucs et retourne un stem choisi (chemin + nom).

    Par défaut on prend `other.wav` (souvent le plus proche 'guitare').
    """
    run_cmd(
        [
            SETTINGS.demucs_path,
            "-n",
            "htdemucs",
            "-o",
            output_dir,
            input_wav,
        ],
        logger,
        timeout_seconds=SETTINGS.subprocess_timeout_seconds,
    )
    basename = os.path.splitext(os.path.basename(input_wav))[0]
    stem_dir = os.path.join(output_dir, "htdemucs", basename)
    candidates = [preferred, "other", "drums", "bass", "vocals"]
    for stem_name in candidates:
        stem_path = os.path.join(stem_dir, f"{stem_name}.wav")
        if os.path.exists(stem_path):
            return stem_path, stem_name
    raise RuntimeError("Demucs n'a produit aucun stem utilisable.")


def run_basic_pitch(input_wav: str, output_dir: str, logger) -> str:
    run_cmd(
        [
            SETTINGS.basic_pitch_path,
            output_dir,
            input_wav,
        ],
        logger,
        timeout_seconds=SETTINGS.subprocess_timeout_seconds,
    )
    for name in os.listdir(output_dir):
        if name.endswith(".mid") or name.endswith(".midi"):
            return os.path.join(output_dir, name)
    raise RuntimeError("Basic Pitch n'a pas généré de MIDI.")


def run_basic_pitch_notes(
    input_wav: str,
    output_dir: str,
    logger=None,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length_ms: float = 50.0,
    min_frequency: float = 80.0,
    max_frequency: float = 1400.0,
) -> Tuple[List[Dict[str, Any]], str]:
    """Extraction Basic Pitch (API Python) avec amplitude (proxy confidence) + export MIDI."""
    from basic_pitch.inference import predict

    os.makedirs(output_dir, exist_ok=True)
    _, midi_data, note_events = predict(
        input_wav,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_ms,
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        melodia_trick=True,
    )
    midi_path = os.path.join(output_dir, "basic_pitch.mid")
    midi_data.write(midi_path)

    raw_notes: List[Dict[str, Any]] = []
    for start_s, end_s, pitch_midi, amplitude, _pitch_bends in note_events:
        raw_notes.append(
            {
                "start": float(start_s),
                "end": float(end_s),
                "pitch": int(pitch_midi),
                "amplitude": float(amplitude),
                "confidence": float(amplitude),
            }
        )
    if logger:
        logger.info("BasicPitch notes (python): %d", len(raw_notes))
    return raw_notes, midi_path


def load_midi_notes(midi_path: str) -> Tuple[List[NoteEvent], Optional[float]]:
    midi = mido.MidiFile(midi_path)
    tempo = 500000
    ticks_per_beat = midi.ticks_per_beat
    events: List[NoteEvent] = []
    note_on: Dict[Tuple[int, int], Tuple[float, int]] = {}
    current_time = 0.0
    for msg in mido.merge_tracks(midi.tracks):
        current_time += mido.tick2second(msg.time, ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        if msg.type == "note_on" and msg.velocity > 0:
            note_on[(msg.channel, msg.note)] = (current_time, msg.velocity)
        elif msg.type in ("note_off", "note_on"):
            key = (msg.channel, msg.note)
            if key in note_on:
                start, velocity = note_on.pop(key)
                duration = max(0.01, current_time - start)
                events.append(NoteEvent(start=start, duration=duration, pitch=msg.note, velocity=velocity))
    bpm = None
    if tempo:
        bpm = 60_000_000 / tempo
    return events, bpm


def post_process_basic_pitch_notes(
    notes: List[Dict[str, Any]],
    *,
    confidence_threshold: float = 0.35,
    midi_min: int = 40,
    midi_max: int = 88,
    merge_gap_seconds: float = 0.04,
    min_duration_seconds: float = 0.08,
    harmonic_window_seconds: float = 0.2,
    harmonic_interval_semitones: int = 12,
    lead_mode: bool = False,
    onsets: Optional[List[float]] = None,
    onset_window_ms: int = 60,
    max_jump_semitones: int = 7,
    lead_bin_ms: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Nettoyage 'musical' des notes Basic Pitch (start/end/pitch/confidence)."""
    if not notes:
        return [], {
            "counts": {
                "raw": 0,
                "afterFilter": 0,
                "afterOnsetGate": 0,
                "afterMerge": 0,
                "afterHarmonics": 0,
                "afterJumpFilter": 0,
                "afterLead": 0,
            }
        }

    raw_count = len(notes)

    # 3.1 Filtrage confidence + plage guitare.
    filtered: List[Dict[str, Any]] = []
    for n in notes:
        try:
            conf = float(n.get("confidence", n.get("amplitude", 0.0)) or 0.0)
            pitch = int(n.get("pitch"))
            start = float(n.get("start"))
            end = float(n.get("end"))
        except (TypeError, ValueError):
            continue
        if conf < confidence_threshold:
            continue
        if pitch < midi_min or pitch > midi_max:
            continue
        if end <= start:
            continue
        filtered.append({"start": start, "end": end, "pitch": pitch, "confidence": conf})
    filtered.sort(key=lambda x: (x["start"], x["pitch"]))
    after_filter = len(filtered)

    # 3.1b Onset gating (optionnel).
    gated = filtered
    onset_debug = {"dropped": 0, "merged": 0, "kept": len(filtered)}
    if onsets:
        gated, onset_debug = _apply_onset_gating_to_note_dicts(
            gated,
            list(onsets),
            onset_window_ms=int(onset_window_ms),
            confidence_soft_threshold=max(float(confidence_threshold) + 0.05, 0.45),
        )
    after_onset_gate = len(gated)

    # 3.2 Fusion notes consécutives (même pitch +/-1, gap < 40ms).
    merged: List[Dict[str, Any]] = []
    for n in gated:
        if not merged:
            merged.append(n)
            continue
        prev = merged[-1]
        gap = float(n["start"]) - float(prev["end"])
        if abs(int(n["pitch"]) - int(prev["pitch"])) <= 1 and gap >= -1e-6 and gap < merge_gap_seconds:
            prev["end"] = max(float(prev["end"]), float(n["end"]))
            prev["confidence"] = float(max(prev["confidence"], n["confidence"]))
            if int(n["pitch"]) < int(prev["pitch"]):
                prev["pitch"] = int(n["pitch"])
        else:
            merged.append(n)
    after_merge = len(merged)

    # 3.3 Anti-harmoniques / notes fantômes: octave courte et moins confiante.
    kept = [True] * len(merged)
    for i, hi in enumerate(merged):
        if not kept[i]:
            continue
        hi_pitch = int(hi["pitch"])
        hi_conf = float(hi["confidence"])
        hi_dur = float(hi["end"]) - float(hi["start"])
        # On cible surtout les notes courtes (parasites).
        if hi_dur >= 0.08:
            continue
        for j, lo in enumerate(merged):
            if i == j:
                continue
            lo_pitch = int(lo["pitch"])
            if abs((hi_pitch - lo_pitch) - harmonic_interval_semitones) > 1:
                continue
            if abs(float(hi["start"]) - float(lo["start"])) > harmonic_window_seconds:
                continue
            overlap_start = max(float(hi["start"]), float(lo["start"]))
            overlap_end = min(float(hi["end"]), float(lo["end"]))
            if overlap_end <= overlap_start:
                continue
            lo_conf = float(lo["confidence"])
            lo_dur = float(lo["end"]) - float(lo["start"])
            if hi_conf < lo_conf and hi_dur < lo_dur:
                kept[i] = False
                break
    after_harm = [n for ok, n in zip(kept, merged) if ok]
    after_harmonics = len(after_harm)

    # Anti-harmoniques (séquentiel): si +12, très court et moins confiant que la note précédente -> drop.
    seq_filtered: List[Dict[str, Any]] = []
    last_kept: Optional[Dict[str, Any]] = None
    for n in sorted(after_harm, key=lambda x: float(x["start"])):
        if last_kept is None:
            seq_filtered.append(n)
            last_kept = n
            continue
        dur = float(n["end"]) - float(n["start"])
        interval = int(n["pitch"]) - int(last_kept["pitch"])
        if abs(interval - harmonic_interval_semitones) <= 1 and dur < 0.12 and float(n["confidence"]) < float(
            last_kept["confidence"]
        ):
            continue
        seq_filtered.append(n)
        last_kept = n
    after_harm = seq_filtered
    after_harmonics = len(after_harm)

    # Filtre durée minimale (après anti-harmoniques).
    after_harm = [
        n for n in after_harm if (float(n["end"]) - float(n["start"])) >= float(min_duration_seconds)
    ]
    after_harmonics = len(after_harm)

    final = after_harm

    # 3.4 Lead mode (optionnel): monodie stable + voice-leading.
    if lead_mode and final:
        final = _extract_lead_from_note_dicts(
            final,
            bin_ms=int(lead_bin_ms),
            max_jump_semitones=int(max_jump_semitones),
        )

    # 3.5 Limiter les sauts (monodie): si jump énorme en très peu de temps, ignorer sauf forte confidence.
    jump_filtered: List[Dict[str, Any]] = []
    last_kept = None
    for n in sorted(final, key=lambda x: float(x["start"])):
        if last_kept is None:
            jump_filtered.append(n)
            last_kept = n
            continue
        dt = float(n["start"]) - float(last_kept["start"])
        jump = abs(int(n["pitch"]) - int(last_kept["pitch"]))
        if dt < 0.12 and jump > int(max_jump_semitones) and float(n["confidence"]) < 0.85:
            continue
        jump_filtered.append(n)
        last_kept = n
    final = jump_filtered
    after_jump = len(final)
    after_lead = len(final)

    stats = {
        "counts": {
            "raw": raw_count,
            "afterFilter": after_filter,
            "afterOnsetGate": after_onset_gate,
            "afterMerge": after_merge,
            "afterHarmonics": after_harmonics,
            "afterJumpFilter": after_jump,
            "afterLead": after_lead,
        },
        "params": {
            "confidenceThreshold": float(confidence_threshold),
            "mergeGapMs": int(round(merge_gap_seconds * 1000.0)),
            "minDurationMs": int(round(min_duration_seconds * 1000.0)),
            "harmonicWindowMs": int(round(harmonic_window_seconds * 1000.0)),
            "leadMode": bool(lead_mode),
            "onsetWindowMs": int(onset_window_ms),
            "maxJumpSemitones": int(max_jump_semitones),
            "leadBinMs": int(lead_bin_ms),
            "onsetGating": onset_debug,
        },
    }
    return final, stats


def quantize_basic_pitch_notes(
    notes: List[Dict[str, Any]],
    *,
    tempo_bpm: float,
    divisions: int = 8,
    grid_ticks: int = 4,
    min_duration_ticks: int = 1,
) -> List[NoteEvent]:
    """Quantifie start/end sur une grille en ticks, puis reconvertit en secondes."""
    if not notes:
        return []
    tempo = float(tempo_bpm or SETTINGS.default_tempo_bpm)
    if tempo <= 0:
        tempo = float(SETTINGS.default_tempo_bpm)
    seconds_per_beat = 60.0 / tempo
    divisions = max(1, int(divisions))
    grid_ticks = max(1, int(grid_ticks))
    min_duration_ticks = max(1, int(min_duration_ticks))

    out: List[NoteEvent] = []
    for n in notes:
        start_s = float(n["start"])
        end_s = float(n["end"])
        pitch = int(n["pitch"])
        conf = float(n.get("confidence", 0.0))
        start_tick = int(round(start_s / seconds_per_beat * divisions))
        end_tick = int(round(end_s / seconds_per_beat * divisions))
        start_tick = int(round(start_tick / grid_ticks) * grid_ticks)
        end_tick = int(round(end_tick / grid_ticks) * grid_ticks)
        end_tick = max(end_tick, start_tick + min_duration_ticks)
        duration_tick = end_tick - start_tick
        start_q = float(start_tick) / float(divisions) * seconds_per_beat
        dur_q = float(duration_tick) / float(divisions) * seconds_per_beat
        velocity = int(clamp(round(conf * 127.0), 24, 127))
        out.append(NoteEvent(start=float(start_q), duration=float(dur_q), pitch=pitch, velocity=velocity))
    out.sort(key=lambda ev: (ev.start, ev.pitch))
    return out


def quantize_basic_pitch_notes_robust(
    notes: List[Dict[str, Any]],
    *,
    tempo_bpm: float,
    divisions: int = 8,
    grid_ticks: int = 4,
    min_duration_ticks: int = 1,
    allowed_duration_ticks: Optional[List[int]] = None,
) -> Tuple[List[NoteEvent], Dict[str, Any]]:
    """Quantization robuste + compteur d'erreurs.

    - snap start/end sur une grille (grid_ticks)
    - évite des durées "impossibles" en les rabattant sur une liste autorisée
      (ex: {1/16, 1/8, 1/4, 1/2, 1})
    """
    if not notes:
        return [], {"quantizationErrorsCount": 0}
    tempo = float(tempo_bpm or SETTINGS.default_tempo_bpm)
    if tempo <= 0:
        tempo = float(SETTINGS.default_tempo_bpm)
    seconds_per_beat = 60.0 / tempo
    divisions = max(1, int(divisions))
    grid_ticks = max(1, int(grid_ticks))
    min_duration_ticks = max(1, int(min_duration_ticks))
    allowed = None
    if allowed_duration_ticks:
        allowed = sorted({int(x) for x in allowed_duration_ticks if x is not None and int(x) > 0})
        if not allowed:
            allowed = None

    errors = 0
    out: List[NoteEvent] = []
    for n in notes:
        start_s = float(n["start"])
        end_s = float(n["end"])
        pitch = int(n["pitch"])
        conf = float(n.get("confidence", 0.0))
        start_tick_raw = float(start_s) / float(seconds_per_beat) * float(divisions)
        end_tick_raw = float(end_s) / float(seconds_per_beat) * float(divisions)
        start_tick = int(round(start_tick_raw))
        end_tick = int(round(end_tick_raw))
        snap_start = int(round(start_tick / grid_ticks) * grid_ticks)
        snap_end = int(round(end_tick / grid_ticks) * grid_ticks)
        if snap_start != start_tick or snap_end != end_tick:
            errors += 1
        snap_end = max(snap_end, snap_start + min_duration_ticks)
        duration_tick = snap_end - snap_start
        if allowed is not None and duration_tick not in allowed:
            # Rabat sur la durée autorisée la plus proche (avec plancher min_duration_ticks).
            candidates = [d for d in allowed if d >= min_duration_ticks]
            if not candidates:
                candidates = allowed
            best = min(candidates, key=lambda d: abs(int(d) - int(duration_tick)))
            if best != duration_tick:
                errors += 1
                duration_tick = int(best)
                snap_end = snap_start + duration_tick
        start_q = float(snap_start) / float(divisions) * seconds_per_beat
        dur_q = float(duration_tick) / float(divisions) * seconds_per_beat
        velocity = int(clamp(round(conf * 127.0), 24, 127))
        out.append(NoteEvent(start=float(start_q), duration=float(dur_q), pitch=pitch, velocity=velocity))
    out.sort(key=lambda ev: (ev.start, ev.pitch))
    return out, {"quantizationErrorsCount": int(errors)}


def post_process_notes(
    notes: List[NoteEvent],
    quality: str,
    tempo_bpm: Optional[float],
) -> Tuple[List[NoteEvent], Dict[str, float]]:
    if not notes:
        return [], {"short_ratio": 1.0, "out_of_range_ratio": 1.0}
    quality = quality.lower()
    subdiv = 2 if quality == "fast" else 4
    tempo_value = float(tempo_bpm) if tempo_bpm and tempo_bpm > 0 else SETTINGS.default_tempo_bpm
    seconds_per_beat = 60.0 / tempo_value if tempo_value else 0.0
    grid = seconds_per_beat / subdiv
    # Post-processing musical:
    # - supprimer notes < 80–120ms
    # - fusionner segments proches (±1 demi-ton, gap < 40ms)
    min_duration = 0.12 if quality == "fast" else 0.08
    processed: List[NoteEvent] = []
    short_notes = 0
    out_of_range = 0
    sorted_notes = sorted(notes, key=lambda n: n.start)

    # Fusion pré-quantification.
    merged: List[NoteEvent] = []
    for note in sorted_notes:
        if not merged:
            merged.append(note)
            continue
        prev = merged[-1]
        gap = note.start - (prev.start + prev.duration)
        if abs(note.pitch - prev.pitch) <= 1 and gap <= 0.04:
            prev_end = prev.start + prev.duration
            new_end = max(prev_end, note.start + note.duration)
            prev.duration = new_end - prev.start
            prev.velocity = int(round((prev.velocity + note.velocity) / 2))
        else:
            merged.append(note)

    for note in merged:
        # Clamp range guitare (E2–E6).
        if note.pitch < 40 or note.pitch > 88:
            out_of_range += 1
            continue
        if note.duration < min_duration:
            short_notes += 1
            continue

        # Quantification: on quantifie start ET end (durée implicite).
        start_q = round(note.start / grid) * grid
        end_q = round((note.start + note.duration) / grid) * grid
        end_q = max(end_q, start_q + grid)
        duration_q = end_q - start_q
        if duration_q < min_duration:
            short_notes += 1
            continue
        processed.append(
            NoteEvent(
                start=float(start_q),
                duration=float(duration_q),
                pitch=note.pitch,
                velocity=note.velocity,
            )
        )

    # Fusion post-quantification (pour éliminer des splits d'onsets).
    processed = sorted(processed, key=lambda n: n.start)
    final: List[NoteEvent] = []
    for note in processed:
        if not final:
            final.append(note)
            continue
        prev = final[-1]
        gap = note.start - (prev.start + prev.duration)
        if abs(note.pitch - prev.pitch) <= 0 and abs(gap) <= 1e-6:
            prev.duration += note.duration
            prev.velocity = int(round((prev.velocity + note.velocity) / 2))
        else:
            final.append(note)
    processed = final
    total = max(1, len(notes))
    return processed, {
        "short_ratio": short_notes / total,
        "out_of_range_ratio": out_of_range / total,
    }


def _seconds_per_beat(tempo_bpm: float) -> float:
    tempo = float(tempo_bpm or SETTINGS.default_tempo_bpm)
    if tempo <= 0:
        tempo = SETTINGS.default_tempo_bpm
    return 60.0 / tempo


def _build_measure_events(
    entries: Iterable[Any],
    get_start: Callable[[Any], float],
    get_duration: Callable[[Any], float],
    serialize_note: Callable[[Any], Dict[str, Any]],
    seconds_per_beat: float,
    divisions: int,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    measure_ticks = divisions * 4
    events: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for entry in entries:
        start_ticks = get_start(entry)
        duration_secs = get_duration(entry)
        start_tick = int(round(start_ticks / seconds_per_beat * divisions))
        duration_tick = max(1, int(round(duration_secs / seconds_per_beat * divisions)))
        remaining = duration_tick
        current_start = start_tick
        segment_index = 0
        while remaining > 0:
            measure_idx = current_start // measure_ticks
            offset = current_start % measure_ticks
            available = measure_ticks - offset
            segment_duration = min(remaining, available)
            has_more = remaining > segment_duration
            if duration_tick > segment_duration:
                if segment_index == 0 and has_more:
                    tie_start = True
                    tie_stop = False
                elif has_more:
                    tie_start = True
                    tie_stop = True
                else:
                    tie_start = False
                    tie_stop = True
            else:
                tie_start = False
                tie_stop = False
            key = (measure_idx, offset)
            entry_payload = events.setdefault(
                key,
                {"notes": [], "duration": 0, "tie_start": False, "tie_stop": False},
            )
            entry_payload["notes"].append(serialize_note(entry))
            entry_payload["duration"] = max(int(entry_payload["duration"]), segment_duration)
            entry_payload["tie_start"] = bool(entry_payload["tie_start"]) or tie_start
            entry_payload["tie_stop"] = bool(entry_payload["tie_stop"]) or tie_stop
            remaining -= segment_duration
            current_start += segment_duration
            segment_index += 1
    return events


def _events_to_measures(events: Dict[Tuple[int, int], Dict[str, Any]], divisions: int) -> List[Dict[str, Any]]:
    measure_indices = sorted({idx for idx, _ in events.keys()})
    measure_count = max(measure_indices) + 1 if measure_indices else 1
    measures = []
    for measure_idx in range(measure_count):
        measure_events = [
            (offset, events[(measure_idx, offset)])
            for (idx, offset) in events.keys()
            if idx == measure_idx
        ]
        measure_events.sort(key=lambda item: item[0])
        event_list = []
        for offset, payload in measure_events:
            event_list.append(
                {
                    "startTick": offset,
                    "durationTick": payload["duration"],
                    "tieStart": payload.get("tie_start", False),
                    "tieStop": payload.get("tie_stop", False),
                    "notes": payload["notes"],
                }
            )
        measures.append({"number": measure_idx + 1, "events": event_list})
    return measures


def _build_measure_items(
    measure_events: List[Dict[str, Any]],
    divisions: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    measure_div = divisions * 4
    items: List[Dict[str, Any]] = []
    cursor = 0
    sorted_events = sorted(measure_events, key=lambda ev: int(ev.get("startTick", 0)))
    for event in sorted_events:
        offset = int(event.get("startTick", 0))
        if offset > cursor:
            items.append({"kind": "rest", "duration": offset - cursor})
        duration = int(event.get("durationTick", 0))
        duration = min(duration, max(0, measure_div - offset))
        if duration > 0:
            items.append(
                {
                    "kind": "event",
                    "duration": duration,
                    "notes": event.get("notes", []),
                    "tie_start": bool(event.get("tieStart")),
                    "tie_stop": bool(event.get("tieStop")),
                }
            )
            cursor = offset + duration
    if cursor < measure_div:
        items.append({"kind": "rest", "duration": measure_div - cursor})
    beam_status: Dict[int, str] = {}
    group: List[int] = []
    for idx, item in enumerate(items):
        if item["kind"] == "event" and int(item["duration"]) <= max(1, divisions // 2):
            group.append(idx)
        else:
            if len(group) >= 2:
                beam_status[group[0]] = "begin"
                for mid in group[1:-1]:
                    beam_status[mid] = "continue"
                beam_status[group[-1]] = "end"
            group = []
    if len(group) >= 2:
        beam_status[group[0]] = "begin"
        for mid in group[1:-1]:
            beam_status[mid] = "continue"
        beam_status[group[-1]] = "end"
    return items, beam_status


def _determine_written_octave_shift(notes: List[NoteEvent]) -> int:
    if not notes:
        return GUITAR_WRITTEN_SHIFT
    pitches = sorted(note.pitch for note in notes)
    median_idx = len(pitches) // 2
    median_value = pitches[median_idx]
    if median_value > HIGH_PITCH_THRESHOLD:
        return -GUITAR_WRITTEN_SHIFT
    if median_value < LOW_PITCH_THRESHOLD:
        return GUITAR_WRITTEN_SHIFT
    return GUITAR_WRITTEN_SHIFT


def group_chords(notes: List[NoteEvent], threshold: float = 0.03) -> List[List[NoteEvent]]:
    if not notes:
        return []
    notes_sorted = sorted(notes, key=lambda n: n.start)
    groups: List[List[NoteEvent]] = []
    current: List[NoteEvent] = [notes_sorted[0]]
    for note in notes_sorted[1:]:
        if abs(note.start - current[-1].start) <= threshold:
            current.append(note)
        else:
            groups.append(current)
            current = [note]
    groups.append(current)
    return groups


def _hand_pos_from_fret(fret: int) -> int:
    return clamp(fret - 1, 0, MAX_HAND_POS)


def _calculate_candidate_cost(
    fret: int,
    string_idx: int,
    hand_pos: int,
    last_string: Optional[int],
    hand_span: int,
    prefer_low: bool,
) -> Tuple[float, int]:
    new_hand_pos = _hand_pos_from_fret(fret)
    shift_cost = abs(new_hand_pos - hand_pos) * SHIFT_PENALTY
    # Stretch cost: si la note sort du "fret_base + span", c'est très pénalisé (jouabilité).
    outside_over = 0
    if fret < hand_pos:
        outside_over = hand_pos - fret
    elif fret > hand_pos + hand_span:
        outside_over = fret - (hand_pos + hand_span)
    outside = 0.0
    if outside_over > 0:
        outside = OUTSIDE_SPAN_PENALTY * (2.5 + 1.5 * outside_over)
    string_cost = (
        abs(string_idx - last_string) * STRING_JUMP_PENALTY if last_string is not None else 0.0
    )
    high_cost = max(0, fret - HIGH_FRET_THRESHOLD) * HIGH_FRET_PENALTY
    low_bonus = LOW_FRET_BONUS if prefer_low and fret <= LOW_FRET_THRESHOLD else 0.0
    total = shift_cost + outside + string_cost + high_cost - low_bonus
    return total, new_hand_pos


def _run_monophonic_dp(
    notes: List[NoteEvent],
    candidate_lists: List[List[Tuple[int, int]]],
    hand_span: int,
    prefer_low: bool,
    start_hand_pos: int,
    start_last_string: Optional[int],
) -> Tuple[List[_MonophonicChoice], int, Optional[int], float]:
    if not notes:
        return [], start_hand_pos, start_last_string, 0.0

    current_states: Dict[Tuple[int, Optional[int]], _DPEntry] = {
        (start_hand_pos, start_last_string): _DPEntry(cost=0.0, prev_state=None, choice=None)
    }
    history: List[Dict[Tuple[int, Optional[int]], _DPEntry]] = []
    for idx, candidates in enumerate(candidate_lists):
        next_states: Dict[Tuple[int, Optional[int]], _DPEntry] = {}
        for state, entry in current_states.items():
            hand_pos, last_string = state
            for string_idx, fret in candidates:
                candidate_cost, next_hand_pos = _calculate_candidate_cost(
                    fret, string_idx, hand_pos, last_string, hand_span, prefer_low
                )
                new_state = (next_hand_pos, string_idx)
                new_cost = entry.cost + candidate_cost
                choice = _MonophonicChoice(
                    event_index=idx,
                    string_idx=string_idx,
                    fret=fret,
                    hand_pos_before=hand_pos,
                    last_string_before=last_string,
                    hand_pos_after=next_hand_pos,
                    candidate_cost=candidate_cost,
                )
                existing = next_states.get(new_state)
                if existing is None or new_cost < existing.cost:
                    next_states[new_state] = _DPEntry(
                        cost=new_cost, prev_state=state, choice=choice
                    )
        if not next_states:
            return [], start_hand_pos, start_last_string, float("inf")
        history.append(next_states)
        current_states = next_states

    best_state, best_entry = min(current_states.items(), key=lambda item: item[1].cost)
    final_choices: List[_MonophonicChoice] = []
    state = best_state
    for step in range(len(history) - 1, -1, -1):
        entry = history[step][state]
        if entry.choice is None or entry.prev_state is None:
            break
        final_choices.append(entry.choice)
        state = entry.prev_state
    final_choices.reverse()
    total_cost = best_entry.cost
    final_hand_pos, final_last_string = best_state
    return final_choices, final_hand_pos, final_last_string, total_cost


def _solve_monophonic_segment(
    group_indices: List[int],
    notes: List[NoteEvent],
    open_strings: List[int],
    hand_span: int,
    prefer_low: bool,
    start_hand_pos: int,
    start_last_string: Optional[int],
) -> Tuple[
    List[TabNote],
    List[Dict[str, Any]],
    Dict[int, int],
    int,
    Optional[int],
    float,
]:
    candidate_lists = [possible_positions(note.pitch, open_strings) for note in notes]
    dp_notes: List[NoteEvent] = []
    dp_candidates: List[List[Tuple[int, int]]] = []
    dp_indices: List[int] = []
    for idx, candidates in enumerate(candidate_lists):
        if candidates:
            dp_notes.append(notes[idx])
            dp_candidates.append(candidates)
            dp_indices.append(idx)

    if not dp_notes:
        debug_events = []
        for idx, note in enumerate(notes):
            debug_events.append(
                {
                    "type": "monophonic",
                    "groupIndex": group_indices[idx],
                    "timeStart": note.start,
                    "duration": note.duration,
                    "pitches": [note.pitch],
                    "handPosBefore": start_hand_pos,
                    "handPosAfter": start_hand_pos,
                    "lastStringBefore": start_last_string,
                    "chosen": None,
                    "candidates": [],
                    "noteMissing": True,
                }
            )
        return [], debug_events, {}, start_hand_pos, start_last_string, 0.0

    choices, final_hand_pos, final_last_string, total_cost = _run_monophonic_dp(
        dp_notes,
        dp_candidates,
        hand_span,
        prefer_low,
        start_hand_pos,
        start_last_string,
    )

    tab_notes: List[TabNote] = []
    last_positions: Dict[int, int] = {}
    choice_map = {dp_indices[idx]: choice for idx, choice in enumerate(choices)}
    debug_events: List[Dict[str, Any]] = []
    for original_idx, note in enumerate(notes):
        event_debug = {
            "type": "monophonic",
            "groupIndex": group_indices[original_idx],
            "timeStart": note.start,
            "duration": note.duration,
            "pitches": [note.pitch],
            "handPosBefore": start_hand_pos,
            "handPosAfter": start_hand_pos,
            "lastStringBefore": start_last_string,
            "chosen": None,
            "candidates": [],
        }
        if original_idx in choice_map:
            choice = choice_map[original_idx]
            event_debug["handPosBefore"] = choice.hand_pos_before
            event_debug["handPosAfter"] = choice.hand_pos_after
            event_debug["lastStringBefore"] = choice.last_string_before
            candidate_debug: List[Dict[str, Any]] = []
            for string_idx, fret in candidate_lists[original_idx]:
                candidate_cost, candidate_hand_pos = _calculate_candidate_cost(
                    fret,
                    string_idx,
                    choice.hand_pos_before,
                    choice.last_string_before,
                    hand_span,
                    prefer_low,
                )
                candidate_debug.append(
                    {
                        "stringIdx": string_idx,
                        "stringNumber": 6 - string_idx,
                        "fret": fret,
                        "handPosAfter": candidate_hand_pos,
                        "cost": candidate_cost,
                    }
                )
            event_debug["candidates"] = candidate_debug
            selected = {
                "stringIdx": choice.string_idx,
                "stringNumber": 6 - choice.string_idx,
                "fret": choice.fret,
                "handPosAfter": choice.hand_pos_after,
                "cost": choice.candidate_cost,
            }
            event_debug["chosen"] = selected
            tab_notes.append(
                TabNote(
                    timeStart=note.start,
                    duration=note.duration,
                    pitch=note.pitch,
                    string=6 - choice.string_idx,
                    fret=choice.fret,
                )
            )
            last_positions[choice.string_idx] = choice.fret
            debug_events.append(event_debug)
        else:
            event_debug["noteMissing"] = True
            debug_events.append(event_debug)
    return tab_notes, debug_events, last_positions, final_hand_pos, final_last_string, total_cost

def possible_positions(pitch: int, open_strings: List[int], max_fret: int = 24) -> List[Tuple[int, int]]:
    positions = []
    for idx, open_pitch in enumerate(open_strings):
        fret = pitch - open_pitch
        if 0 <= fret <= max_fret:
            positions.append((idx, fret))
    return positions


def assign_chord(
    notes: List[NoteEvent],
    open_strings: List[int],
    last_positions: Dict[int, int],
    hand_span: int = DEFAULT_SPAN_FRETS,
    prefer_low: bool = False,
) -> Tuple[Dict[int, Tuple[int, int]], int, float]:
    candidates = [possible_positions(n.pitch, open_strings) for n in notes]
    if any(len(c) == 0 for c in candidates):
        return {}, 0, float("inf")

    best_cost = float("inf")
    best_assignment: Dict[int, Tuple[int, int]] = {}
    best_hand_pos = 0

    def evaluate_assignment(assignment: Dict[int, Tuple[int, int]]) -> Tuple[float, int]:
        frets = [assignment[i][1] for i in range(len(notes))]
        min_fret = min(frets)
        max_fret = max(frets)
        span_needed = max_fret - min_fret
        hand_pos_candidate = clamp(min_fret, 0, MAX_HAND_POS)
        hand_pos_candidate = clamp(hand_pos_candidate, 0, MAX_HAND_POS)
        if max_fret > hand_pos_candidate + hand_span:
            hand_pos_candidate = clamp(max_fret - hand_span, 0, MAX_HAND_POS)
        if min_fret < hand_pos_candidate:
            hand_pos_candidate = clamp(min_fret, 0, MAX_HAND_POS)
        outside_penalty = 0.0
        if min_fret < hand_pos_candidate or max_fret > hand_pos_candidate + hand_span:
            outside_penalty = OUTSIDE_SPAN_PENALTY

        span_over = max(0, span_needed - hand_span)
        span_penalty = span_over * (
            CHORD_SPAN_OVER_PENALTY if hand_span <= DEFAULT_SPAN_FRETS else CHORD_SPAN_OVER_PENALTY * 0.5
        )
        center = hand_pos_candidate + hand_span / 2
        dispersion_penalty = sum(abs(fret - center) for fret in frets) * CHORD_DISPERSION_PENALTY
        movement_cost = sum(
            abs(fret - last_positions.get(string_idx, fret))
            for string_idx, fret in assignment.values()
        )
        high_cost = sum(max(0, fret - 12) for _, fret in assignment.values()) * HIGH_FRET_PENALTY
        low_bonus = sum(
            LOW_FRET_BONUS for _, fret in assignment.values() if prefer_low and fret <= LOW_FRET_THRESHOLD
        )
        total_cost = (
            movement_cost
            + outside_penalty
            + span_penalty
            + dispersion_penalty
            + high_cost
            - low_bonus
        )
        return total_cost, hand_pos_candidate

    def recurse(i: int, used_strings: set, current_cost: float, assignment: Dict[int, Tuple[int, int]]):
        nonlocal best_cost, best_assignment, best_hand_pos
        if i >= len(notes):
            cost, hand_pos = evaluate_assignment(assignment)
            if cost < best_cost:
                best_cost = cost
                best_assignment = dict(assignment)
                best_hand_pos = hand_pos
            return
        if current_cost >= best_cost:
            return
        for string_idx, fret in candidates[i]:
            if string_idx in used_strings:
                continue
            last_fret = last_positions.get(string_idx)
            move_cost = abs(fret - last_fret) if last_fret is not None else 0
            assignment[i] = (string_idx, fret)
            recurse(i + 1, used_strings | {string_idx}, current_cost + move_cost, assignment)
            assignment.pop(i, None)

    recurse(0, set(), 0.0, {})
    return best_assignment, best_hand_pos, best_cost


def map_notes_to_tab(
    notes: List[NoteEvent],
    tuning: str,
    capo: int,
    hand_span: int = DEFAULT_SPAN_FRETS,
    prefer_low_frets: bool = False,
) -> Tuple[List[TabNote], Optional[str], Dict[str, Any]]:
    hand_span = max(1, min(MAX_HAND_POS, hand_span))
    open_strings, tuning_warning = parse_tuning(tuning, capo)
    chord_groups = group_chords(notes)
    tab_notes: List[TabNote] = []
    last_positions: Dict[int, int] = {}
    fingering_debug: List[Dict[str, Any]] = []
    total_cost = 0.0
    processed_events = 0
    current_hand_pos = 0
    current_last_string: Optional[int] = None

    idx = 0
    while idx < len(chord_groups):
        group = chord_groups[idx]
        if len(group) == 1:
            start_idx = idx
            while idx < len(chord_groups) and len(chord_groups[idx]) == 1:
                idx += 1
            segment_groups = chord_groups[start_idx:idx]
            segment_notes = [item[0] for item in segment_groups]
            segment_indices = list(range(start_idx, idx))
            (
                segment_tab,
                segment_debug,
                segment_last_positions,
                segment_hand_pos,
                segment_last_string,
                segment_cost,
            ) = _solve_monophonic_segment(
                segment_indices,
                segment_notes,
                open_strings,
                hand_span,
                prefer_low_frets,
                current_hand_pos,
                current_last_string,
            )
            tab_notes.extend(segment_tab)
            fingering_debug.extend(segment_debug)
            last_positions.update(segment_last_positions)
            total_cost += segment_cost
            processed_events += sum(1 for entry in segment_debug if entry.get("chosen"))
            current_hand_pos = segment_hand_pos
            current_last_string = segment_last_string
            continue

        assignment, chord_hand_pos, chord_cost = assign_chord(
            group,
            open_strings,
            last_positions,
            hand_span,
            prefer_low_frets,
        )

        if assignment and chord_cost != float("inf"):
            choice_entries: List[Dict[str, Any]] = []
            chord_tab: List[TabNote] = []
            for note_idx, note in enumerate(group):
                string_idx, fret = assignment.get(note_idx, (None, None))
                if string_idx is None:
                    continue
                chord_tab.append(
                    TabNote(
                        timeStart=note.start,
                        duration=note.duration,
                        pitch=note.pitch,
                        string=6 - string_idx,
                        fret=fret,
                    )
                )
                last_positions[string_idx] = fret
                choice_entries.append(
                    {
                        "noteIndex": note_idx,
                        "stringIdx": string_idx,
                        "stringNumber": 6 - string_idx,
                        "fret": fret,
                    }
                )
            tab_notes.extend(chord_tab)
            fingering_debug.append(
                {
                    "type": "chord",
                    "groupIndex": idx,
                    "timeStart": group[0].start,
                    "duration": max(n.duration for n in group),
                    "pitches": [n.pitch for n in group],
                    "handPosBefore": current_hand_pos,
                    "handPosAfter": chord_hand_pos,
                    "lastStringBefore": current_last_string,
                    "chosen": choice_entries,
                    "playabilityCost": chord_cost,
                }
            )
            total_cost += chord_cost
            processed_events += 1
            current_hand_pos = chord_hand_pos
            if choice_entries:
                current_last_string = max(choice_entries, key=lambda entry: entry["fret"])["stringIdx"]
        else:
            for note in group:
                candidates = possible_positions(note.pitch, open_strings)
                event_debug = {
                    "type": "chordFallback",
                    "groupIndex": idx,
                    "timeStart": note.start,
                    "duration": note.duration,
                    "pitches": [note.pitch],
                    "handPosBefore": current_hand_pos,
                    "handPosAfter": current_hand_pos,
                    "lastStringBefore": current_last_string,
                    "chosen": None,
                    "candidates": [],
                }
                if not candidates:
                    event_debug["noteMissing"] = True
                    fingering_debug.append(event_debug)
                    continue
                best_choice = None
                best_cost = float("inf")
                fallback_candidates: List[Dict[str, Any]] = []
                for string_idx, fret in candidates:
                    candidate_cost, candidate_hand_pos = _calculate_candidate_cost(
                        fret,
                        string_idx,
                        current_hand_pos,
                        current_last_string,
                        hand_span,
                        prefer_low_frets,
                    )
                    fallback_candidates.append(
                        {
                            "stringIdx": string_idx,
                            "stringNumber": 6 - string_idx,
                            "fret": fret,
                            "handPosAfter": candidate_hand_pos,
                            "cost": candidate_cost,
                        }
                    )
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_choice = (string_idx, fret, candidate_hand_pos, candidate_cost)
                if best_choice is None:
                    event_debug["noteMissing"] = True
                    fingering_debug.append(event_debug)
                    continue
                string_idx, fret, new_hand_pos, chosen_cost = best_choice
                event_debug["candidates"] = fallback_candidates
                event_debug["handPosAfter"] = new_hand_pos
                event_debug["chosen"] = {
                    "stringIdx": string_idx,
                    "stringNumber": 6 - string_idx,
                    "fret": fret,
                    "handPosAfter": new_hand_pos,
                    "cost": chosen_cost,
                }
                tab_notes.append(
                    TabNote(
                        timeStart=note.start,
                        duration=note.duration,
                        pitch=note.pitch,
                        string=6 - string_idx,
                        fret=fret,
                    )
                )
                last_positions[string_idx] = fret
                total_cost += chosen_cost
                processed_events += 1
                current_hand_pos = new_hand_pos
                current_last_string = string_idx
                fingering_debug.append(event_debug)
        idx += 1

    average_cost = total_cost / max(1, processed_events)
    playability_score = clamp(
        1.0 - min(1.0, average_cost / PLAYABILITY_COST_SCALE),
        0.0,
        1.0,
    )

    shifts: List[float] = []
    max_stretch = 0
    unreachable_count = 0
    for ev in fingering_debug:
        if ev.get("noteMissing"):
            unreachable_count += 1
        chosen = ev.get("chosen")
        if not chosen:
            continue
        try:
            before = int(ev.get("handPosBefore", 0))
            after = int(ev.get("handPosAfter", 0))
        except (TypeError, ValueError):
            before = 0
            after = 0
        shifts.append(abs(after - before))
        # "stretch" = distance fret - fret_base (handPosBefore).
        if isinstance(chosen, dict):
            fret = int(chosen.get("fret", 0))
            max_stretch = max(max_stretch, max(0, fret - before))
        elif isinstance(chosen, list):
            try:
                frets = [int(item.get("fret", 0)) for item in chosen if isinstance(item, dict)]
            except Exception:  # noqa: BLE001
                frets = []
            if frets:
                max_stretch = max(max_stretch, max(0, max(frets) - before))
    avg_shift = float(np.mean(shifts)) if shifts else 0.0

    playability_debug = {
        "handSpan": hand_span,
        "preferLowFrets": prefer_low_frets,
        "totalCost": total_cost,
        "averageCost": average_cost,
        "playabilityScore": playability_score,
        "avgShift": avg_shift,
        "maxStretch": int(max_stretch),
        "unreachableCount": int(unreachable_count),
        "events": fingering_debug,
    }

    return sorted(tab_notes, key=lambda n: n.timeStart), tuning_warning, playability_debug


def compute_confidence(metrics: Dict[str, float], notes: List[NoteEvent]) -> float:
    if not notes:
        return 0.0
    density = len(notes) / max(1.0, max(n.start + n.duration for n in notes))
    density_penalty = clamp(density / 8.0, 0.0, 1.0)
    score = 1.0 - (metrics["short_ratio"] * 0.4 + metrics["out_of_range_ratio"] * 0.4 + density_penalty * 0.2)
    return clamp(score, 0.0, 1.0)


def pitch_from_string_fret(string_number: int, fret: int, open_strings: List[int]) -> int:
    if not (1 <= string_number <= len(open_strings)):
        raise ValueError(f"Numéro de corde invalide: {string_number}")
    open_index = len(open_strings) - string_number
    return open_strings[open_index] + fret


def _midi_to_note_label(midi: int) -> str:
    step, alter, octave = _midi_to_pitch(midi)
    label = step
    if alter == 1:
        label += "#"
    elif alter == -1:
        label += "b"
    return f"{label}{octave}"


def build_tab_json(
    tab_notes: List[TabNote],
    tuning: str,
    capo: int,
    quality: str,
    tempo_bpm: float,
    tempo_source: str,
    warnings: List[str],
    divisions: int = DEFAULT_DIVISIONS,
) -> Tuple[Dict[str, object], int]:
    divisions = max(1, int(divisions))
    seconds_per_beat = _seconds_per_beat(tempo_bpm)
    events = _build_measure_events(
        tab_notes,
        lambda note: note.timeStart,
        lambda note: note.duration,
        lambda note: {"string": note.string, "fret": note.fret, "midi": note.pitch},
        seconds_per_beat,
        divisions,
    )
    measures = _events_to_measures(events, divisions)
    open_strings, _ = parse_tuning(tuning, capo)

    tuning_names = [_midi_to_note_label(midi) for midi in reversed(open_strings)]
    metadata = {
        "tuning": tuning_names,
        "tempo": tempo_bpm,
        "tempoSource": tempo_source,
        "timeSignature": DEFAULT_TIME_SIGNATURE,
        "divisions": divisions,
        "quality": quality,
    }
    total_notes = sum(len(event["notes"]) for measure in measures for event in measure["events"])
    tab_json = {
        "metadata": metadata,
        "quality": quality,
        "warnings": warnings,
        "measures": measures,
    }
    return tab_json, total_notes


def build_score_json(
    note_events: List[NoteEvent],
    quality: str,
    tempo_bpm: float,
    tempo_source: str,
    warnings: Optional[List[str]] = None,
    divisions: int = DEFAULT_DIVISIONS,
) -> Tuple[Dict[str, object], int]:
    divisions = max(1, int(divisions))
    seconds_per_beat = _seconds_per_beat(tempo_bpm)
    written_shift = _determine_written_octave_shift(note_events)

    def serialize(note: NoteEvent) -> Dict[str, int]:
        return {"midi": note.pitch}

    sorted_notes = sorted(note_events, key=lambda note: note.start)
    events = _build_measure_events(
        sorted_notes,
        lambda note: note.start,
        lambda note: note.duration,
        serialize,
        seconds_per_beat,
        divisions,
    )
    measures = _events_to_measures(events, divisions)
    total_notes = sum(len(event["notes"]) for measure in measures for event in measure["events"])

    metadata = {
        "tempo": tempo_bpm,
        "tempoSource": tempo_source,
        "timeSignature": DEFAULT_TIME_SIGNATURE,
        "divisions": divisions,
        "quality": quality,
        "scoreWrittenOctaveShift": written_shift,
    }
    score_json: Dict[str, object] = {
        "metadata": metadata,
        "measures": measures,
    }
    if warnings:
        score_json["warnings"] = list(warnings)
    return score_json, total_notes


def render_tab_txt_from_json(
    tab_json: Dict[str, object],
    quality: str,
    measures_per_system: int = 2,
    logger=None,
) -> Tuple[str, int, List[str]]:
    metadata = tab_json.get("metadata", {})
    divisions = int(metadata.get("divisions") or 4)
    measure_ticks = divisions * 4
    measures = tab_json.get("measures", [])
    measure_count = max(len(measures), 1)
    measure_grids = [
        {string_num: ["---"] * measure_ticks for string_num in range(1, 7)}
        for _ in range(measure_count)
    ]

    warnings: List[str] = []
    note_count = 0
    for idx, measure in enumerate(measures):
        grid = measure_grids[idx]
        for event in measure.get("events", []):
            start_tick = int(event.get("startTick", 0))
            if start_tick < 0 or start_tick >= measure_ticks:
                warning = f"Tic de départ hors mesure {idx + 1}: {start_tick}"
                warnings.append(warning)
                if logger:
                    logger.warning(warning)
                continue
            for note in event.get("notes", []):
                string_number = int(note.get("string", 1))
                if string_number < 1 or string_number > 6:
                    warning = f"Corde invalide ({string_number}) dans la mesure {idx + 1}."
                    warnings.append(warning)
                    if logger:
                        logger.warning(warning)
                    continue
                row = grid[string_number]
                fret_str = str(note.get("fret", ""))
                token = (fret_str + "-" * (3 - len(fret_str)))[:3]
                if row[start_tick] != "---":
                    warning = (
                        f"Collision mesure {idx + 1}, corde {string_number}, tick {start_tick}: "
                        f"{row[start_tick]} déjà présent, {token} ignoré."
                    )
                    warnings.append(warning)
                    if logger:
                        logger.warning(warning)
                    continue
                row[start_tick] = token
                note_count += 1

    tuning_names = metadata.get("tuning") or []
    tempo_value = metadata.get("tempo")
    header_lines: List[str] = []
    if tuning_names:
        header_lines.append(f"Accordage: {', '.join(str(name) for name in tuning_names)}")
    if tempo_value is not None:
        tempo_str = f"{int(round(tempo_value))}" if isinstance(tempo_value, (int, float)) else str(tempo_value)
        header_lines.append(f"Tempo: {tempo_str}")
    if quality:
        header_lines.append(f"Qualité: {quality}")
    if header_lines:
        header_lines.append("")

    line_order = [1, 2, 3, 4, 5, 6]
    names = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}
    lines: List[str] = []
    for block_start in range(0, measure_count, measures_per_system):
        block_grids = measure_grids[block_start : block_start + measures_per_system]
        if not block_grids:
            continue
        block_label = f"Mesures {block_start + 1}-{block_start + len(block_grids)}"
        lines.append(block_label)
        for string in line_order:
            block_cells = "|".join("".join(grid[string]) for grid in block_grids)
            lines.append(f"{names[string]}|{block_cells}|")
        lines.append("")

    output_lines = header_lines + lines if header_lines else lines
    while output_lines and output_lines[-1] == "":
        output_lines.pop()

    return "\n".join(output_lines), note_count, warnings


def write_tab_outputs(
    tab_notes: List[TabNote],
    tuning: str,
    capo: int,
    quality: str,
    tempo_bpm: float,
    tempo_source: str,
    warnings: List[str],
    tab_txt_path: str,
    tab_json_path: str,
    logger=None,
    measures_per_system: int = 2,
    pretty_export: bool = False,
    divisions: int = DEFAULT_DIVISIONS,
) -> Tuple[Dict[str, object], int, List[str], int]:
    tab_json, total_notes = build_tab_json(
        tab_notes, tuning, capo, quality, tempo_bpm, tempo_source, warnings, divisions=divisions
    )
    if total_notes == 0:
        raise RuntimeError("Aucune tablature générée (pas de notes exploitables).")
    write_json(tab_json_path, tab_json)
    tab_txt, tab_txt_notes, render_warnings = render_tab_txt_from_json(
        tab_json, quality, measures_per_system=measures_per_system, logger=logger
    )
    with open(tab_txt_path, "w", encoding="utf-8") as f:
        f.write(tab_txt)

    if pretty_export:
        pretty_path = os.path.join(os.path.dirname(tab_txt_path), "tab.pretty.txt")
        pretty_text = tab_txt.replace("|", " | ").replace("---", "   ")
        with open(pretty_path, "w", encoding="utf-8") as pretty_file:
            pretty_file.write(pretty_text)

    output_warnings = list(render_warnings)
    if total_notes != tab_txt_notes:
        diff_path = os.path.join(os.path.dirname(tab_txt_path), "diff_tabtxt.txt")
        with open(diff_path, "w", encoding="utf-8") as diff_file:
            diff_file.write(f"Notes tab.json : {total_notes}\n")
            diff_file.write(f"Notes tab.txt : {tab_txt_notes}\n")
        warning = (
            f"Différence entre tab.txt ({tab_txt_notes}) et tab.json ({total_notes}) "
            f"(voir {os.path.basename(diff_path)})."
        )
        output_warnings.append(warning)
        if logger:
            logger.warning(warning)

    return tab_json, tab_txt_notes, output_warnings, total_notes


def load_tab_json(path: str) -> Dict[str, object]:
    return read_json(path)


def tab_json_note_tuples(tab_json: Dict[str, Any]) -> List[Tuple[Optional[int], int, int]]:
    tuples: List[Tuple[Optional[int], int, int]] = []
    for measure in tab_json.get("measures", []):
        measure_number = measure.get("number")
        for event in measure.get("events", []):
            for note in event.get("notes", []):
                string_number = int(note.get("string", 0))
                fret_number = int(note.get("fret", 0))
                tuples.append(
                    (
                        int(measure_number) if isinstance(measure_number, int) else None,
                        string_number,
                        fret_number,
                    )
                )
    return tuples


def parse_musicxml_notes(xml_path: str) -> List[Tuple[Optional[int], int, int]]:
    if not os.path.exists(xml_path):
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    part = root.find("part")
    if part is None:
        return []
    collected: List[Tuple[Optional[int], int, int]] = []
    for measure in part.findall("measure"):
        measure_number = measure.get("number")
        for note_el in measure.findall("note"):
            technical = note_el.find("notations/technical")
            if technical is None:
                continue
            string_text = technical.findtext("string")
            fret_text = technical.findtext("fret")
            if string_text is None or fret_text is None:
                continue
            try:
                string_num = int(string_text)
                fret_num = int(fret_text)
            except ValueError:
                continue
            collected.append(
                (
                    int(measure_number) if measure_number and measure_number.isdigit() else None,
                    string_num,
                    fret_num,
                )
            )
    return collected


def compare_tab_json_and_musicxml(
    tab_json: Dict[str, Any],
    xml_path: str,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    tab_tuples = tab_json_note_tuples(tab_json)
    xml_tuples = parse_musicxml_notes(xml_path)
    tab_counter = Counter(tab_tuples)
    xml_counter = Counter(xml_tuples)
    diffs: List[Dict[str, Any]] = []
    for key in set(tab_counter) | set(xml_counter):
        tab_count = tab_counter.get(key, 0)
        xml_count = xml_counter.get(key, 0)
        if tab_count != xml_count:
            diffs.append(
                {
                    "measure": key[0],
                    "string": key[1],
                    "fret": key[2],
                    "countTabJson": tab_count,
                    "countMusicXML": xml_count,
                }
            )
    return len(tab_tuples), len(xml_tuples), diffs


def _midi_to_pitch(midi: int) -> Tuple[str, int, int]:
    steps = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
    alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    step = steps[midi % 12]
    alter = alters[midi % 12]
    octave = midi // 12 - 1
    return step, alter, octave


def _duration_type(duration_div: int, divisions: int) -> str:
    mapping = {
        divisions * 4: "whole",
        divisions * 2: "half",
        divisions: "quarter",
        max(1, divisions // 2): "eighth",
        max(1, divisions // 4): "16th",
    }
    return mapping.get(duration_div, "quarter")


def _format_tuning_label(open_strings: List[int]) -> str:
    names: List[str] = []
    for idx, midi in enumerate(reversed(open_strings), start=1):
        step, alter, _ = _midi_to_pitch(midi)
        note = step + ("#" if alter == 1 else "")
        names.append(f"{idx} - {note}")
    return "Accordage: " + ", ".join(names)


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def write_tab_musicxml(
    tab_json: Dict[str, object],
    tuning: str,
    capo: int,
    tempo_bpm: float,
    title: str,
    artist: str,
    annotations: Optional[List[Dict[str, str]]],
    output_path: str,
    logger=None,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    metadata = tab_json.get("metadata", {})
    divisions = int(metadata.get("divisions", 4) or 4)
    seconds_per_beat = 60.0 / tempo_bpm if tempo_bpm else 0.0
    open_strings, _ = parse_tuning(tuning, capo)
    measures = tab_json.get("measures", [])
    measure_count = max(len(measures), 1)

    annotations = annotations or []
    annotations_by_measure: Dict[int, List[Dict[str, str]]] = {}
    for item in annotations:
        try:
            ts = float(item.get("timeStart", 0))
        except (TypeError, ValueError):
            ts = 0.0
        idx = int((ts / seconds_per_beat) // 4) if seconds_per_beat else 0
        annotations_by_measure.setdefault(idx, []).append(item)

    score = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title or "Transcription"
    identification = ET.SubElement(score, "identification")
    ET.SubElement(identification, "creator", type="composer").text = artist or "Artiste inconnu"
    defaults = ET.SubElement(score, "defaults")
    page_layout = ET.SubElement(defaults, "page-layout")
    ET.SubElement(page_layout, "page-height").text = "1683.78"
    ET.SubElement(page_layout, "page-width").text = "1190.55"
    margins = ET.SubElement(page_layout, "page-margins", type="both")
    ET.SubElement(margins, "left-margin").text = "70"
    ET.SubElement(margins, "right-margin").text = "70"
    ET.SubElement(margins, "top-margin").text = "70"
    ET.SubElement(margins, "bottom-margin").text = "70"

    credit = ET.SubElement(score, "credit")
    credit_words = ET.SubElement(
        credit,
        "credit-words",
        default_x="1120",
        default_y="1600",
        justify="right",
    )
    credit_words.text = _format_tuning_label(open_strings)

    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Guitare"

    part = ET.SubElement(score, "part", id="P1")
    section_labels = ["A", "B", "C", "D", "E"]

    for measure_idx in range(measure_count):
        measure = ET.SubElement(part, "measure", number=str(measure_idx + 1))
        if measure_idx == 0:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = str(divisions)
            key = ET.SubElement(attributes, "key")
            ET.SubElement(key, "fifths").text = "0"
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "beats").text = "4"
            ET.SubElement(time, "beat-type").text = "4"
            clef = ET.SubElement(attributes, "clef")
            ET.SubElement(clef, "sign").text = "TAB"
            ET.SubElement(clef, "line").text = "5"
            staff_details = ET.SubElement(attributes, "staff-details")
            ET.SubElement(staff_details, "staff-lines").text = "6"
            for line_idx, midi in enumerate(reversed(open_strings), start=1):
                step, alter, octave = _midi_to_pitch(midi)
                tuning_el = ET.SubElement(staff_details, "staff-tuning", line=str(line_idx))
                ET.SubElement(tuning_el, "tuning-step").text = step
                if alter:
                    ET.SubElement(tuning_el, "tuning-alter").text = str(alter)
                ET.SubElement(tuning_el, "tuning-octave").text = str(octave)
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = str(int(round(tempo_bpm)))
            ET.SubElement(direction, "sound", tempo=str(int(round(tempo_bpm))))

        if measure_idx < len(section_labels):
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            ET.SubElement(direction_type, "rehearsal").text = section_labels[measure_idx]

        for annotation in annotations_by_measure.get(measure_idx, []):
            text = (annotation.get("text") or "").strip()
            if not text:
                continue
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            ET.SubElement(direction_type, "words").text = text

        measure_data = measures[measure_idx] if measure_idx < len(measures) else {"events": []}
        measure_events = list(measure_data.get("events", []))
        items, beam_status = _build_measure_items(measure_events, divisions)

        for item_idx, item in enumerate(items):
            duration = int(item["duration"])
            if item["kind"] == "rest":
                note_el = ET.SubElement(measure, "note")
                ET.SubElement(note_el, "rest")
                ET.SubElement(note_el, "duration").text = str(duration)
                ET.SubElement(note_el, "type").text = _duration_type(duration, divisions)
                continue

            notes_data = item.get("notes", [])
            base_tie_start = bool(item.get("tie_start"))
            base_tie_stop = bool(item.get("tie_stop"))
            for n_idx, chord_note in enumerate(notes_data):
                note_el = ET.SubElement(measure, "note")
                if n_idx > 0:
                    ET.SubElement(note_el, "chord")
                string_number = int(chord_note.get("string", 1))
                fret_number = int(chord_note.get("fret", 0))
                midi_value = pitch_from_string_fret(string_number, fret_number, open_strings)
                step, alter, octave = _midi_to_pitch(midi_value)
                pitch_el = ET.SubElement(note_el, "pitch")
                ET.SubElement(pitch_el, "step").text = step
                if alter:
                    ET.SubElement(pitch_el, "alter").text = str(alter)
                ET.SubElement(pitch_el, "octave").text = str(octave)
                ET.SubElement(note_el, "duration").text = str(duration)
                ET.SubElement(note_el, "type").text = _duration_type(duration, divisions)
                ET.SubElement(note_el, "voice").text = "1"
                if item_idx in beam_status and n_idx == 0:
                    ET.SubElement(note_el, "beam", number="1").text = beam_status[item_idx]
                notations = ET.SubElement(note_el, "notations")
                if base_tie_start:
                    ET.SubElement(notations, "tied", type="start")
                    ET.SubElement(note_el, "tie", type="start")
                if base_tie_stop:
                    ET.SubElement(notations, "tied", type="stop")
                    ET.SubElement(note_el, "tie", type="stop")
                technical = ET.SubElement(notations, "technical")
                ET.SubElement(technical, "string").text = str(string_number)
                ET.SubElement(technical, "fret").text = str(fret_number)

    _indent_xml(score)
    tree = ET.ElementTree(score)
    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
            b'"http://www.musicxml.org/dtds/partwise.dtd">\n'
        )
        tree.write(f, encoding="utf-8")
    tab_count, xml_count, diff_report = compare_tab_json_and_musicxml(tab_json, output_path)
    if diff_report:
        diff_path = os.path.join(os.path.dirname(output_path), "diff_report_tab.txt")
        with open(diff_path, "w", encoding="utf-8") as diff_file:
            diff_file.write(f"Notes tab.json : {tab_count}\n")
            diff_file.write(f"Notes tab MusicXML : {xml_count}\n")
            diff_file.write("Différences tab/musical:\n")
            for diff in diff_report:
                diff_file.write(
                    f"Mesure {diff.get('measure')} corde {diff.get('string')} frette {diff.get('fret')}: "
                    f"{diff.get('countTabJson')} vs {diff.get('countMusicXML')}\n"
                )
        message = f"Mismatch entre tab.json ({tab_count}) et MusicXML ({xml_count})."
        if logger:
            logger.error("%s (voir %s)", message, diff_path)
        else:
            print(message)
        raise RuntimeError(message)
    return tab_count, xml_count, diff_report


def write_score_musicxml(
    score_json: Dict[str, Any],
    tempo_bpm: float,
    title: str,
    artist: str,
    annotations: Optional[List[Dict[str, str]]],
    output_path: str,
    guitar_notation: bool = True,
    logger=None,
) -> Tuple[int, int]:
    metadata = score_json.get("metadata", {})
    divisions = int(metadata.get("divisions", DEFAULT_DIVISIONS) or DEFAULT_DIVISIONS)
    time_sig = metadata.get("timeSignature", DEFAULT_TIME_SIGNATURE)
    written_shift = int(metadata.get("scoreWrittenOctaveShift", GUITAR_WRITTEN_SHIFT))
    try:
        beats, beat_type = [int(part) for part in time_sig.split("/") if part]
    except ValueError:
        beats, beat_type = 4, 4
    annotations = annotations or []
    seconds_per_beat = _seconds_per_beat(tempo_bpm)
    annotations_by_measure: Dict[int, List[Dict[str, str]]] = {}
    for item in annotations:
        try:
            ts = float(item.get("timeStart", 0))
        except (TypeError, ValueError):
            ts = 0.0
        idx = int((ts / seconds_per_beat) // beats) if seconds_per_beat else 0
        annotations_by_measure.setdefault(idx, []).append(item)

    score = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title or "Partition"
    identification = ET.SubElement(score, "identification")
    ET.SubElement(identification, "creator", type="composer").text = artist or "Artiste inconnu"
    defaults = ET.SubElement(score, "defaults")
    page_layout = ET.SubElement(defaults, "page-layout")
    ET.SubElement(page_layout, "page-height").text = "1683.78"
    ET.SubElement(page_layout, "page-width").text = "1190.55"
    margins = ET.SubElement(page_layout, "page-margins", type="both")
    ET.SubElement(margins, "left-margin").text = "70"
    ET.SubElement(margins, "right-margin").text = "70"
    ET.SubElement(margins, "top-margin").text = "70"
    ET.SubElement(margins, "bottom-margin").text = "70"
    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Partition guitare"

    part = ET.SubElement(score, "part", id="P1")
    section_labels = ["A", "B", "C", "D", "E"]
    measures = score_json.get("measures", [])
    measure_count = max(len(measures), 1)
    score_note_count = sum(len(event["notes"]) for measure in measures for event in measure["events"])
    xml_note_count = 0

    for measure_idx in range(measure_count):
        measure = ET.SubElement(part, "measure", number=str(measure_idx + 1))
        if measure_idx == 0:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = str(divisions)
            key = ET.SubElement(attributes, "key")
            ET.SubElement(key, "fifths").text = "0"
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "beats").text = str(beats)
            ET.SubElement(time, "beat-type").text = str(beat_type)
            if guitar_notation:
                clef = ET.SubElement(attributes, "clef")
                ET.SubElement(clef, "sign").text = "G"
                ET.SubElement(clef, "line").text = "2"
                ET.SubElement(clef, "clef-octave-change").text = "-1"
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = str(int(round(tempo_bpm)))
            ET.SubElement(direction, "sound", tempo=str(int(round(tempo_bpm))))

        if measure_idx < len(section_labels):
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            ET.SubElement(direction_type, "rehearsal").text = section_labels[measure_idx]

        for annotation in annotations_by_measure.get(measure_idx, []):
            text = (annotation.get("text") or "").strip()
            if not text:
                continue
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            ET.SubElement(direction_type, "words").text = text

        measure_data = measures[measure_idx] if measure_idx < len(measures) else {"events": []}
        measure_events = list(measure_data.get("events", []))
        items, beam_status = _build_measure_items(measure_events, divisions)
        for item_idx, item in enumerate(items):
            duration = int(item["duration"])
            if item["kind"] == "rest":
                note_el = ET.SubElement(measure, "note")
                ET.SubElement(note_el, "rest")
                ET.SubElement(note_el, "duration").text = str(duration)
                ET.SubElement(note_el, "type").text = _duration_type(duration, divisions)
                continue

            notes_data = item.get("notes", [])
            base_tie_start = bool(item.get("tie_start"))
            base_tie_stop = bool(item.get("tie_stop"))
            for n_idx, note_data in enumerate(notes_data):
                note_el = ET.SubElement(measure, "note")
                if n_idx > 0:
                    ET.SubElement(note_el, "chord")
                midi_value = int(note_data.get("midi", 0))
                written_midi = int(clamp(midi_value + written_shift, 0, 127))
                step, alter, octave = _midi_to_pitch(written_midi)
                pitch_el = ET.SubElement(note_el, "pitch")
                ET.SubElement(pitch_el, "step").text = step
                if alter:
                    ET.SubElement(pitch_el, "alter").text = str(alter)
                ET.SubElement(pitch_el, "octave").text = str(octave)
                ET.SubElement(note_el, "duration").text = str(duration)
                ET.SubElement(note_el, "type").text = _duration_type(duration, divisions)
                ET.SubElement(note_el, "voice").text = "1"
                if item_idx in beam_status and n_idx == 0:
                    ET.SubElement(note_el, "beam", number="1").text = beam_status[item_idx]
                notations = ET.SubElement(note_el, "notations")
                if base_tie_start:
                    ET.SubElement(notations, "tied", type="start")
                    ET.SubElement(note_el, "tie", type="start")
                if base_tie_stop:
                    ET.SubElement(notations, "tied", type="stop")
                    ET.SubElement(note_el, "tie", type="stop")
                xml_note_count += 1

    _indent_xml(score)
    tree = ET.ElementTree(score)
    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
            b'"http://www.musicxml.org/dtds/partwise.dtd">\n'
        )
        tree.write(f, encoding="utf-8")

    if score_note_count != xml_note_count:
        diff_path = os.path.join(os.path.dirname(output_path), "diff_report_score.txt")
        with open(diff_path, "w", encoding="utf-8") as diff_file:
            diff_file.write(f"Notes score.json : {score_note_count}\n")
            diff_file.write(f"Notes score MusicXML : {xml_note_count}\n")
        message = f"Mismatch entre score.json ({score_note_count}) et MusicXML ({xml_note_count})."
        if logger:
            logger.error("%s (voir %s)", message, diff_path)
        else:
            print(message)
        raise RuntimeError(message)
    return score_note_count, xml_note_count


def render_musicxml_to_pdf(musicxml_path: str, pdf_path: str, logger) -> None:
    run_cmd(
        [
            SETTINGS.musescore_path,
            "-o",
            pdf_path,
            musicxml_path,
        ],
        logger,
        env={
            "QT_QPA_PLATFORM": "offscreen",
            "DISPLAY": ":0",
        },
    )

