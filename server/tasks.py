import csv
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sqlalchemy.orm import Session

from .config import SETTINGS
from .db import SessionLocal
from .models import Job
from .pipeline import (
    DEFAULT_DIVISIONS,
    NoteEvent,
    NoteExtractionStats,
    PitchTrackingDiagnostics,
    build_score_json,
    compute_confidence,
    convert_to_wav,
    detect_onsets,
    detect_onsets_from_wav,
    download_youtube_audio,
    ensure_dependencies,
    estimate_tempo,
    estimate_tempo_from_wav,
    ffprobe_duration,
    f0_to_note_events,
    load_midi_notes,
    map_notes_to_tab,
    post_process_basic_pitch_notes,
    post_process_notes,
    preprocess_guitar_stem,
    quantize_basic_pitch_notes,
    quantize_basic_pitch_notes_robust,
    render_musicxml_to_pdf,
    run_basic_pitch,
    run_basic_pitch_notes,
    run_demucs,
    run_demucs_pick_stem,
    run_pitch_tracker,
    trim_wav,
    write_score_musicxml,
    write_tab_musicxml,
    write_tab_outputs,
)
from .utils import ensure_dir, get_job_logger, write_json


def _update_job(db: Session, job_id: str, **kwargs) -> None:
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return
    for key, value in kwargs.items():
        setattr(job, key, value)
    db.commit()


def _fail_job(db: Session, job_id: str, message: str, logger) -> None:
    logger.error(message)
    _update_job(db, job_id, status="FAILED", error_message=message, stage="FAILED")


POLYPHONY_SCORE_THRESHOLD = 0.45


@dataclass
class TranscriptionResult:
    notes: List[NoteEvent]
    metrics: Dict[str, float]
    tempo_bpm: float
    tempo_source: str
    detected_tempo: Optional[float]
    extraction_stats: Optional[NoteExtractionStats]
    f0_preview_path: Optional[str]
    diagnostics: Optional[PitchTrackingDiagnostics]
    midi_path: Optional[str]


class PolyphonyFallback(Exception):
    def __init__(
        self,
        warning: str,
        stats: NoteExtractionStats,
        diagnostics: PitchTrackingDiagnostics,
        preview_path: Optional[str],
    ):
        super().__init__(warning)
        self.warning = warning
        self.stats = stats
        self.diagnostics = diagnostics
        self.preview_path = preview_path


def _write_f0_preview_csv(
    path: str,
    times: np.ndarray,
    f0_values: np.ndarray,
    confidence: np.ndarray,
    midi_smoothed: np.ndarray,
) -> None:
    length = min(len(times), len(f0_values), len(confidence), len(midi_smoothed))
    with open(path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time", "f0", "confidence", "midi_smoothed"])
        for idx in range(length):
            time_val = f"{times[idx]:.5f}"
            f0_val = f"{float(f0_values[idx]):.2f}" if np.isfinite(f0_values[idx]) else ""
            conf_val = f"{float(confidence[idx]):.4f}"
            midi_val = (
                f"{float(midi_smoothed[idx]):.2f}" if np.isfinite(midi_smoothed[idx]) else ""
            )
            writer.writerow([time_val, f0_val, conf_val, midi_val])


def _stage(db: Session, job_id: str, stage: str, progress: int, logger, message: Optional[str] = None) -> None:
    if message:
        logger.info(message)
    _update_job(db, job_id, stage=stage, progress=progress, status="RUNNING")


def _derive_title_artist(job: Job) -> Tuple[str, str]:
    title = "Transcription"
    artist = "Artiste inconnu"
    if job.input_filename:
        title = os.path.splitext(os.path.basename(job.input_filename))[0] or title
    if job.source_url:
        artist = "YouTube"
    return title, artist


def _process_monophonic_transcription(
    job: Job, candidate_path: str, logger, warnings: List[str], output_dir: str
) -> TranscriptionResult:
    tracker_result = run_pitch_tracker(candidate_path, logger)
    onsets = detect_onsets(tracker_result, logger=logger)
    onsets_json_path = os.path.join(output_dir, "onsets.json")
    with open(onsets_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "hopLength": tracker_result.hop_length,
                "sampleRate": tracker_result.sr,
                "onsetFrames": [int(x) for x in onsets.tolist()],
                "onsetTimes": [float(x) for x in (onsets * (tracker_result.hop_length / tracker_result.sr)).tolist()],
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    notes_raw, stats, diagnostics = f0_to_note_events(tracker_result, onset_frames=onsets)
    f0_preview_path = os.path.join(output_dir, "f0_preview.csv")
    _write_f0_preview_csv(
        f0_preview_path,
        diagnostics.times,
        tracker_result.f0,
        tracker_result.confidence,
        diagnostics.midi_smoothed,
    )
    logger.info(
        "Monophonic extraction: %d notes, instability %.2f, voiced ratio %.2f",
        stats.notes_count,
        stats.instability_ratio,
        stats.voiced_ratio,
    )
    if stats.polyphony_score >= POLYPHONY_SCORE_THRESHOLD:
        warning = "Signal trop polyphonique pour mode accordeur, bascule polyphonique."
        raise PolyphonyFallback(warning, stats, diagnostics, f0_preview_path)
    tempo_bpm, tempo_source = estimate_tempo(tracker_result.audio, tracker_result.sr, logger)
    tempo_detected = tempo_bpm if tempo_source != "default" else None
    if tempo_source == "default":
        warnings.append("Tempo inconnu, valeur par défaut utilisée.")
    processed, metrics = post_process_notes(
        notes_raw,
        job.quality,
        tempo_bpm,
        arrangement=(job.arrangement or "lead"),
        midi_min=int(getattr(job, "midi_min", 40) or 40),
        midi_max=int(getattr(job, "midi_max", 88) or 88),
        min_duration_ms=int(getattr(job, "note_min_duration_ms", 60) or 60),
        snap_tolerance_ms=int(getattr(job, "snap_tolerance_ms", 35) or 35),
        poly_max_notes=int(getattr(job, "poly_max_notes", 3) or 3),
    )
    if not processed:
        raise RuntimeError("Aucune note exploitable après post-traitement.")
    return TranscriptionResult(
        notes=processed,
        metrics=metrics,
        tempo_bpm=tempo_bpm,
        tempo_source=tempo_source,
        detected_tempo=tempo_detected,
        extraction_stats=stats,
        f0_preview_path=f0_preview_path,
        diagnostics=diagnostics,
        midi_path=None,
    )


def _process_basic_pitch_transcription(
    job: Job, candidate_path: str, logger, warnings: List[str], output_dir: str
) -> TranscriptionResult:
    midi_dir = os.path.join(output_dir, "midi")
    ensure_dir(midi_dir)
    midi_generated = run_basic_pitch(candidate_path, midi_dir, logger)
    midi_path = os.path.join(output_dir, "output.mid")
    shutil.copyfile(midi_generated, midi_path)
    notes, detected_tempo = load_midi_notes(midi_path)
    if detected_tempo:
        logger.info("BPM détecté depuis MIDI: %.2f", detected_tempo)
    else:
        logger.warning("Aucun BPM détecté depuis MIDI.")
    tempo_used = detected_tempo if detected_tempo and detected_tempo > 0 else SETTINGS.default_tempo_bpm
    tempo_source = "midi" if detected_tempo and detected_tempo > 0 else "default"
    if tempo_source == "default":
        warnings.append("Tempo inconnu, valeur par défaut utilisée.")
    processed, metrics = post_process_notes(
        notes,
        job.quality,
        tempo_used,
        arrangement=(job.arrangement or "lead"),
        midi_min=int(getattr(job, "midi_min", 40) or 40),
        midi_max=int(getattr(job, "midi_max", 88) or 88),
        min_duration_ms=int(getattr(job, "note_min_duration_ms", 60) or 60),
        snap_tolerance_ms=int(getattr(job, "snap_tolerance_ms", 35) or 35),
        poly_max_notes=int(getattr(job, "poly_max_notes", 3) or 3),
    )
    if not processed:
        raise RuntimeError("Aucune note exploitable après post-traitement.")
    return TranscriptionResult(
        notes=processed,
        metrics=metrics,
        tempo_bpm=tempo_used,
        tempo_source=tempo_source,
        detected_tempo=detected_tempo,
        extraction_stats=None,
        f0_preview_path=None,
        diagnostics=None,
        midi_path=midi_path,
    )


def _process_best_free_transcription(
    job: Job,
    stem_path: str,
    logger,
    warnings: List[str],
    output_dir: str,
    *,
    confidence_threshold: float = 0.2,
    divisions: int = 8,
) -> TranscriptionResult:
    """Pipeline recommandé: stem guitare (pré-traité) -> BasicPitch raw -> post-process -> tempo -> quantize."""
    # Tempo détecté sur l'audio (pas sur MIDI BasicPitch, qui est typiquement à un tempo minimal).
    tempo_bpm, tempo_source = estimate_tempo_from_wav(stem_path, logger)
    tempo_detected = tempo_bpm if tempo_source != "default" else None
    if tempo_source == "default":
        warnings.append("Tempo inconnu, valeur par défaut utilisée.")

    basic_pitch_dir = os.path.join(output_dir, "basic_pitch")
    ensure_dir(basic_pitch_dir)
    raw_path = os.path.join(output_dir, "raw_basic_pitch.json")
    clean_path = os.path.join(output_dir, "clean_notes.json")
    onsets_json_path = os.path.join(output_dir, "onsets.json")

    raw_notes: List[Dict[str, object]] = []
    midi_path = os.path.join(output_dir, "output.mid")

    try:
        raw_notes_any, midi_generated = run_basic_pitch_notes(stem_path, basic_pitch_dir, logger=logger)
        raw_notes = [dict(item) for item in raw_notes_any]
        shutil.copyfile(midi_generated, midi_path)
    except Exception as exc:  # noqa: BLE001
        # Fallback: CLI -> MIDI, on synthétise une confidence via velocity.
        warnings.append(f"Basic Pitch python indisponible, fallback CLI/MIDI ({exc}).")
        midi_generated = run_basic_pitch(stem_path, basic_pitch_dir, logger)
        shutil.copyfile(midi_generated, midi_path)
        midi_notes, _ = load_midi_notes(midi_path)
        for ev in midi_notes:
            raw_notes.append(
                {
                    "start": float(ev.start),
                    "end": float(ev.start + ev.duration),
                    "pitch": int(ev.pitch),
                    "amplitude": float(clamp(ev.velocity / 127.0, 0.0, 1.0)),
                    "confidence": float(clamp(ev.velocity / 127.0, 0.0, 1.0)),
                }
            )

    write_json(
        raw_path,
        {
            "notes": raw_notes,
        },
    )

    onsets = []
    if bool(getattr(job, "onset_detection", 1)):
        onsets = detect_onsets_from_wav(stem_path, sr=SETTINGS.sample_rate, hop_length=512, logger=logger)
    with open(onsets_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "hopLength": 512,
                "sampleRate": int(SETTINGS.sample_rate),
                "onsetTimes": [float(x) for x in onsets],
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    arrangement = (getattr(job, "arrangement", None) or "lead").lower()
    lead_mode = arrangement == "lead"
    cleaned_notes, clean_stats = post_process_basic_pitch_notes(
        raw_notes,
        confidence_threshold=confidence_threshold,
        midi_min=int(getattr(job, "midi_min", 40) or 40),
        midi_max=int(getattr(job, "midi_max", 88) or 88),
        lead_mode=lead_mode,
        onsets=onsets,
        onset_window_ms=int(getattr(job, "onset_window_ms", 60) or 60),
        max_jump_semitones=int(getattr(job, "max_jump_semitones", 7) or 7),
        onset_align=bool(getattr(job, "onset_align", 1)),
        onset_gate=bool(getattr(job, "onset_gate", 0)),
        min_duration_seconds=max(0.01, float(getattr(job, "note_min_duration_ms", 60) or 60) / 1000.0),
    )
    counts = (clean_stats or {}).get("counts") if isinstance(clean_stats, dict) else None
    if isinstance(counts, dict):
        logger.info(
            "best_free notes: raw=%s, afterFilter=%s, afterOnset=%s, afterMerge=%s, afterHarm=%s, afterJump=%s, afterLead=%s",
            counts.get("raw"),
            counts.get("afterFilter"),
            counts.get("afterOnsetGate"),
            counts.get("afterMerge"),
            counts.get("afterHarmonics"),
            counts.get("afterJumpFilter"),
            counts.get("afterLead"),
        )

    write_json(
        clean_path,
        {
            "notes": cleaned_notes,
            "stats": clean_stats,
        },
    )

    # Quantization: divisions=8, grille dépend de la qualité / override gridResolution.
    quality = (job.quality or "fast").lower()
    grid_resolution = (getattr(job, "grid_resolution", None) or "auto").lower()
    if grid_resolution == "eighth":
        grid_ticks = 4
        grid_used = "eighth"
    elif grid_resolution == "sixteenth":
        grid_ticks = 2
        grid_used = "sixteenth"
    else:
        grid_ticks = 4 if quality == "fast" else 2  # spb/2 ou spb/4
        grid_used = "auto-fast" if quality == "fast" else "auto-normal"
    allowed = [2, 4, 8, 16, 32]  # {1/16, 1/8, 1/4, 1/2, 1} en ticks (divisions=8)
    quantized, quant_debug = quantize_basic_pitch_notes_robust(
        cleaned_notes,
        tempo_bpm=tempo_bpm,
        divisions=divisions,
        grid_ticks=grid_ticks,
        min_duration_ticks=1,
        allowed_duration_ticks=allowed,
    )

    # Metrics pour compute_confidence (fallback simple + densité).
    metrics = {
        "short_ratio": 0.0,
        "out_of_range_ratio": 0.0,
    }

    # On expose aussi quelques infos de debug via `metrics` (agrégé) pour `debug.json`.
    counts = (clean_stats or {}).get("counts") if isinstance(clean_stats, dict) else None
    if not isinstance(counts, dict):
        counts = {}
    metrics.update(
        {
            "bestFreeRawNotes": float(len(raw_notes)),
            "bestFreeAfterFilter": float(counts.get("afterFilter") or 0),
            "bestFreeAfterOnsetGate": float(counts.get("afterOnsetGate") or 0),
            "bestFreeAfterMerge": float(counts.get("afterMerge") or 0),
            "bestFreeAfterHarmonics": float(counts.get("afterHarmonics") or 0),
            "bestFreeAfterJumpFilter": float(counts.get("afterJumpFilter") or 0),
            "bestFreeAfterLead": float(counts.get("afterLead") or 0),
            "bestFreeCleanNotes": float(len(cleaned_notes)),
            "bestFreeQuantizedNotes": float(len(quantized)),
            "bestFreeGridTicks": float(grid_ticks),
            "bestFreeDivisions": float(divisions),
            "bestFreeGridUsed": 1.0 if grid_used else 0.0,
            "bestFreeArrangementIsLead": 1.0 if lead_mode else 0.0,
            "bestFreeQuantizationErrors": float(int((quant_debug or {}).get("quantizationErrorsCount") or 0)),
        }
    )

    if not quantized:
        raise RuntimeError("Aucune note exploitable après post-traitement best_free.")

    return TranscriptionResult(
        notes=quantized,
        metrics=metrics,
        tempo_bpm=tempo_bpm,
        tempo_source=tempo_source,
        detected_tempo=tempo_detected,
        extraction_stats=None,
        f0_preview_path=None,
        diagnostics=None,
        midi_path=midi_path,
    )


def process_job(job_id: str) -> None:
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        db.close()
        return

    job_dir = os.path.join(SETTINGS.data_dir, job_id)
    ensure_dir(job_dir)
    logs_path = os.path.join(job_dir, "logs.txt")
    logger = get_job_logger(job_id, logs_path)
    _update_job(db, job_id, logs_path=logs_path)
    warnings: List[str] = []

    try:
        _stage(db, job_id, "VALIDATION", 5, logger, "Validation des paramètres.")
        require_youtube = job.input_type == "youtube"
        ensure_dependencies(require_youtube=require_youtube)

        input_dir = os.path.join(job_dir, "input")
        output_dir = os.path.join(job_dir, "output")
        ensure_dir(input_dir)
        ensure_dir(output_dir)

        _stage(db, job_id, "EXTRACTION_AUDIO", 15, logger, "Extraction et conversion audio.")
        if job.input_type == "youtube":
            if not job.source_url:
                raise RuntimeError("URL YouTube manquante.")
            input_path = download_youtube_audio(job.source_url, input_dir, logger)
        else:
            if not job.input_path or not os.path.exists(job.input_path):
                raise RuntimeError("Fichier upload manquant.")
            input_path = job.input_path

        wav_path = os.path.join(input_dir, "input.wav")
        convert_to_wav(input_path, wav_path, logger)
        duration = ffprobe_duration(wav_path)
        if duration > SETTINGS.max_duration_seconds:
            raise RuntimeError("Durée audio trop longue (max 12 minutes).")
        _update_job(db, job_id, duration_seconds=duration)

        if job.start_seconds is not None or job.end_seconds is not None:
            start = float(job.start_seconds or 0)
            end = float(job.end_seconds) if job.end_seconds is not None else duration
            if start >= duration:
                raise RuntimeError("startSeconds dépasse la durée audio.")
            if end > duration:
                end = duration
                warnings.append("Fin tronquée à la durée audio.")
            segment_duration = end - start
            if segment_duration <= 0:
                raise RuntimeError("Segment invalide (endSeconds doit être > startSeconds).")
            if segment_duration > SETTINGS.max_duration_seconds:
                raise RuntimeError("Segment audio trop long (max 12 minutes).")
            trimmed_path = os.path.join(input_dir, "input_trimmed.wav")
            trim_wav(wav_path, trimmed_path, start, end, logger)
            wav_path = trimmed_path
            duration = segment_duration
            _update_job(db, job_id, duration_seconds=duration)
            warnings.append("Audio tronqué via timecode.")

        candidate_path = wav_path
        stem_used = None
        stem_preprocess = None
        stem_guitar_path = None
        if not job.input_is_isolated:
            _stage(db, job_id, "ISOLATION_DEMUCS", 35, logger, "Isolation de la guitare (Demucs).")
            demucs_dir = os.path.join(output_dir, "demucs")
            ensure_dir(demucs_dir)
            if job.transcription_mode == "best_free":
                candidate_path, stem_used = run_demucs_pick_stem(wav_path, demucs_dir, logger, preferred="other")
            else:
                candidate_path = run_demucs(wav_path, demucs_dir, logger)
        else:
            warnings.append("Piste guitare isolée: Demucs ignoré.")

        if job.transcription_mode == "best_free":
            _stage(
                db,
                job_id,
                "PREPROCESS_GUITAR_STEM",
                45,
                logger,
                "Pré-traitement audio (filtre guitare + normalisation).",
            )
            stem_guitar_path = os.path.join(output_dir, "stem_guitar.wav")
            stem_preprocess = preprocess_guitar_stem(
                candidate_path,
                stem_guitar_path,
                logger=logger,
                hp_hz=80.0,
                lp_hz=7000.0,
                target_rms_dbfs=-18.0,
                apply_gate=(job.quality or "").lower() == "fast",
            )
            candidate_path = stem_guitar_path

        pitch_mode = "monophonic"
        fallback_stats = None
        fallback_preview = None
        if job.transcription_mode == "best_free":
            _stage(
                db,
                job_id,
                "TRANSCRIPTION_BEST_FREE",
                55,
                logger,
                "Transcription best_free (Demucs + Basic Pitch + post-processing).",
            )
            result = _process_best_free_transcription(
                job,
                candidate_path,
                logger,
                warnings,
                output_dir,
                confidence_threshold=float(getattr(job, "confidence_threshold", 0.2) or 0.2),
                divisions=8,
            )
            pitch_mode = "best_free"
        elif job.transcription_mode == "monophonic_tuner":
            _stage(db, job_id, "TRANSCRIPTION_MONOPHONIC", 55, logger, "Transcription monophonique type accordeur.")
            try:
                result = _process_monophonic_transcription(
                    job, candidate_path, logger, warnings, output_dir
                )
                pitch_mode = "monophonic"
            except PolyphonyFallback as exc:
                warnings.append(exc.warning)
                fallback_stats = exc.stats
                fallback_preview = exc.preview_path
                _stage(
                    db,
                    job_id,
                    "TRANSCRIPTION_BASIC_PITCH",
                    55,
                    logger,
                    "Transcription audio -> MIDI.",
                )
                result = _process_basic_pitch_transcription(
                    job, candidate_path, logger, warnings, output_dir
                )
                pitch_mode = "polyphonic_basic_pitch"
        else:
            _stage(
                db,
                job_id,
                "TRANSCRIPTION_BASIC_PITCH",
                55,
                logger,
                "Transcription audio -> MIDI.",
            )
            result = _process_basic_pitch_transcription(
                job, candidate_path, logger, warnings, output_dir
            )
            pitch_mode = "polyphonic_basic_pitch"
        note_events_count = len(result.notes)
        processed = result.notes
        metrics = result.metrics
        tempo_used = result.tempo_bpm
        tempo_source = result.tempo_source
        detected_tempo = result.detected_tempo
        extraction_stats = result.extraction_stats or fallback_stats
        midi_path = result.midi_path
        _stage(
            db,
            job_id,
            "POST_PROCESSING",
            70,
            logger,
            "Nettoyage et quantification des notes.",
        )

        divisions = 8 if job.transcription_mode == "best_free" else DEFAULT_DIVISIONS
        measure_ticks = divisions * 4
        logger.info(
            "Préparation score JSON : tempo=%s, divisions=%s, measureTicks=%s",
            tempo_used,
            divisions,
            measure_ticks,
        )
        score_json, score_json_notes = build_score_json(
            processed,
            job.quality,
            tempo_used,
            tempo_source,
            warnings,
            divisions=divisions,
        )
        score_json_path = os.path.join(output_dir, "score.json")
        write_json(score_json_path, score_json)

        _stage(db, job_id, "TAB_GENERATION", 85, logger, "Génération de la tablature.")
        logger.info(
            "Préparation tablature JSON : tempo=%s, divisions=%s, measureTicks=%s",
            tempo_used,
            divisions,
            measure_ticks,
        )
        tab_notes, tuning_warning, playability_debug = map_notes_to_tab(
            processed,
            job.tuning,
            job.capo,
            hand_span=job.hand_span,
            prefer_low_frets=bool(job.prefer_low_frets),
            max_fret=int(getattr(job, "tab_max_fret", 15) or 15),
            max_fret_span_chord=int(getattr(job, "tab_max_fret_span_chord", 5) or 5),
            max_position_jump=int(getattr(job, "tab_max_position_jump", 4) or 4),
            max_notes_per_chord=int(getattr(job, "tab_max_notes_per_chord", 3) or 3),
        )
        fingering_debug_path = os.path.join(output_dir, "fingering_debug.json")
        write_json(fingering_debug_path, playability_debug)
        if tuning_warning:
            warnings.append(tuning_warning)
        tab_txt_path = os.path.join(output_dir, "tab.txt")
        tab_json_path = os.path.join(output_dir, "tab.json")
        tab_json, tab_txt_notes, tab_warnings, tab_json_notes = write_tab_outputs(
            tab_notes=tab_notes,
            tuning=job.tuning,
            capo=job.capo,
            quality=job.quality,
            tempo_bpm=tempo_used,
            tempo_source=tempo_source,
            warnings=warnings,
            tab_txt_path=tab_txt_path,
            tab_json_path=tab_json_path,
            logger=logger,
            divisions=divisions,
            measures_per_system=int(getattr(job, "tab_measures_per_system", 2) or 2),
            wrap_columns=int(getattr(job, "tab_wrap_columns", 80) or 80),
            token_width=int(getattr(job, "tab_token_width", 3) or 3),
        )
        warnings.extend(tab_warnings)

        _stage(db, job_id, "EXPORT", 95, logger, "Export des fichiers.")
        title, artist = _derive_title_artist(job)
        tab_musicxml_path = os.path.join(output_dir, "result_tab.musicxml")
        _, tab_musicxml_notes_count, _ = write_tab_musicxml(
            tab_json=tab_json,
            tuning=job.tuning,
            capo=job.capo,
            tempo_bpm=tempo_used,
            title=title,
            artist=artist,
            annotations=None,
            output_path=tab_musicxml_path,
            logger=logger,
        )
        tab_pdf_path = os.path.join(output_dir, "result_tab.pdf")
        render_musicxml_to_pdf(tab_musicxml_path, tab_pdf_path, logger)
        if not os.path.exists(tab_pdf_path) or os.path.getsize(tab_pdf_path) == 0:
            raise RuntimeError("PDF tablature non généré.")
        score_musicxml_path = os.path.join(output_dir, "result_score.musicxml")
        _, score_musicxml_notes_count = write_score_musicxml(
            score_json=score_json,
            tempo_bpm=tempo_used,
            title=title,
            artist=artist,
            annotations=None,
            output_path=score_musicxml_path,
            logger=logger,
        )
        score_pdf_path = os.path.join(output_dir, "result_score.pdf")
        render_musicxml_to_pdf(score_musicxml_path, score_pdf_path, logger)
        if not os.path.exists(score_pdf_path) or os.path.getsize(score_pdf_path) == 0:
            raise RuntimeError("PDF partition non généré.")

        score_metadata = score_json.get("metadata", {})
        confidence = compute_confidence(metrics, processed)
        if confidence < 0.35:
            warnings.append("Confidence faible: piste guitare isolée recommandée.")

        chunk_stats = extraction_stats
        pitch_chunks = []
        if chunk_stats:
            pitch_chunks.append(
                {
                    "start": 0.0,
                    "end": duration,
                    "mode": pitch_mode,
                    "voiced_ratio": chunk_stats.voiced_ratio,
                    "note_change_rate": chunk_stats.note_change_rate,
                    "jump_rate": chunk_stats.jump_rate,
                    "polyphony_score": chunk_stats.polyphony_score,
                    "low_energy_ratio": chunk_stats.low_energy_ratio,
                }
            )
        preview_path = result.f0_preview_path or fallback_preview
        debug_info = {
            "midiBpmDetected": detected_tempo,
            "tempoUsedForQuantization": tempo_used,
            "tempoSource": tempo_source,
            "divisions": divisions,
            "measureTicks": measure_ticks,
            "scoreWrittenOctaveShift": score_metadata.get("scoreWrittenOctaveShift"),
            "noteEventsCount": note_events_count,
            "scoreJsonNotesCount": score_json_notes,
            "tabJsonNotesCount": tab_json_notes,
            "scoreMusicXmlNotesCount": score_musicxml_notes_count,
            "tabMusicXmlNotesCount": tab_musicxml_notes_count,
            "playabilityScore": playability_debug.get("playabilityScore"),
            "playabilityCost": playability_debug.get("totalCost"),
            "avgShift": playability_debug.get("avgShift"),
            "maxStretch": playability_debug.get("maxStretch"),
            "unreachableCount": playability_debug.get("unreachableCount"),
            "handSpan": job.hand_span,
            "preferLowFrets": bool(job.prefer_low_frets),
            "fingeringDebugPath": fingering_debug_path,
        }
        if job.transcription_mode == "best_free":
            debug_info.update(
                {
                    "stemUsed": stem_used,
                    "stemGuitarPath": stem_guitar_path,
                    "stemPreprocess": stem_preprocess,
                    "arrangement": job.arrangement,
                    "confidenceThreshold": job.confidence_threshold,
                    "onsetWindowMs": job.onset_window_ms,
                    "maxJumpSemitones": job.max_jump_semitones,
                    "gridResolution": job.grid_resolution,
                    "basicPitchNotesCountRaw": int(metrics.get("bestFreeRawNotes") or 0),
                    "basicPitchNotesCountAfterFilter": int(metrics.get("bestFreeAfterFilter") or 0),
                    "basicPitchNotesCountAfterOnsetGate": int(metrics.get("bestFreeAfterOnsetGate") or 0),
                    "basicPitchNotesCountAfterMerge": int(metrics.get("bestFreeAfterMerge") or 0),
                    "basicPitchNotesCountAfterHarmonics": int(metrics.get("bestFreeAfterHarmonics") or 0),
                    "basicPitchNotesCountAfterJumpFilter": int(metrics.get("bestFreeAfterJumpFilter") or 0),
                    "basicPitchNotesCountAfterLead": int(metrics.get("bestFreeAfterLead") or 0),
                    "basicPitchNotesCountQuantized": int(metrics.get("bestFreeQuantizedNotes") or 0),
                    "quantizationGridTicks": int(metrics.get("bestFreeGridTicks") or 0),
                    "quantizationDivisions": int(metrics.get("bestFreeDivisions") or divisions),
                    "quantizationErrorsCount": int(metrics.get("bestFreeQuantizationErrors") or 0),
                    "rawBasicPitchPath": os.path.join(output_dir, "raw_basic_pitch.json"),
                    "cleanNotesPath": os.path.join(output_dir, "clean_notes.json"),
                    "onsetsPath": os.path.join(output_dir, "onsets.json"),
                }
            )
        debug_info.update(
            {
                "transcriptionMode": job.transcription_mode,
                "tempoDetected": detected_tempo,
                "voiced_ratio": extraction_stats.voiced_ratio if extraction_stats else None,
                "jump_rate": extraction_stats.jump_rate if extraction_stats else None,
                "note_count_before_filters": note_events_count,
                "note_count_after_filters": len(processed),
                "avgVoicedRatio": extraction_stats.voiced_ratio if extraction_stats else None,
                "instabilityRatio": extraction_stats.instability_ratio if extraction_stats else None,
                "pitchMedian": extraction_stats.pitch_median if extraction_stats else None,
                "pitchMin": extraction_stats.pitch_min if extraction_stats else None,
                "pitchMax": extraction_stats.pitch_max if extraction_stats else None,
                "pitchMode": pitch_mode,
                "pitchChunks": pitch_chunks,
                "f0PreviewPath": preview_path,
                "onsetsPath": os.path.join(output_dir, "onsets.json")
                if job.transcription_mode == "monophonic_tuner"
                else None,
                "polyphonyScore": chunk_stats.polyphony_score if chunk_stats else None,
                "lowEnergyRatio": chunk_stats.low_energy_ratio if chunk_stats else None,
                "warnings": list(warnings),
            }
        )
        debug_json_path = os.path.join(output_dir, "debug.json")
        write_json(debug_json_path, debug_info)
        _update_job(
            db,
            job_id,
            status="DONE",
            progress=100,
            stage="DONE",
            confidence=confidence,
            warnings_json=json.dumps(warnings),
            tempo_bpm=tempo_used,
            tab_txt_path=tab_txt_path,
            tab_json_path=tab_json_path,
            musicxml_path=tab_musicxml_path,
            pdf_path=tab_pdf_path,
            score_json_path=score_json_path,
            score_musicxml_path=score_musicxml_path,
            score_pdf_path=score_pdf_path,
            midi_path=midi_path,
            fingering_debug_path=fingering_debug_path,
            debug_info_json=json.dumps(debug_info),
        )
    except Exception as exc:  # noqa: BLE001
        _fail_job(db, job_id, str(exc), logger)
    finally:
        db.close()
