import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from .config import SETTINGS
from .db import SessionLocal
from .models import Job
from .pipeline import (
    DEFAULT_DIVISIONS,
    NoteEvent,
    NoteExtractionStats,
    build_score_json,
    compute_confidence,
    convert_to_wav,
    download_youtube_audio,
    ensure_dependencies,
    estimate_tempo,
    ffprobe_duration,
    f0_to_note_events,
    load_midi_notes,
    map_notes_to_tab,
    post_process_notes,
    render_musicxml_to_pdf,
    run_basic_pitch,
    run_demucs,
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
    job: Job, candidate_path: str, logger, warnings: List[str]
) -> Tuple[List[NoteEvent], Dict[str, float], float, str, Optional[float], Optional[NoteExtractionStats]]:
    tracker = run_pitch_tracker(candidate_path, logger)
    notes, stats = f0_to_note_events(tracker)
    logger.info(
        "Monophonic extraction: %d notes, instability %.2f, voiced ratio %.2f",
        stats.notes_count,
        stats.instability_ratio,
        stats.voiced_ratio,
    )
    if stats.instability_ratio > 0.4 and stats.notes_count > 0:
        raise RuntimeError("Signal trop polyphonique, essayez le mode polyphonique.")
    tempo_bpm, tempo_source = estimate_tempo(tracker.audio, tracker.sr, logger)
    tempo_detected = tempo_bpm if tempo_source != "default" else None
    if tempo_source == "default":
        warnings.append("Tempo inconnu, valeur par défaut utilisée.")
    processed, metrics = post_process_notes(notes, job.quality, tempo_bpm)
    if not processed:
        raise RuntimeError("Aucune note exploitable après post-traitement.")
    return processed, metrics, tempo_bpm, tempo_source, tempo_detected, stats


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
        if not job.input_is_isolated:
            _stage(db, job_id, "ISOLATION_DEMUCS", 35, logger, "Isolation de la guitare (Demucs).")
            demucs_dir = os.path.join(output_dir, "demucs")
            ensure_dir(demucs_dir)
            candidate_path = run_demucs(wav_path, demucs_dir, logger)
        else:
            warnings.append("Piste guitare isolée: Demucs ignoré.")

        if job.transcription_mode == "monophonic_tuner":
            _stage(db, job_id, "TRANSCRIPTION_MONOPHONIC", 55, logger, "Transcription monophonique type accordeur.")
            (
                processed,
                metrics,
                tempo_used,
                tempo_source,
                detected_tempo,
                extraction_stats,
            ) = _process_monophonic_transcription(job, candidate_path, logger, warnings)
            note_events_count = len(processed)
            midi_path = None
        else:
            _stage(db, job_id, "TRANSCRIPTION_BASIC_PITCH", 55, logger, "Transcription audio -> MIDI.")
            midi_dir = os.path.join(output_dir, "midi")
            ensure_dir(midi_dir)
            midi_generated = run_basic_pitch(candidate_path, midi_dir, logger)
            midi_path = os.path.join(output_dir, "output.mid")
            shutil.copyfile(midi_generated, midi_path)
            _stage(db, job_id, "POST_PROCESSING", 70, logger, "Nettoyage et quantification des notes.")
            notes, detected_tempo = load_midi_notes(midi_path)
            if detected_tempo:
                logger.info("BPM détecté depuis MIDI: %.2f", detected_tempo)
            else:
                logger.warning("Aucun BPM détecté depuis MIDI.")
            tempo_used = detected_tempo if detected_tempo and detected_tempo > 0 else SETTINGS.default_tempo_bpm
            tempo_source = "midi" if detected_tempo and detected_tempo > 0 else "default"
            if tempo_source == "default":
                warnings.append("Tempo inconnu, valeur par défaut utilisée.")
            logger.info("Tempo utilisé pour quantization: %.2f BPM", tempo_used)
            processed, metrics = post_process_notes(notes, job.quality, tempo_used)
            if not processed:
                raise RuntimeError("Aucune note exploitable après post-traitement.")
            note_events_count = len(processed)
            extraction_stats = None

        divisions = DEFAULT_DIVISIONS
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
            "handSpan": job.hand_span,
            "preferLowFrets": bool(job.prefer_low_frets),
            "fingeringDebugPath": fingering_debug_path,
        }
        debug_info.update(
            {
                "transcriptionMode": job.transcription_mode,
                "tempoDetected": detected_tempo,
                "avgVoicedRatio": extraction_stats.voiced_ratio if extraction_stats else None,
                "instabilityRatio": extraction_stats.instability_ratio if extraction_stats else None,
                "pitchMedian": extraction_stats.pitch_median if extraction_stats else None,
                "pitchMin": extraction_stats.pitch_min if extraction_stats else None,
                "pitchMax": extraction_stats.pitch_max if extraction_stats else None,
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
