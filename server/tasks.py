import json
import os
import shutil
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from .config import SETTINGS
from .db import SessionLocal
from .models import Job
from .pipeline import (
    compute_confidence,
    convert_to_wav,
    download_youtube_audio,
    ensure_dependencies,
    ffprobe_duration,
    load_midi_notes,
    map_notes_to_tab,
    post_process_notes,
    render_musicxml_to_pdf,
    run_basic_pitch,
    run_demucs,
    trim_wav,
    write_tab_musicxml,
    write_tab_outputs,
)
from .utils import ensure_dir, get_job_logger


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

        _stage(db, job_id, "TRANSCRIPTION_BASIC_PITCH", 55, logger, "Transcription audio -> MIDI.")
        midi_dir = os.path.join(output_dir, "midi")
        ensure_dir(midi_dir)
        midi_generated = run_basic_pitch(candidate_path, midi_dir, logger)
        midi_path = os.path.join(output_dir, "output.mid")
        shutil.copyfile(midi_generated, midi_path)

        _stage(db, job_id, "POST_PROCESSING", 70, logger, "Nettoyage et quantification des notes.")
        notes, detected_tempo = load_midi_notes(midi_path)
        tempo_bpm = detected_tempo or SETTINGS.default_tempo_bpm
        tempo_source = "estimated" if detected_tempo else "unknown"
        if tempo_source == "unknown":
            warnings.append("Tempo inconnu, valeur par défaut utilisée.")
        processed, metrics = post_process_notes(notes, job.quality)
        if not processed:
            raise RuntimeError("Aucune note exploitable après post-traitement.")

        _stage(db, job_id, "TAB_GENERATION", 85, logger, "Génération de la tablature.")
        tab_notes, tuning_warning = map_notes_to_tab(processed, job.tuning, job.capo)
        if tuning_warning:
            warnings.append(tuning_warning)
        tab_txt_path = os.path.join(output_dir, "tab.txt")
        tab_json_path = os.path.join(output_dir, "tab.json")
        write_tab_outputs(
            tab_notes=tab_notes,
            tuning=job.tuning,
            capo=job.capo,
            quality=job.quality,
            tempo_bpm=tempo_bpm,
            tempo_source=tempo_source,
            warnings=warnings,
            tab_txt_path=tab_txt_path,
            tab_json_path=tab_json_path,
        )

        _stage(db, job_id, "EXPORT", 95, logger, "Export des fichiers.")
        title, artist = _derive_title_artist(job)
        musicxml_path = os.path.join(output_dir, "result.musicxml")
        write_tab_musicxml(
            tab_notes=tab_notes,
            tuning=job.tuning,
            capo=job.capo,
            tempo_bpm=tempo_bpm,
            title=title,
            artist=artist,
            annotations=None,
            output_path=musicxml_path,
        )
        pdf_path = os.path.join(output_dir, "result.pdf")
        render_musicxml_to_pdf(musicxml_path, pdf_path, logger)

        confidence = compute_confidence(metrics, processed)

        if confidence < 0.35:
            warnings.append("Confidence faible: piste guitare isolée recommandée.")

        _update_job(
            db,
            job_id,
            status="DONE",
            progress=100,
            stage="DONE",
            confidence=confidence,
            warnings_json=json.dumps(warnings),
            tempo_bpm=tempo_bpm,
            tab_txt_path=tab_txt_path,
            tab_json_path=tab_json_path,
            musicxml_path=musicxml_path,
            pdf_path=pdf_path,
            midi_path=midi_path,
        )
    except Exception as exc:  # noqa: BLE001
        _fail_job(db, job_id, str(exc), logger)
    finally:
        db.close()
