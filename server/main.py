import json
import os
import re
import uuid
from typing import Optional

import redis
from fastapi import Body, FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import SETTINGS
from .auth import install_auth_middleware
from .db import SessionLocal, init_db
from .models import Job
from .schemas import (
    CreateJobResponse,
    JobMetadata,
    JobResultResponse,
    JobStatusResponse,
    LibraryItem,
    LibraryResponse,
    YoutubeRequest,
)
from .pipeline import (
    MAX_HAND_POS,
    compare_tab_json_and_musicxml,
    load_tab_json,
    render_musicxml_to_pdf,
)
from .tasks import process_job
from .utils import ensure_dir, get_job_logger, parse_last_musescore_run

app = FastAPI(title="TabScore Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[SETTINGS.frontend_origin],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Range"],
    expose_headers=["Content-Length", "Content-Type", "Accept-Ranges"],
)
install_auth_middleware(app)

ensure_dir(SETTINGS.data_dir)

redis_conn = redis.Redis.from_url(SETTINGS.redis_url)


def _queue():
    from rq import Queue

    return Queue("tabscore", connection=redis_conn)


def _is_youtube_url(url: str) -> bool:
    return bool(re.match(r"^https?://(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com)/", url))


def _save_upload(audio: UploadFile, job_dir: str) -> str:
    ensure_dir(job_dir)
    filename = audio.filename or "upload.bin"
    extension = os.path.splitext(filename)[1].lower()
    content_type = (audio.content_type or "").lower()
    allowed_ext = {".mp3", ".wav", ".m4a", ".aac"}
    mime_map = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/m4a": ".m4a",
        "audio/mp4": ".m4a",
        "audio/aac": ".aac",
    }
    if extension not in allowed_ext:
        inferred = mime_map.get(content_type)
        if inferred:
            extension = inferred
        elif content_type.startswith("audio/"):
            extension = ".wav"
        else:
            raise HTTPException(
                status_code=400,
                detail="Format audio non supporté (mp3, wav, m4a, aac uniquement).",
            )
    output_path = os.path.join(job_dir, f"raw{extension}")
    max_bytes = SETTINGS.max_upload_mb * 1024 * 1024
    size = 0
    with open(output_path, "wb") as f:
        while True:
            chunk = audio.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                f.close()
                os.remove(output_path)
                raise HTTPException(status_code=400, detail="Fichier trop volumineux (max 50MB).")
            f.write(chunk)
    return output_path


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/jobs", response_model=CreateJobResponse)
async def create_job(
    outputType: str = Query(...),
    tuning: str = Query("EADGBE"),
    capo: int = Query(0),
    quality: str = Query("fast"),
    transcriptionMode: str = Query("best_free"),
    arrangement: str = Query("lead"),
    confidenceThreshold: float = Query(0.35),
    onsetWindowMs: int = Query(60),
    maxJumpSemitones: int = Query(7),
    gridResolution: str = Query("auto"),
    mode: Optional[str] = Query(None),
    target: str = Query("GUITAR_BEST_EFFORT"),
    handSpan: int = Query(4),
    preferLowFrets: bool = Query(False),
    inputIsIsolatedGuitar: bool = Query(False),
    startSeconds: Optional[int] = Query(None),
    endSeconds: Optional[int] = Query(None),
    audio: UploadFile = File(None),
    body: Optional[YoutubeRequest] = Body(default=None),
    request: Request = None,
):
    output_type = outputType.lower()
    quality = quality.lower()
    transcription_mode = transcriptionMode.lower()
    arrangement = arrangement.lower()
    grid_resolution = gridResolution.lower()
    if transcription_mode not in ("monophonic_tuner", "polyphonic_basic_pitch", "best_free"):
        raise HTTPException(
            status_code=400,
            detail="transcriptionMode doit être 'monophonic_tuner', 'polyphonic_basic_pitch' ou 'best_free'.",
        )
    if arrangement not in ("lead", "poly"):
        raise HTTPException(status_code=400, detail="arrangement doit être 'lead' ou 'poly'.")
    if grid_resolution not in ("auto", "eighth", "sixteenth"):
        raise HTTPException(status_code=400, detail="gridResolution doit être 'auto', 'eighth' ou 'sixteenth'.")
    if output_type not in ("tab", "score", "both"):
        raise HTTPException(status_code=400, detail="outputType doit être tab, score ou both.")
    if audio is None and body is None and request is not None:
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = None
        if isinstance(payload, dict) and payload.get("youtubeUrl"):
            body = YoutubeRequest(youtubeUrl=payload["youtubeUrl"])
    if audio is None and body is None:
        raise HTTPException(status_code=400, detail="Fichier audio ou youtubeUrl requis.")
    if audio is not None and body is not None:
        raise HTTPException(status_code=400, detail="Choisissez soit un upload audio, soit un lien YouTube.")

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(SETTINGS.data_dir, job_id)
    input_path = None
    source_url = None
    input_type = "upload" if audio is not None else "youtube"

    if audio is not None:
        input_path = _save_upload(audio, os.path.join(job_dir, "input"))
    else:
        if body is None or not body.youtubeUrl:
            raise HTTPException(status_code=400, detail="youtubeUrl requis.")
        if not _is_youtube_url(body.youtubeUrl):
            raise HTTPException(status_code=400, detail="URL YouTube invalide (youtube.com ou youtu.be uniquement).")
        source_url = body.youtubeUrl

    db = SessionLocal()
    capo = max(0, min(12, capo))
    confidence_threshold = float(confidenceThreshold or 0.0)
    if confidence_threshold <= 0.0 or confidence_threshold > 1.0:
        raise HTTPException(status_code=400, detail="confidenceThreshold doit être dans (0, 1].")
    onset_window_ms = int(onsetWindowMs)
    onset_window_ms = max(10, min(250, onset_window_ms))
    max_jump = int(maxJumpSemitones)
    max_jump = max(1, min(24, max_jump))
    if startSeconds is not None and startSeconds < 0:
        raise HTTPException(status_code=400, detail="startSeconds doit être >= 0.")
    if endSeconds is not None and endSeconds < 0:
        raise HTTPException(status_code=400, detail="endSeconds doit être >= 0.")
    if startSeconds is not None and endSeconds is not None and endSeconds <= startSeconds:
        raise HTTPException(status_code=400, detail="endSeconds doit être > startSeconds.")
    hand_span = max(1, min(MAX_HAND_POS, handSpan))
    job = Job(
        id=job_id,
        status="PENDING",
        progress=0,
        stage="PENDING",
        input_type=input_type,
        source_url=source_url,
        input_filename=audio.filename if audio else None,
        input_path=input_path,
        input_is_isolated=1 if (inputIsIsolatedGuitar or (mode == "isolated_track")) else 0,
        start_seconds=startSeconds,
        end_seconds=endSeconds,
        output_type=output_type,
        tuning=tuning.upper(),
        capo=capo,
        quality=quality,
        hand_span=hand_span,
        prefer_low_frets=1 if preferLowFrets else 0,
        mode=mode,
        target=target,
        warnings_json=json.dumps([]),
        transcription_mode=transcription_mode,
        arrangement=arrangement,
        confidence_threshold=confidence_threshold,
        onset_window_ms=onset_window_ms,
        max_jump_semitones=max_jump,
        grid_resolution=grid_resolution,
    )
    db.add(job)
    db.commit()
    db.close()

    _queue().enqueue(process_job, job_id, job_timeout=SETTINGS.job_timeout_seconds)
    return CreateJobResponse(jobId=job_id)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        db.close()
        return JobStatusResponse(
            status="FAILED",
            progress=0,
            stage="FAILED",
            errorMessage="Job introuvable",
            warnings=[],
        )
    warnings = json.loads(job.warnings_json or "[]")
    response = JobStatusResponse(
        status=job.status,
        progress=job.progress,
        stage=job.stage,
        errorMessage=job.error_message,
        confidence=job.confidence,
        createdAt=job.created_at.isoformat() if job.created_at else None,
        warnings=warnings,
    )
    db.close()
    return response


@app.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        db.close()
        return JobResultResponse(
            tabTxtUrl=None,
            tabJsonUrl=None,
            musicXmlUrl=None,
            pdfUrl=None,
            midiUrl=None,
            metadata=JobMetadata(),
            warnings=[],
        )
    warnings = json.loads(job.warnings_json or "[]")
    base_url = SETTINGS.public_base_url
    response = JobResultResponse(
        tabTxtUrl=f"{base_url}/files/{job_id}/tab.txt" if job.tab_txt_path else None,
        tabJsonUrl=f"{base_url}/files/{job_id}/tab.json" if job.tab_json_path else None,
        tabMusicXmlUrl=f"{base_url}/files/{job_id}/result_tab.musicxml" if job.musicxml_path else None,
        tabPdfUrl=f"{base_url}/files/{job_id}/result_tab.pdf" if job.pdf_path else None,
        scoreJsonUrl=f"{base_url}/files/{job_id}/score.json" if job.score_json_path else None,
        scoreMusicXmlUrl=f"{base_url}/files/{job_id}/result_score.musicxml" if job.score_musicxml_path else None,
        scorePdfUrl=f"{base_url}/files/{job_id}/result_score.pdf" if job.score_pdf_path else None,
        musicXmlUrl=f"{base_url}/files/{job_id}/result.musicxml" if job.musicxml_path else None,
        pdfUrl=f"{base_url}/files/{job_id}/result.pdf" if job.pdf_path else None,
        midiUrl=f"{base_url}/files/{job_id}/output.mid" if job.midi_path else None,
        metadata=JobMetadata(
            durationSeconds=job.duration_seconds,
            tuning=job.tuning,
            capo=job.capo,
            quality=job.quality,
            tempoBpm=job.tempo_bpm,
            arrangement=job.arrangement,
        ),
        warnings=warnings,
    )
    db.close()
    return response


@app.get("/jobs/{job_id}/debug")
def get_job_debug(job_id: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        db.close()
        raise HTTPException(status_code=404, detail="Job introuvable.")

    def _absolute(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        return os.path.abspath(path)

    def _size(path: Optional[str]) -> Optional[int]:
        if not path:
            return None
        try:
            return os.path.getsize(path)
        except OSError:
            return None

    output_dir = os.path.join(SETTINGS.data_dir, job_id, "output")
    debug_json_path = os.path.join(SETTINGS.data_dir, job_id, "debug.json")
    stem_guitar_path = os.path.join(output_dir, "stem_guitar.wav")
    raw_basic_pitch_path = os.path.join(output_dir, "raw_basic_pitch.json")
    clean_notes_path = os.path.join(output_dir, "clean_notes.json")
    onsets_path = os.path.join(output_dir, "onsets.json")
    debug_paths = {
        "pdf": _absolute(job.pdf_path),
        "musicxml": _absolute(job.musicxml_path),
        "tabTxt": _absolute(job.tab_txt_path),
        "tabJson": _absolute(job.tab_json_path),
        "scoreJson": _absolute(job.score_json_path),
        "scoreMusicxml": _absolute(job.score_musicxml_path),
        "scorePdf": _absolute(job.score_pdf_path),
        "logs": _absolute(job.logs_path),
        "debugJson": _absolute(debug_json_path) if os.path.exists(debug_json_path) else None,
        "stemGuitarWav": _absolute(stem_guitar_path) if os.path.exists(stem_guitar_path) else None,
        "rawBasicPitchJson": _absolute(raw_basic_pitch_path) if os.path.exists(raw_basic_pitch_path) else None,
        "cleanNotesJson": _absolute(clean_notes_path) if os.path.exists(clean_notes_path) else None,
        "onsetsJson": _absolute(onsets_path) if os.path.exists(onsets_path) else None,
    }
    debug_sizes = {
        "pdf": _size(job.pdf_path),
        "musicxml": _size(job.musicxml_path),
        "tabTxt": _size(job.tab_txt_path),
        "tabJson": _size(job.tab_json_path),
        "scoreJson": _size(job.score_json_path),
        "scoreMusicxml": _size(job.score_musicxml_path),
        "scorePdf": _size(job.score_pdf_path),
        "debugJson": _size(debug_json_path),
        "stemGuitarWav": _size(stem_guitar_path),
        "rawBasicPitchJson": _size(raw_basic_pitch_path),
        "cleanNotesJson": _size(clean_notes_path),
        "onsetsJson": _size(onsets_path),
    }
    last_musescore = parse_last_musescore_run(job.logs_path) if job.logs_path else None
    tab_json_count = None
    musicxml_count = None
    diff_report = []
    if job.tab_json_path and job.musicxml_path and os.path.exists(job.tab_json_path) and os.path.exists(
        job.musicxml_path
    ):
        try:
            tab_json = load_tab_json(job.tab_json_path)
            tab_json_count, musicxml_count, diff_report = compare_tab_json_and_musicxml(
                tab_json, job.musicxml_path
            )
        except Exception as exc:
            diff_report = [{"error": str(exc)}]
    debug_info = json.loads(job.debug_info_json or "{}")
    db.close()
    return {
        "paths": debug_paths,
        "sizes": debug_sizes,
        "lastMuseScore": last_musescore,
        "totalNotesTabJson": tab_json_count,
        "totalNotesMusicXML": musicxml_count,
        "totalNotesTabTxt": tab_json_count,
        "diffReport": diff_report,
        "midiBpmDetected": debug_info.get("midiBpmDetected"),
        "tempoUsedForQuantization": debug_info.get("tempoUsedForQuantization"),
        "tempoSource": debug_info.get("tempoSource"),
        "tempoDetected": debug_info.get("tempoDetected"),
        "divisions": debug_info.get("divisions"),
        "measureTicks": debug_info.get("measureTicks"),
        "scoreWrittenOctaveShift": debug_info.get("scoreWrittenOctaveShift"),
        "writtenOctaveShift": debug_info.get("scoreWrittenOctaveShift"),
        "avgVoicedRatio": debug_info.get("avgVoicedRatio"),
        "instabilityRatio": debug_info.get("instabilityRatio"),
        "pitchMedian": debug_info.get("pitchMedian"),
        "pitchMin": debug_info.get("pitchMin"),
        "pitchMax": debug_info.get("pitchMax"),
        "notesCount": debug_info.get("noteEventsCount"),
        "warnings": debug_info.get("warnings", []),
        "transcriptionMode": job.transcription_mode,
        "noteEventsCount": debug_info.get("noteEventsCount"),
        "scoreJsonNotesCount": debug_info.get("scoreJsonNotesCount"),
        "tabJsonNotesCount": debug_info.get("tabJsonNotesCount"),
        "scoreMusicXmlNotesCount": debug_info.get("scoreMusicXmlNotesCount"),
        "tabMusicXmlNotesCount": debug_info.get("tabMusicXmlNotesCount"),
        "playabilityScore": debug_info.get("playabilityScore"),
        "playabilityCost": debug_info.get("playabilityCost"),
        "handSpan": job.hand_span,
        "preferLowFrets": bool(job.prefer_low_frets),
        "fingeringDebugUrl": (
            f"{SETTINGS.public_base_url}/files/{job_id}/fingering_debug.json"
            if job.fingering_debug_path
            else None
        ),
        "stemUsed": debug_info.get("stemUsed"),
        "stemPreprocess": debug_info.get("stemPreprocess"),
        "basicPitchNotesCountRaw": debug_info.get("basicPitchNotesCountRaw"),
        "basicPitchNotesCountAfterFilter": debug_info.get("basicPitchNotesCountAfterFilter"),
        "basicPitchNotesCountAfterMerge": debug_info.get("basicPitchNotesCountAfterMerge"),
        "basicPitchNotesCountAfterHarmonics": debug_info.get("basicPitchNotesCountAfterHarmonics"),
        "basicPitchNotesCountAfterLead": debug_info.get("basicPitchNotesCountAfterLead"),
        "basicPitchNotesCountQuantized": debug_info.get("basicPitchNotesCountQuantized"),
        "quantizationGridTicks": debug_info.get("quantizationGridTicks"),
        "quantizationDivisions": debug_info.get("quantizationDivisions"),
        "quantizationErrorsCount": debug_info.get("quantizationErrorsCount"),
        "arrangement": debug_info.get("arrangement"),
        "confidenceThreshold": debug_info.get("confidenceThreshold"),
        "onsetWindowMs": debug_info.get("onsetWindowMs"),
        "maxJumpSemitones": debug_info.get("maxJumpSemitones"),
        "gridResolution": debug_info.get("gridResolution"),
        "avgShift": debug_info.get("avgShift"),
        "maxStretch": debug_info.get("maxStretch"),
        "unreachableCount": debug_info.get("unreachableCount"),
    }


@app.get("/files/{job_id}/{file_name}")
def download_file(job_id: str, file_name: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        db.close()
        raise HTTPException(status_code=404, detail="Job introuvable.")
    path_map = {
        "tab.txt": job.tab_txt_path,
        "tab.json": job.tab_json_path,
        "result_tab.musicxml": job.musicxml_path,
        "result_tab.pdf": job.pdf_path,
        "score.json": job.score_json_path,
        "result_score.musicxml": job.score_musicxml_path,
        "result_score.pdf": job.score_pdf_path,
        "score.musicxml": job.musicxml_path,
        "result.musicxml": job.musicxml_path,
        "result.pdf": job.pdf_path,
        "output.mid": job.midi_path,
        "fingering_debug.json": job.fingering_debug_path,
    }
    output_dir = os.path.join(SETTINGS.data_dir, job_id, "output")
    path_map.update(
        {
            "stem_guitar.wav": os.path.join(output_dir, "stem_guitar.wav"),
            "raw_basic_pitch.json": os.path.join(output_dir, "raw_basic_pitch.json"),
            "clean_notes.json": os.path.join(output_dir, "clean_notes.json"),
            "onsets.json": os.path.join(output_dir, "onsets.json"),
        }
    )
    target = path_map.get(file_name)
    db.close()
    if not target or not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Fichier indisponible.")
    media_type = "application/octet-stream"
    if file_name.endswith(".txt"):
        media_type = "text/plain"
    elif file_name.endswith(".json"):
        media_type = "application/json"
    elif file_name.endswith(".musicxml"):
        media_type = "application/xml"
    elif file_name.endswith(".mid"):
        media_type = "audio/midi"
    elif file_name.endswith(".wav"):
        media_type = "audio/wav"
    elif file_name.endswith(".pdf"):
        media_type = "application/pdf"
        safe_name = os.path.basename(file_name)
        headers = {
            "Content-Disposition": f'inline; filename="{safe_name}"',
            "Cache-Control": "no-store",
            "Accept-Ranges": "bytes",
        }
        return FileResponse(target, media_type=media_type, headers=headers)
    return FileResponse(target, media_type=media_type, filename=file_name)


@app.get("/test/render-pdf")
def test_render_pdf():
    sample_path = os.path.join(os.path.dirname(__file__), "samples", "sample_tab.musicxml")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=500, detail="MusicXML de test introuvable.")
    output_dir = os.path.join(SETTINGS.data_dir, "test")
    ensure_dir(output_dir)
    logs_path = os.path.join(output_dir, "logs.txt")
    logger = get_job_logger("test-render", logs_path)
    pdf_path = os.path.join(output_dir, "sample.pdf")
    render_musicxml_to_pdf(sample_path, pdf_path, logger)
    return FileResponse(pdf_path, media_type="application/pdf", filename="sample.pdf")


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        db.close()
        return {"status": "NOT_FOUND"}
    job_dir = os.path.join(SETTINGS.data_dir, job_id)
    db.delete(job)
    db.commit()
    db.close()
    if os.path.exists(job_dir):
        for root, dirs, files in os.walk(job_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(job_dir)
    return {"status": "DELETED"}


@app.get("/library", response_model=LibraryResponse)
def get_library(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    db = SessionLocal()
    base_query = db.query(Job).filter(Job.status.in_(["DONE", "FAILED"]))
    total = base_query.count()
    jobs = (
        base_query.order_by(Job.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    items = [
        LibraryItem(
            jobId=job.id,
            status=job.status,
            createdAt=job.created_at.isoformat() if job.created_at else None,
            outputType=job.output_type,
            confidence=job.confidence,
            title=job.input_filename if job.input_filename else None,
            sourceUrl=job.source_url,
        )
        for job in jobs
    ]
    next_cursor = str(offset + limit) if offset + limit < total else None
    db.close()
    return LibraryResponse(items=items, total=total, nextCursor=next_cursor)
