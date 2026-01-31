import json
import os
import re
import uuid
from typing import Optional

import redis
from fastapi import Body, FastAPI, File, UploadFile, Query, HTTPException, Request
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
from .pipeline import render_musicxml_to_pdf
from .tasks import process_job
from .utils import ensure_dir, get_job_logger

app = FastAPI(title="TabScore Backend")
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
    mode: Optional[str] = Query(None),
    target: str = Query("GUITAR_BEST_EFFORT"),
    inputIsIsolatedGuitar: bool = Query(False),
    startSeconds: Optional[int] = Query(None),
    endSeconds: Optional[int] = Query(None),
    audio: UploadFile = File(None),
    body: Optional[YoutubeRequest] = Body(default=None),
    request: Request = None,
):
    output_type = outputType.lower()
    quality = quality.lower()
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
    if startSeconds is not None and startSeconds < 0:
        raise HTTPException(status_code=400, detail="startSeconds doit être >= 0.")
    if endSeconds is not None and endSeconds < 0:
        raise HTTPException(status_code=400, detail="endSeconds doit être >= 0.")
    if startSeconds is not None and endSeconds is not None and endSeconds <= startSeconds:
        raise HTTPException(status_code=400, detail="endSeconds doit être > startSeconds.")
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
        mode=mode,
        target=target,
        warnings_json=json.dumps([]),
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
        musicXmlUrl=f"{base_url}/files/{job_id}/result.musicxml" if job.musicxml_path else None,
        pdfUrl=f"{base_url}/files/{job_id}/result.pdf" if job.pdf_path else None,
        midiUrl=f"{base_url}/files/{job_id}/output.mid" if job.midi_path else None,
        metadata=JobMetadata(
            durationSeconds=job.duration_seconds,
            tuning=job.tuning,
            capo=job.capo,
            quality=job.quality,
            tempoBpm=job.tempo_bpm,
        ),
        warnings=warnings,
    )
    db.close()
    return response


@app.get("/files/{job_id}/{file_name}")
def download_file(job_id: str, file_name: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        db.close()
        return {"error": "Fichier indisponible"}
    path_map = {
        "tab.txt": job.tab_txt_path,
        "tab.json": job.tab_json_path,
        "score.musicxml": job.musicxml_path,
        "result.musicxml": job.musicxml_path,
        "result.pdf": job.pdf_path,
        "output.mid": job.midi_path,
    }
    target = path_map.get(file_name)
    db.close()
    if not target or not os.path.exists(target):
        return {"error": "Fichier indisponible"}
    media_type = "application/octet-stream"
    if file_name.endswith(".txt"):
        media_type = "text/plain"
    elif file_name.endswith(".json"):
        media_type = "application/json"
    elif file_name.endswith(".musicxml"):
        media_type = "application/xml"
    elif file_name.endswith(".mid"):
        media_type = "audio/midi"
    elif file_name.endswith(".pdf"):
        media_type = "application/pdf"
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
