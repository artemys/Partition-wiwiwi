import json
import os
import re
import uuid
from typing import Optional

import redis
from fastapi import Body, FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import FileResponse

from .config import SETTINGS
from .db import SessionLocal, init_db
from .models import Job
from .schemas import CreateJobResponse, JobMetadata, JobResultResponse, JobStatusResponse, YoutubeRequest
from .tasks import process_job
from .utils import ensure_dir

app = FastAPI(title="TabScore Backend")

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
    if extension not in (".mp3", ".wav", ".m4a"):
        raise HTTPException(status_code=400, detail="Format audio non supporté (mp3, wav, m4a uniquement).")
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
    audio: UploadFile = File(None),
    body: Optional[YoutubeRequest] = Body(default=None),
):
    output_type = outputType.lower()
    quality = quality.lower()
    if output_type not in ("tab", "score", "both"):
        raise HTTPException(status_code=400, detail="outputType doit être tab, score ou both.")
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
    return CreateJobResponse(jobId=job_id, status="PENDING")


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
            midiUrl=None,
            metadata=JobMetadata(),
            warnings=[],
        )
    warnings = json.loads(job.warnings_json or "[]")
    base_url = SETTINGS.public_base_url
    response = JobResultResponse(
        tabTxtUrl=f"{base_url}/files/{job_id}/tab.txt" if job.tab_txt_path else None,
        tabJsonUrl=f"{base_url}/files/{job_id}/tab.json" if job.tab_json_path else None,
        musicXmlUrl=f"{base_url}/files/{job_id}/score.musicxml" if job.musicxml_path else None,
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
    return FileResponse(target, media_type=media_type, filename=file_name)


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
