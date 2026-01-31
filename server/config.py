import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    data_dir: str = os.getenv("TABSERVER_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
    db_url: str = os.getenv("TABSERVER_DB_URL", "")
    redis_url: str = os.getenv("TABSERVER_REDIS_URL", "redis://localhost:6379/0")
    public_base_url: str = os.getenv("TABSERVER_PUBLIC_URL", "http://localhost:8000")
    max_upload_mb: int = int(os.getenv("TABSERVER_MAX_UPLOAD_MB", "50"))
    max_duration_seconds: int = int(os.getenv("TABSERVER_MAX_DURATION_SECONDS", str(12 * 60)))
    sample_rate: int = int(os.getenv("TABSERVER_SAMPLE_RATE", "22050"))
    job_timeout_seconds: int = int(os.getenv("TABSERVER_JOB_TIMEOUT_SECONDS", str(30 * 60)))
    subprocess_timeout_seconds: int = int(os.getenv("TABSERVER_SUBPROCESS_TIMEOUT_SECONDS", str(20 * 60)))
    ffmpeg_path: str = os.getenv("TABSERVER_FFMPEG", "ffmpeg")
    ffprobe_path: str = os.getenv("TABSERVER_FFPROBE", "ffprobe")
    yt_dlp_path: str = os.getenv("TABSERVER_YTDLP", "yt-dlp")
    demucs_path: str = os.getenv("TABSERVER_DEMUCS", "demucs")
    basic_pitch_path: str = os.getenv("TABSERVER_BASIC_PITCH", "basic-pitch")
    default_tempo_bpm: int = int(os.getenv("TABSERVER_DEFAULT_TEMPO_BPM", "120"))


_raw = Settings()
_db_url = _raw.db_url or f"sqlite:///{os.path.join(_raw.data_dir, 'tabscore.db')}"
SETTINGS = Settings(
    data_dir=_raw.data_dir,
    db_url=_db_url,
    redis_url=_raw.redis_url,
    public_base_url=_raw.public_base_url,
    max_upload_mb=_raw.max_upload_mb,
    max_duration_seconds=_raw.max_duration_seconds,
    sample_rate=_raw.sample_rate,
    job_timeout_seconds=_raw.job_timeout_seconds,
    subprocess_timeout_seconds=_raw.subprocess_timeout_seconds,
    ffmpeg_path=_raw.ffmpeg_path,
    ffprobe_path=_raw.ffprobe_path,
    yt_dlp_path=_raw.yt_dlp_path,
    demucs_path=_raw.demucs_path,
    basic_pitch_path=_raw.basic_pitch_path,
    default_tempo_bpm=_raw.default_tempo_bpm,
)
