from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

from .config import SETTINGS


engine = create_engine(
    SETTINGS.db_url,
    connect_args={"check_same_thread": False} if SETTINGS.db_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    if SETTINGS.db_url.startswith("sqlite"):
        with engine.connect() as conn:
            cols = conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
            existing = {row[1] for row in cols}
            if "start_seconds" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN start_seconds INTEGER"))
            if "end_seconds" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN end_seconds INTEGER"))
            if "pdf_path" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN pdf_path TEXT"))
            if "score_json_path" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN score_json_path TEXT"))
            if "score_musicxml_path" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN score_musicxml_path TEXT"))
            if "score_pdf_path" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN score_pdf_path TEXT"))
            if "debug_info_json" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN debug_info_json TEXT"))
            if "hand_span" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN hand_span INTEGER DEFAULT 4"))
            if "prefer_low_frets" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN prefer_low_frets INTEGER DEFAULT 0"))
            if "fingering_debug_path" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN fingering_debug_path TEXT"))
            if "transcription_mode" not in existing:
                conn.execute(
                    text(
                        "ALTER TABLE jobs ADD COLUMN transcription_mode TEXT DEFAULT 'polyphonic_basic_pitch'"
                    )
                )