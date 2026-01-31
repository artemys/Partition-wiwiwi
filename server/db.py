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
            if "arrangement" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN arrangement TEXT DEFAULT 'lead'"))
            if "confidence_threshold" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN confidence_threshold FLOAT DEFAULT 0.35"))
            if "onset_window_ms" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN onset_window_ms INTEGER DEFAULT 60"))
            if "max_jump_semitones" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN max_jump_semitones INTEGER DEFAULT 7"))
            if "grid_resolution" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN grid_resolution TEXT DEFAULT 'auto'"))
            if "onset_detection" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN onset_detection INTEGER DEFAULT 1"))
            if "onset_align" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN onset_align INTEGER DEFAULT 1"))
            if "onset_gate" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN onset_gate INTEGER DEFAULT 0"))
            if "note_min_duration_ms" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN note_min_duration_ms INTEGER DEFAULT 60"))
            if "midi_min" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN midi_min INTEGER DEFAULT 40"))
            if "midi_max" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN midi_max INTEGER DEFAULT 88"))
            if "snap_tolerance_ms" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN snap_tolerance_ms INTEGER DEFAULT 35"))
            if "poly_max_notes" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN poly_max_notes INTEGER DEFAULT 3"))

            if "tab_max_fret" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN tab_max_fret INTEGER DEFAULT 15"))
            if "tab_max_fret_span_chord" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN tab_max_fret_span_chord INTEGER DEFAULT 5"))
            if "tab_max_position_jump" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN tab_max_position_jump INTEGER DEFAULT 4"))
            if "tab_max_notes_per_chord" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN tab_max_notes_per_chord INTEGER DEFAULT 3"))

            if "tab_measures_per_system" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN tab_measures_per_system INTEGER DEFAULT 2"))
            if "tab_wrap_columns" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN tab_wrap_columns INTEGER DEFAULT 80"))
            if "tab_token_width" not in existing:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN tab_token_width INTEGER DEFAULT 3"))