from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func

from .db import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    status = Column(String, nullable=False, default="PENDING")
    progress = Column(Integer, nullable=False, default=0)
    stage = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    warnings_json = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    input_type = Column(String, nullable=False)  # upload|youtube
    source_url = Column(Text, nullable=True)
    input_filename = Column(Text, nullable=True)
    input_path = Column(Text, nullable=True)
    input_is_isolated = Column(Integer, nullable=False, default=0)
    start_seconds = Column(Integer, nullable=True)
    end_seconds = Column(Integer, nullable=True)

    output_type = Column(String, nullable=False, default="both")
    tuning = Column(String, nullable=False, default="EADGBE")
    capo = Column(Integer, nullable=False, default=0)
    quality = Column(String, nullable=False, default="fast")
    mode = Column(String, nullable=True)
    target = Column(String, nullable=True)

    duration_seconds = Column(Float, nullable=True)
    tempo_bpm = Column(Float, nullable=True)

    tab_txt_path = Column(Text, nullable=True)
    tab_json_path = Column(Text, nullable=True)
    score_json_path = Column(Text, nullable=True)
    musicxml_path = Column(Text, nullable=True)
    pdf_path = Column(Text, nullable=True)
    score_musicxml_path = Column(Text, nullable=True)
    score_pdf_path = Column(Text, nullable=True)
    midi_path = Column(Text, nullable=True)
    logs_path = Column(Text, nullable=True)
