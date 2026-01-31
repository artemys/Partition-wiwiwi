from sqlalchemy import create_engine
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
