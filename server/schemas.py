from typing import List, Optional

from pydantic import BaseModel, Field


class CreateJobResponse(BaseModel):
    jobId: str


class JobStatusResponse(BaseModel):
    status: str
    progress: int
    stage: Optional[str] = None
    errorMessage: Optional[str] = None
    confidence: Optional[float] = None
    createdAt: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class JobMetadata(BaseModel):
    durationSeconds: Optional[float] = None
    tuning: Optional[str] = None
    capo: Optional[int] = None
    quality: Optional[str] = None
    tempoBpm: Optional[float] = None
    arrangement: Optional[str] = None


class JobResultResponse(BaseModel):
    tabTxtUrl: Optional[str] = None
    tabJsonUrl: Optional[str] = None
    tabMusicXmlUrl: Optional[str] = None
    tabPdfUrl: Optional[str] = None
    scoreJsonUrl: Optional[str] = None
    scoreMusicXmlUrl: Optional[str] = None
    scorePdfUrl: Optional[str] = None
    musicXmlUrl: Optional[str] = None
    pdfUrl: Optional[str] = None
    midiUrl: Optional[str] = None
    metadata: JobMetadata
    warnings: List[str] = Field(default_factory=list)


class YoutubeRequest(BaseModel):
    youtubeUrl: str


class LibraryItem(BaseModel):
    jobId: str
    status: str
    createdAt: Optional[str] = None
    outputType: Optional[str] = None
    confidence: Optional[float] = None
    title: Optional[str] = None
    sourceUrl: Optional[str] = None


class LibraryResponse(BaseModel):
    items: List[LibraryItem]
    total: Optional[int] = None
    nextCursor: Optional[str] = None

