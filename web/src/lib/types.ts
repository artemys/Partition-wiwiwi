export type JobStatusResponse = {
  status: "PENDING" | "RUNNING" | "DONE" | "FAILED";
  progress: number;
  stage?: string | null;
  errorMessage?: string | null;
  confidence?: number | null;
  createdAt?: string | null;
};

export type JobResultResponse = {
  tabTxtUrl?: string | null;
  tabJsonUrl?: string | null;
  tabMusicXmlUrl?: string | null;
  tabPdfUrl?: string | null;
  scoreJsonUrl?: string | null;
  scoreMusicXmlUrl?: string | null;
  scorePdfUrl?: string | null;
  musicXmlUrl?: string | null;
  pdfUrl?: string | null;
  midiUrl?: string | null;
};

export type LibraryItem = {
  jobId: string;
  status: "DONE" | "FAILED";
  createdAt?: string | null;
  outputType?: "tab" | "score" | "both" | null;
  confidence?: number | null;
  title?: string | null;
  sourceUrl?: string | null;
};

export type LibraryResponse = {
  items: LibraryItem[];
  nextCursor?: string | null;
  total?: number | null;
};

export type JobDebugResponse = {
  paths: {
    pdf?: string | null;
    musicxml?: string | null;
    tabTxt?: string | null;
    tabJson?: string | null;
    scoreJson?: string | null;
    scoreMusicxml?: string | null;
    scorePdf?: string | null;
    logs?: string | null;
    debugJson?: string | null;
    stemGuitarWav?: string | null;
    rawBasicPitchJson?: string | null;
    cleanNotesJson?: string | null;
    onsetsJson?: string | null;
  };
  sizes: {
    pdf?: number | null;
    musicxml?: number | null;
    tabTxt?: number | null;
    tabJson?: number | null;
    scoreJson?: number | null;
    scoreMusicxml?: number | null;
    scorePdf?: number | null;
    debugJson?: number | null;
    stemGuitarWav?: number | null;
    rawBasicPitchJson?: number | null;
    cleanNotesJson?: number | null;
    onsetsJson?: number | null;
  };
  lastMuseScore?: {
    command: string;
    stdout?: string | null;
    stderr?: string | null;
  } | null;
  totalNotesTabJson?: number | null;
  totalNotesMusicXML?: number | null;
  totalNotesTabTxt?: number | null;
  diffReport?: Array<{
    measure?: number | null;
    string?: number | null;
    fret?: number | null;
    countTabJson?: number | null;
    countMusicXML?: number | null;
  }>;
  midiBpmDetected?: number | null;
  tempoUsedForQuantization?: number | null;
  tempoSource?: string | null;
  divisions?: number | null;
  measureTicks?: number | null;
  scoreWrittenOctaveShift?: number | null;
  noteEventsCount?: number | null;
  scoreJsonNotesCount?: number | null;
  tabJsonNotesCount?: number | null;
  scoreMusicXmlNotesCount?: number | null;
  tabMusicXmlNotesCount?: number | null;
  playabilityScore?: number | null;
  playabilityCost?: number | null;
  handSpan?: number | null;
  preferLowFrets?: boolean | null;
  fingeringDebugUrl?: string | null;
  transcriptionMode?: "best_free" | "monophonic_tuner" | "polyphonic_basic_pitch" | null;
  tempoDetected?: number | null;
  avgVoicedRatio?: number | null;
  instabilityRatio?: number | null;
  pitchMedian?: number | null;
  pitchMin?: number | null;
  pitchMax?: number | null;
  notesCount?: number | null;
  warnings?: string[] | null;
  writtenOctaveShift?: number | null;
  stemUsed?: string | null;
  stemPreprocess?: Record<string, unknown> | null;
  basicPitchNotesCountRaw?: number | null;
  basicPitchNotesCountAfterFilter?: number | null;
  basicPitchNotesCountAfterMerge?: number | null;
  basicPitchNotesCountAfterHarmonics?: number | null;
  basicPitchNotesCountAfterLead?: number | null;
  basicPitchNotesCountQuantized?: number | null;
  quantizationGridTicks?: number | null;
  quantizationDivisions?: number | null;
  quantizationErrorsCount?: number | null;
  arrangement?: "lead" | "poly" | null;
  confidenceThreshold?: number | null;
  onsetWindowMs?: number | null;
  maxJumpSemitones?: number | null;
  gridResolution?: "auto" | "eighth" | "sixteenth" | null;
  avgShift?: number | null;
  maxStretch?: number | null;
  unreachableCount?: number | null;
};
