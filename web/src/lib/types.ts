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
    logs?: string | null;
  };
  sizes: {
    pdf?: number | null;
    musicxml?: number | null;
    tabTxt?: number | null;
    tabJson?: number | null;
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
};
