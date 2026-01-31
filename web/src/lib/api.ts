import type { JobResultResponse, JobStatusResponse, LibraryResponse, JobDebugResponse } from "./types";
import { getApiUrlOverride } from "./storage";

const DEFAULT_API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function normalizeBaseUrl(baseUrl: string) {
  return baseUrl.replace(/\/+$/, "");
}

export function getApiBaseUrl() {
  const override = getApiUrlOverride();
  if (override && override.trim()) {
    return normalizeBaseUrl(override.trim());
  }
  return normalizeBaseUrl(DEFAULT_API_URL);
}

function buildUrl(path: string) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${getApiBaseUrl()}${normalizedPath}`;
}

async function fetchJson<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    let message = `Erreur HTTP ${response.status}`;
    try {
      const data = await response.json();
      if (data?.detail) {
        message = data.detail;
      } else if (data?.error) {
        message = data.error;
      }
    } catch {
      // ignore json parse
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function createJob(params: {
  outputType: "tab" | "score" | "both";
  tuning: string;
  capo: number;
  quality: "fast" | "accurate";
  transcriptionMode?: "best_free" | "monophonic_tuner" | "polyphonic_basic_pitch";
  arrangement?: "lead" | "poly";
  confidenceThreshold?: number;
  onsetWindowMs?: number;
  maxJumpSemitones?: number;
  gridResolution?: "auto" | "eighth" | "sixteenth";
  onsetDetection?: boolean;
  onsetAlign?: boolean;
  onsetGate?: boolean;
  minNoteDurationMs?: number;
  midiMin?: number;
  midiMax?: number;
  snapToleranceMs?: number;
  polyMaxNotes?: number;
  tabMaxFret?: number;
  tabMaxFretSpanChord?: number;
  tabMaxPositionJump?: number;
  tabMaxNotesPerChord?: number;
  tabMeasuresPerSystem?: number;
  tabWrapColumns?: number;
  tabTokenWidth?: number;
  audioFile?: File | null;
  youtubeUrl?: string | null;
  startSeconds?: number;
  endSeconds?: number;
  handSpan?: number;
  preferLowFrets?: boolean;
}) {
  const query = new URLSearchParams({
    outputType: params.outputType,
    tuning: params.tuning,
    capo: String(params.capo),
    quality: params.quality,
  });
  const transcriptionMode = params.transcriptionMode ?? "best_free";
  query.set("transcriptionMode", transcriptionMode);
  if (params.arrangement) {
    query.set("arrangement", params.arrangement);
  }
  if (Number.isFinite(params.confidenceThreshold ?? NaN)) {
    query.set("confidenceThreshold", String(params.confidenceThreshold));
  }
  if (Number.isFinite(params.onsetWindowMs ?? NaN)) {
    query.set("onsetWindowMs", String(params.onsetWindowMs));
  }
  if (Number.isFinite(params.maxJumpSemitones ?? NaN)) {
    query.set("maxJumpSemitones", String(params.maxJumpSemitones));
  }
  if (params.gridResolution) {
    query.set("gridResolution", params.gridResolution);
  }
  if (params.onsetDetection === false) {
    query.set("onsetDetection", "false");
  }
  if (params.onsetAlign === false) {
    query.set("onsetAlign", "false");
  }
  if (params.onsetGate) {
    query.set("onsetGate", "true");
  }
  if (Number.isFinite(params.minNoteDurationMs ?? NaN)) {
    query.set("minNoteDurationMs", String(params.minNoteDurationMs));
  }
  if (Number.isFinite(params.midiMin ?? NaN)) {
    query.set("midiMin", String(params.midiMin));
  }
  if (Number.isFinite(params.midiMax ?? NaN)) {
    query.set("midiMax", String(params.midiMax));
  }
  if (Number.isFinite(params.snapToleranceMs ?? NaN)) {
    query.set("snapToleranceMs", String(params.snapToleranceMs));
  }
  if (Number.isFinite(params.polyMaxNotes ?? NaN)) {
    query.set("polyMaxNotes", String(params.polyMaxNotes));
  }
  if (Number.isFinite(params.tabMaxFret ?? NaN)) {
    query.set("tabMaxFret", String(params.tabMaxFret));
  }
  if (Number.isFinite(params.tabMaxFretSpanChord ?? NaN)) {
    query.set("tabMaxFretSpanChord", String(params.tabMaxFretSpanChord));
  }
  if (Number.isFinite(params.tabMaxPositionJump ?? NaN)) {
    query.set("tabMaxPositionJump", String(params.tabMaxPositionJump));
  }
  if (Number.isFinite(params.tabMaxNotesPerChord ?? NaN)) {
    query.set("tabMaxNotesPerChord", String(params.tabMaxNotesPerChord));
  }
  if (Number.isFinite(params.tabMeasuresPerSystem ?? NaN)) {
    query.set("tabMeasuresPerSystem", String(params.tabMeasuresPerSystem));
  }
  if (Number.isFinite(params.tabWrapColumns ?? NaN)) {
    query.set("tabWrapColumns", String(params.tabWrapColumns));
  }
  if (Number.isFinite(params.tabTokenWidth ?? NaN)) {
    query.set("tabTokenWidth", String(params.tabTokenWidth));
  }
  if (Number.isFinite(params.handSpan ?? NaN)) {
    query.set("handSpan", String(params.handSpan));
  }
  if (params.preferLowFrets) {
    query.set("preferLowFrets", "true");
  }
  if (Number.isFinite(params.startSeconds)) {
    query.set("startSeconds", String(params.startSeconds));
  }
  if (Number.isFinite(params.endSeconds)) {
    query.set("endSeconds", String(params.endSeconds));
  }

  if (params.audioFile) {
    const formData = new FormData();
    formData.append("audio", params.audioFile);
    return fetchJson<{ jobId: string }>(buildUrl(`/jobs?${query.toString()}`), {
      method: "POST",
      body: formData,
    });
  }

  return fetchJson<{ jobId: string }>(buildUrl(`/jobs?${query.toString()}`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ youtubeUrl: params.youtubeUrl }),
  });
}

export function getJobStatus(jobId: string) {
  return fetchJson<JobStatusResponse>(buildUrl(`/jobs/${jobId}`));
}

export function getJobResult(jobId: string) {
  return fetchJson<JobResultResponse>(buildUrl(`/jobs/${jobId}/result`));
}

export function getLibrary() {
  return fetchJson<LibraryResponse>(buildUrl("/library"));
}

export function deleteJob(jobId: string) {
  return fetchJson<{ status: string }>(buildUrl(`/jobs/${jobId}`), { method: "DELETE" });
}

export function getJobDebug(jobId: string) {
  return fetchJson<JobDebugResponse>(buildUrl(`/jobs/${jobId}/debug`));
}
