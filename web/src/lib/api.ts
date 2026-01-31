import type { JobResultResponse, JobStatusResponse, LibraryResponse } from "./types";
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
  audioFile?: File | null;
  youtubeUrl?: string | null;
  startSeconds?: number;
  endSeconds?: number;
}) {
  const query = new URLSearchParams({
    outputType: params.outputType,
    tuning: params.tuning,
    capo: String(params.capo),
    quality: params.quality,
  });
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
