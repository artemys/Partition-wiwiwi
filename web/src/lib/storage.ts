const API_URL_OVERRIDE_KEY = "tabscore_api_url_override";
const EXPORT_FORMAT_KEY = "tabscore_export_format";
const PLAYABILITY_SPAN_KEY = "tabscore_playability_span";
const PREFER_LOW_FRETS_KEY = "tabscore_prefer_low_frets";

export type ExportFormat = "pdf" | "musicxml" | "tab" | "midi";

export function getApiUrlOverride(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return localStorage.getItem(API_URL_OVERRIDE_KEY);
}

export function setApiUrlOverride(value: string) {
  if (typeof window === "undefined") {
    return;
  }
  if (!value) {
    localStorage.removeItem(API_URL_OVERRIDE_KEY);
    return;
  }
  localStorage.setItem(API_URL_OVERRIDE_KEY, value);
}

export function getPreferredExportFormat(): ExportFormat | null {
  if (typeof window === "undefined") {
    return null;
  }
  const value = localStorage.getItem(EXPORT_FORMAT_KEY);
  if (!value) {
    return null;
  }
  if (value === "pdf" || value === "musicxml" || value === "tab" || value === "midi") {
    return value;
  }
  return null;
}

export function setPreferredExportFormat(value: ExportFormat) {
  if (typeof window === "undefined") {
    return;
  }
  localStorage.setItem(EXPORT_FORMAT_KEY, value);
}

export type PlayabilitySpan = 4 | 5 | 6;

export function getPlayabilitySpan(): PlayabilitySpan {
  if (typeof window === "undefined") {
    return 4;
  }
  const value = localStorage.getItem(PLAYABILITY_SPAN_KEY);
  if (value === "5") {
    return 5;
  }
  if (value === "6") {
    return 6;
  }
  return 4;
}

export function setPlayabilitySpan(value: PlayabilitySpan) {
  if (typeof window === "undefined") {
    return;
  }
  localStorage.setItem(PLAYABILITY_SPAN_KEY, String(value));
}

export function getPreferLowFrets(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  return localStorage.getItem(PREFER_LOW_FRETS_KEY) === "1";
}

export function setPreferLowFrets(value: boolean) {
  if (typeof window === "undefined") {
    return;
  }
  if (value) {
    localStorage.setItem(PREFER_LOW_FRETS_KEY, "1");
  } else {
    localStorage.removeItem(PREFER_LOW_FRETS_KEY);
  }
}
