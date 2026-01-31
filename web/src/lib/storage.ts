const API_URL_OVERRIDE_KEY = "tabscore_api_url_override";
const EXPORT_FORMAT_KEY = "tabscore_export_format";

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
