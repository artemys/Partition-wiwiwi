import type { LibraryItem } from "./types";

export function formatDateTime(value?: string | null) {
  if (!value) {
    return "Date inconnue";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("fr-FR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function formatJobTitle(item: LibraryItem) {
  if (item.title) {
    return item.title;
  }
  if (item.sourceUrl) {
    return "Lien YouTube";
  }
  return `Transcription ${item.jobId.slice(0, 6)}`;
}
