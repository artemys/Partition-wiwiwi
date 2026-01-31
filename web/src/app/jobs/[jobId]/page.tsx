"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { deleteJob, getJobDebug, getJobResult, getJobStatus } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import { getPreferredExportFormat } from "@/lib/storage";

type PdfFetchInfo = {
  status: number;
  contentType: string | null;
  contentLength: number | null;
};

type PdfPreviewState = {
  objectUrl: string | null;
  fetchInfo: PdfFetchInfo | null;
  error: string | null;
};

function usePdfPreview(url?: string | null): PdfPreviewState {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [fetchInfo, setFetchInfo] = useState<PdfFetchInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!url) {
      setObjectUrl(null);
      setFetchInfo(null);
      setError(null);
      return;
    }
    let objectUrlRef: string | null = null;
    const controller = new AbortController();
    setObjectUrl(null);
    setFetchInfo(null);
    setError(null);

    const loadPdf = async () => {
      try {
        const response = await fetch(url, { signal: controller.signal });
        const contentType = response.headers.get("content-type");
        const contentLengthHeader = response.headers.get("content-length");
        const parsedLength = contentLengthHeader ? Number(contentLengthHeader) : null;
        const info: PdfFetchInfo = {
          status: response.status,
          contentType,
          contentLength:
            parsedLength !== null && !Number.isNaN(parsedLength) ? parsedLength : null,
        };
        if (controller.signal.aborted) {
          return;
        }
        setFetchInfo(info);
        if (!response.ok) {
          throw new Error(`Réponse HTTP ${response.status}`);
        }
        if ((contentType ?? "").split(";")[0] !== "application/pdf") {
          throw new Error("Ressource reçue n'est pas un PDF.");
        }
        const blob = await response.blob();
        if (controller.signal.aborted) {
          return;
        }
        objectUrlRef = URL.createObjectURL(blob);
        setObjectUrl(objectUrlRef);
        setError(null);
      } catch (err) {
        if (controller.signal.aborted) {
          return;
        }
        const message = err instanceof Error && err.message ? err.message : "Impossible de charger le PDF.";
        setError(message);
        setObjectUrl(null);
      }
    };

    void loadPdf();

    return () => {
      controller.abort();
      if (objectUrlRef) {
        URL.revokeObjectURL(objectUrlRef);
      }
    };
  }, [url]);

  return { objectUrl, fetchInfo, error };
}

type PdfPreviewCardProps = {
  title: string;
  url?: string | null;
  preview: PdfPreviewState;
  onOpen: () => void;
};

function PdfPreviewCard({ title, url, preview, onOpen }: PdfPreviewCardProps) {
  return (
    <Card className="space-y-3">
      <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">{title}</h2>
      {url ? (
        preview.objectUrl ? (
          <div className="relative overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800">
            <embed
              src={preview.objectUrl}
              type="application/pdf"
              className="h-[520px] w-full"
              title={title}
            />
            <div className="flex justify-end border-t border-zinc-200 px-3 py-2 text-xs dark:border-zinc-800">
              <Button variant="ghost" onClick={onOpen}>
                Ouvrir dans un nouvel onglet
              </Button>
            </div>
          </div>
        ) : preview.error ? (
          <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-zinc-200 p-6 text-sm text-zinc-600 dark:border-zinc-800 dark:text-zinc-400">
            <p>{preview.error}</p>
            <Button variant="ghost" onClick={onOpen}>
              Ouvrir dans un nouvel onglet
            </Button>
          </div>
        ) : (
          <p className="p-6 text-sm text-zinc-500 dark:text-zinc-400">Chargement du PDF…</p>
        )
      ) : (
        <p className="text-sm text-zinc-500 dark:text-zinc-400">Aucun PDF disponible.</p>
      )}
    </Card>
  );
}

export default function JobDetailsPage() {
  const params = useParams();
  const router = useRouter();
  const jobId = Array.isArray(params.jobId) ? params.jobId[0] : params.jobId;
  const resolvedJobId = typeof jobId === "string" ? jobId : "";
  const [tabPreview, setTabPreview] = useState<string | null>(null);

  const {
    data: status,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["job", resolvedJobId],
    queryFn: () => getJobStatus(resolvedJobId),
    enabled: Boolean(resolvedJobId),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 2000;
      return data.status === "DONE" || data.status === "FAILED" ? false : 2000;
    },
  });

  const { data: result } = useQuery({
    queryKey: ["job-result", resolvedJobId],
    queryFn: () => getJobResult(resolvedJobId),
    enabled: status?.status === "DONE" && Boolean(resolvedJobId),
  });
  const tabPdfPreview = usePdfPreview(result?.tabPdfUrl ?? null);
  const scorePdfPreview = usePdfPreview(result?.scorePdfUrl ?? null);

  const {
    data: debugInfo,
    isLoading: isDebugLoading,
    isError: isDebugError,
  } = useQuery({
    queryKey: ["job-debug", resolvedJobId],
    queryFn: () => getJobDebug(resolvedJobId),
    enabled: status?.status === "DONE" && Boolean(resolvedJobId),
    retry: false,
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteJob(resolvedJobId),
    onSuccess: () => {
      toast.success("Transcription supprimée.");
      router.push("/library");
    },
    onError: (err) => {
      toast.error(err instanceof Error ? err.message : "Suppression impossible.");
    },
  });

  const downloadUrl = useMemo(() => {
    if (!result) return null;
    const preference = getPreferredExportFormat();
    if (preference === "pdf") {
      return result.scorePdfUrl ?? result.tabPdfUrl ?? result.pdfUrl ?? null;
    }
    if (preference === "musicxml") {
      return (
        result.scoreMusicXmlUrl ??
        result.tabMusicXmlUrl ??
        result.musicXmlUrl ??
        null
      );
    }
    if (preference === "tab") {
      return result.tabTxtUrl ?? result.tabJsonUrl ?? null;
    }
    if (preference === "midi") {
      return result.midiUrl ?? null;
    }
    return (
      result.scorePdfUrl ??
      result.tabPdfUrl ??
      result.pdfUrl ??
      result.scoreMusicXmlUrl ??
      result.tabMusicXmlUrl ??
      result.musicXmlUrl ??
      result.tabTxtUrl ??
      result.tabJsonUrl ??
      result.midiUrl ??
      null
    );
  }, [result]);

  const openPdf = (url?: string | null, info?: PdfFetchInfo | null) => {
    if (!url) {
      return;
    }
    if (info?.status === 404) {
      toast.error("PDF introuvable (404).");
    } else if (info && (info.contentType ?? "").split(";")[0] !== "application/pdf") {
      toast.error("La ressource n'est pas un PDF.");
    }
    window.open(url, "_blank", "noopener,noreferrer");
  };

  const formatBytes = (value?: number | null) =>
    value != null ? `${value.toLocaleString("fr-FR")} octets` : "—";
  const formatCount = (value?: number | null) => (value != null ? value.toString() : "—");
  const formatDecimal = (value?: number | null) =>
    value != null ? value.toLocaleString("fr-FR", { maximumFractionDigits: 2 }) : "—";
  const formatSigned = (value?: number | null) =>
    value != null ? `${value >= 0 ? "+" : ""}${value}` : "—";

  const handleCopyDebug = async () => {
    if (!debugInfo) {
      return;
    }
    if (!navigator?.clipboard) {
      toast.error("Copie impossible (clipboard absent).");
      return;
    }
    try {
      await navigator.clipboard.writeText(JSON.stringify(debugInfo, null, 2));
      toast.success("Infos debug copiées.");
    } catch {
      toast.error("Impossible de copier les infos debug.");
    }
  };

  useEffect(() => {
    const tabUrl = result?.tabTxtUrl;
    if (!tabUrl) {
      setTabPreview(null);
      return;
    }
    let cancelled = false;
    fetch(tabUrl)
      .then((res) => res.text())
      .then((text) => {
        if (!cancelled) {
          setTabPreview(text);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setTabPreview("Impossible de charger la tablature.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [result?.tabTxtUrl]);

  if (!resolvedJobId) {
    return <Card>Job introuvable.</Card>;
  }

  if (isLoading) {
    return <Card>Chargement du job...</Card>;
  }

  if (error || !status) {
    return <Card>Impossible de charger le job.</Card>;
  }

  const statusBadge =
    status.status === "DONE"
      ? "success"
      : status.status === "FAILED"
        ? "danger"
        : "warning";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-100">Détails du job</h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">ID: {resolvedJobId}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={() => router.push("/library")}>
            Retour bibliothèque
          </Button>
          <Button variant="danger" onClick={() => deleteMutation.mutate()} disabled={deleteMutation.isPending}>
            Supprimer
          </Button>
        </div>
      </div>

      <Card className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <Badge variant={statusBadge as "success" | "danger" | "warning"}>
            {status.status === "DONE"
              ? "Terminé"
              : status.status === "FAILED"
                ? "Échec"
                : status.status === "RUNNING"
                  ? "En cours"
                  : "En attente"}
          </Badge>
          {status.confidence != null && (
            <Badge variant="neutral">Confiance {Math.round(status.confidence * 100)}%</Badge>
          )}
          {status.createdAt && (
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              {formatDateTime(status.createdAt)}
            </span>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm text-zinc-600 dark:text-zinc-300">
            <span>Étape : {status.stage ?? "—"}</span>
            <span>{status.progress}%</span>
          </div>
          <Progress value={status.progress} />
        </div>

        {status.status === "FAILED" && status.errorMessage && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
            {status.errorMessage}
          </div>
        )}

        {status.status === "DONE" && (
          <div className="flex flex-wrap gap-2">
            {downloadUrl && (
              <Button onClick={() => window.open(downloadUrl, "_blank", "noopener,noreferrer")}>Télécharger</Button>
            )}
            {result?.tabTxtUrl && (
              <Button
                variant="secondary"
                onClick={() =>
                  window.open(result.tabTxtUrl ?? undefined, "_blank", "noopener,noreferrer")
                }
              >
                Ouvrir TAB
              </Button>
            )}
            {result?.tabPdfUrl && (
              <Button
                variant="secondary"
                onClick={() => openPdf(result.tabPdfUrl, tabPdfPreview.fetchInfo)}
              >
                Ouvrir PDF TAB
              </Button>
            )}
            {result?.scorePdfUrl && (
              <Button
                variant="secondary"
                onClick={() => openPdf(result.scorePdfUrl, scorePdfPreview.fetchInfo)}
              >
                Ouvrir PDF Partition
              </Button>
            )}
          </div>
        )}
      </Card>

      {status.status === "DONE" && (
        <>
          <div className="space-y-4">
            <Card className="space-y-3">
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Prévisualisation TAB</h2>
              {result?.tabTxtUrl ? (
                <pre className="max-h-[520px] overflow-auto rounded-xl bg-zinc-900 p-4 text-xs text-zinc-100">
                  {tabPreview ?? "Chargement du TAB..."}
                </pre>
              ) : (
                <p className="text-sm text-zinc-500 dark:text-zinc-400">Aucune tablature disponible.</p>
              )}
            </Card>
            <div className="grid gap-4 lg:grid-cols-2">
              <PdfPreviewCard
                title="PDF TAB"
                url={result?.tabPdfUrl ?? null}
                preview={tabPdfPreview}
                onOpen={() => openPdf(result?.tabPdfUrl, tabPdfPreview.fetchInfo)}
              />
              <PdfPreviewCard
                title="PDF Partition"
                url={result?.scorePdfUrl ?? null}
                preview={scorePdfPreview}
                onOpen={() => openPdf(result?.scorePdfUrl, scorePdfPreview.fetchInfo)}
              />
            </div>
          </div>
          <Card className="space-y-4">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Debug (dev)</h2>
              <div className="flex gap-2">
                <Button variant="ghost" disabled={!debugInfo} onClick={handleCopyDebug}>
                  Copier debug
                </Button>
                {debugInfo?.fingeringDebugUrl && (
                  <Button
                    variant="ghost"
                    onClick={() =>
                      window.open(
                        debugInfo.fingeringDebugUrl ?? undefined,
                        "_blank",
                        "noopener,noreferrer"
                      )
                    }
                  >
                    Export fingering
                  </Button>
                )}
              </div>
            </div>
            <div className="space-y-3 text-sm text-zinc-600 dark:text-zinc-300">
              <div className="space-y-2">
                <p className="text-xs uppercase text-zinc-500 dark:text-zinc-400">PDF TAB</p>
                <div className="flex justify-between">
                  <span>URL</span>
                  <span className="font-mono text-xs text-zinc-700 dark:text-zinc-200 break-all">
                    {result?.tabPdfUrl ?? "—"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Status HTTP</span>
                  <span>{tabPdfPreview.fetchInfo?.status ?? "—"}</span>
                </div>
                <div className="flex justify-between">
                  <span>Content-Type</span>
                  <span>{tabPdfPreview.fetchInfo?.contentType ?? "—"}</span>
                </div>
                <div className="flex justify-between">
                  <span>Content-Length</span>
                  <span>{formatBytes(tabPdfPreview.fetchInfo?.contentLength)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Erreur</span>
                  <span>{tabPdfPreview.error ?? "—"}</span>
                </div>
              </div>
              <div className="space-y-2">
                <p className="text-xs uppercase text-zinc-500 dark:text-zinc-400">PDF PARTITION</p>
                <div className="flex justify-between">
                  <span>URL</span>
                  <span className="font-mono text-xs text-zinc-700 dark:text-zinc-200 break-all">
                    {result?.scorePdfUrl ?? "—"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Status HTTP</span>
                  <span>{scorePdfPreview.fetchInfo?.status ?? "—"}</span>
                </div>
                <div className="flex justify-between">
                  <span>Content-Type</span>
                  <span>{scorePdfPreview.fetchInfo?.contentType ?? "—"}</span>
                </div>
                <div className="flex justify-between">
                  <span>Content-Length</span>
                  <span>{formatBytes(scorePdfPreview.fetchInfo?.contentLength)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Erreur</span>
                  <span>{scorePdfPreview.error ?? "—"}</span>
                </div>
              </div>
              {isDebugLoading && <p>Chargement des informations de debug…</p>}
              {isDebugError && <p>Impossible de charger les infos de debug.</p>}
              {debugInfo && (
                <div className="space-y-4 rounded-xl border border-zinc-200 bg-white/60 p-3 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950/60 dark:text-zinc-200">
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-1">
                      <p className="text-[10px] uppercase text-zinc-500 dark:text-zinc-400">Tempo</p>
                      <p>
                        BPM détecté: <span className="font-mono">{formatDecimal(debugInfo.midiBpmDetected)}</span>
                      </p>
                      <p>
                        Tempo utilisé:{" "}
                        <span className="font-mono">{formatDecimal(debugInfo.tempoUsedForQuantization)}</span>
                      </p>
                      <p>
                        Source: <span className="font-mono">{debugInfo.tempoSource ?? "—"}</span>
                      </p>
                      <p>
                        Divisions: <span className="font-mono">{formatCount(debugInfo.divisions)}</span>
                      </p>
                      <p>
                        Mesure en ticks: <span className="font-mono">{formatCount(debugInfo.measureTicks)}</span>
                      </p>
                      <p>
                        Déflexion octave:{" "}
                        <span className="font-mono">{formatSigned(debugInfo.scoreWrittenOctaveShift)}</span>
                      </p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-[10px] uppercase text-zinc-500 dark:text-zinc-400">Comptes</p>
                      <p>
                        Events musicaux: <span className="font-mono">{formatCount(debugInfo.noteEventsCount)}</span>
                      </p>
                      <p>
                        Notes score.json:{" "}
                        <span className="font-mono">{formatCount(debugInfo.scoreJsonNotesCount)}</span>
                      </p>
                      <p>
                        Notes tab.json: <span className="font-mono">{formatCount(debugInfo.tabJsonNotesCount)}</span>
                      </p>
                      <p>
                        Notes score MusicXML:{" "}
                        <span className="font-mono">{formatCount(debugInfo.scoreMusicXmlNotesCount)}</span>
                      </p>
                      <p>
                        Notes tab MusicXML:{" "}
                        <span className="font-mono">{formatCount(debugInfo.tabMusicXmlNotesCount)}</span>
                      </p>
                    </div>
                  </div>
                  <div className="space-y-1 rounded-xl border border-zinc-200 bg-white/60 p-3 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950/60 dark:text-zinc-200">
                    <p className="text-[10px] uppercase text-zinc-500 dark:text-zinc-400">Jouabilité</p>
                    <p>
                      Score: <span className="font-mono">{formatDecimal(debugInfo.playabilityScore)}</span>
                    </p>
                    <p>
                      Coût: <span className="font-mono">{formatDecimal(debugInfo.playabilityCost)}</span>
                    </p>
                    <p>
                      Span: <span className="font-mono">{debugInfo.handSpan ?? "—"}</span>
                    </p>
                    <p>
                      Bas du manche:&nbsp;
                      <span className="font-mono">{debugInfo.preferLowFrets ? "Oui" : "Non"}</span>
                    </p>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div>
                      <p className="text-[10px] uppercase text-zinc-500 dark:text-zinc-400">Chemins réels</p>
                      <p className="font-mono break-all">PDF: {debugInfo.paths.pdf ?? "—"}</p>
                      <p className="font-mono break-all">MusicXML: {debugInfo.paths.musicxml ?? "—"}</p>
                      <p className="font-mono break-all">TAB: {debugInfo.paths.tabTxt ?? "—"}</p>
                      <p className="font-mono break-all">Score JSON: {debugInfo.paths.scoreJson ?? "—"}</p>
                      <p className="font-mono break-all">Score MusicXML: {debugInfo.paths.scoreMusicxml ?? "—"}</p>
                      <p className="font-mono break-all">Logs: {debugInfo.paths.logs ?? "—"}</p>
                    </div>
                    <div>
                      <p className="text-[10px] uppercase text-zinc-500 dark:text-zinc-400">Tailles</p>
                      <p>PDF: {formatBytes(debugInfo.sizes.pdf)}</p>
                      <p>MusicXML: {formatBytes(debugInfo.sizes.musicxml)}</p>
                      <p>TAB: {formatBytes(debugInfo.sizes.tabTxt)}</p>
                      <p>Score MusicXML: {formatBytes(debugInfo.sizes.scoreMusicxml)}</p>
                      <p>Score PDF: {formatBytes(debugInfo.sizes.scorePdf)}</p>
                    </div>
                  </div>
                  {debugInfo.lastMuseScore && (
                    <div>
                      <p className="text-[10px] uppercase text-zinc-500 dark:text-zinc-400">
                        Dernière commande MuseScore
                      </p>
                      <p className="font-mono text-[11px] break-all">{debugInfo.lastMuseScore.command}</p>
                      {debugInfo.lastMuseScore.stdout && (
                        <p className="text-[10px] text-zinc-500 dark:text-zinc-400">
                          stdout: {debugInfo.lastMuseScore.stdout}
                        </p>
                      )}
                      {debugInfo.lastMuseScore.stderr && (
                        <p className="text-[10px] text-zinc-500 dark:text-zinc-400">
                          stderr: {debugInfo.lastMuseScore.stderr}
                        </p>
                      )}
                    </div>
                  )}
                  {debugInfo.diffReport && debugInfo.diffReport.length > 0 && (
                    <div className="space-y-1 rounded-xl border border-red-200 bg-red-50 p-3 text-xs text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
                      <p className="text-[10px] uppercase text-red-500 dark:text-red-400">Diff MusicXML</p>
                      {debugInfo.diffReport.slice(0, 3).map((diff, idx) => (
                        <p key={`diff-${idx}`}>
                          {idx + 1}. Mesure {diff.measure ?? "?"} corde {diff.string} frette{" "}
                          {diff.fret}: tab.json {diff.countTabJson ?? 0} vs MusicXML{" "}
                          {diff.countMusicXML ?? 0}
                        </p>
                      ))}
                      {debugInfo.diffReport.length > 3 && (
                        <p>... et {debugInfo.diffReport.length - 3} autres différences</p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
