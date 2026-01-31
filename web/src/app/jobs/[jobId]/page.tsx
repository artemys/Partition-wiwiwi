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

export default function JobDetailsPage() {
  const params = useParams();
  const router = useRouter();
  const jobId = Array.isArray(params.jobId) ? params.jobId[0] : params.jobId;
  const resolvedJobId = typeof jobId === "string" ? jobId : "";
  const [tabPreview, setTabPreview] = useState<string | null>(null);
  const [pdfObjectUrl, setPdfObjectUrl] = useState<string | null>(null);
  const [pdfFetchInfo, setPdfFetchInfo] = useState<PdfFetchInfo | null>(null);
  const [pdfError, setPdfError] = useState<string | null>(null);

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
    if (preference === "pdf" && result.pdfUrl) return result.pdfUrl;
    if (preference === "musicxml" && result.musicXmlUrl) return result.musicXmlUrl;
    if (preference === "tab" && result.tabTxtUrl) return result.tabTxtUrl;
    if (preference === "midi" && result.midiUrl) return result.midiUrl;
    return result.pdfUrl || result.musicXmlUrl || result.tabTxtUrl || result.tabJsonUrl || result.midiUrl || null;
  }, [result]);

  const pdfUrl = result?.pdfUrl ?? null;

  const handleOpenPdf = () => {
    if (!pdfUrl) {
      return;
    }
    if (pdfFetchInfo?.status === 404) {
      toast.error("PDF introuvable (404).");
    } else if (pdfFetchInfo && (pdfFetchInfo.contentType ?? "").split(";")[0] !== "application/pdf") {
      toast.error("La ressource n'est pas un PDF.");
    }
    window.open(pdfUrl, "_blank", "noopener,noreferrer");
  };

  const formatBytes = (value?: number | null) =>
    value != null ? `${value.toLocaleString("fr-FR")} octets` : "—";

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

  useEffect(() => {
    if (!pdfUrl) {
      setPdfObjectUrl(null);
      setPdfFetchInfo(null);
      setPdfError(null);
      return;
    }
    let objectUrl: string | null = null;
    const controller = new AbortController();
    setPdfObjectUrl(null);
    setPdfFetchInfo(null);
    setPdfError(null);

    const loadPdf = async () => {
      try {
        const response = await fetch(pdfUrl, { signal: controller.signal });
        const contentType = response.headers.get("content-type");
        const contentLengthHeader = response.headers.get("content-length");
        const parsedLength = contentLengthHeader ? Number(contentLengthHeader) : null;
        const contentLength = parsedLength !== null && !Number.isNaN(parsedLength) ? parsedLength : null;
        const info: PdfFetchInfo = {
          status: response.status,
          contentType,
          contentLength,
        };
        if (controller.signal.aborted) {
          return;
        }
        setPdfFetchInfo(info);
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
        objectUrl = URL.createObjectURL(blob);
        setPdfObjectUrl(objectUrl);
        setPdfError(null);
      } catch (err) {
        if (controller.signal.aborted) {
          return;
        }
        const message = err instanceof Error && err.message ? err.message : "Impossible de charger le PDF.";
        setPdfError(message);
        setPdfObjectUrl(null);
      }
    };

    void loadPdf();

    return () => {
      controller.abort();
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [pdfUrl]);

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
            {result?.pdfUrl && (
              <Button variant="secondary" onClick={handleOpenPdf}>
                Ouvrir PDF
              </Button>
            )}
          </div>
        )}
      </Card>

      {status.status === "DONE" && (
        <>
          <div className="grid gap-4 lg:grid-cols-2">
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
            <Card className="space-y-3">
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Prévisualisation PDF</h2>
              {result?.pdfUrl ? (
                <div className="min-h-[520px] overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800">
                  {pdfObjectUrl ? (
                    <embed
                      src={pdfObjectUrl}
                      type="application/pdf"
                      className="h-[520px] w-full"
                      title="Prévisualisation PDF"
                    />
                  ) : pdfError ? (
                    <div className="flex flex-col items-center justify-center gap-3 p-6 text-sm text-zinc-600 dark:text-zinc-400">
                      <p>{pdfError}</p>
                      <Button variant="ghost" onClick={handleOpenPdf}>
                        Ouvrir dans un nouvel onglet
                      </Button>
                    </div>
                  ) : (
                    <p className="p-6 text-sm text-zinc-500 dark:text-zinc-400">Chargement du PDF…</p>
                  )}
                </div>
              ) : (
                <p className="text-sm text-zinc-500 dark:text-zinc-400">Aucun PDF disponible.</p>
              )}
            </Card>
          </div>
          <Card className="space-y-3">
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Info debug</h2>
            <div className="space-y-2 text-sm text-zinc-600 dark:text-zinc-300">
              <p className="text-xs uppercase text-zinc-500 dark:text-zinc-400">Ressource PDF</p>
              <div className="flex justify-between">
                <span>pdfUrl</span>
                <span className="font-mono text-xs text-zinc-700 dark:text-zinc-200 break-all">
                  {result?.pdfUrl ?? "—"}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Status HTTP</span>
                <span>{pdfFetchInfo?.status ?? "—"}</span>
              </div>
              <div className="flex justify-between">
                <span>Content-Type</span>
                <span>{pdfFetchInfo?.contentType ?? "—"}</span>
              </div>
              <div className="flex justify-between">
                <span>Content-Length</span>
                <span>{formatBytes(pdfFetchInfo?.contentLength)}</span>
              </div>
            </div>
            <div className="space-y-2 text-sm text-zinc-600 dark:text-zinc-300">
              {isDebugLoading && <p>Chargement des informations de debug…</p>}
              {isDebugError && <p>Impossible de charger les infos de debug.</p>}
              {debugInfo && (
                <div className="space-y-2 rounded-xl border border-zinc-200 p-3 text-xs text-zinc-700 dark:border-zinc-800 dark:text-zinc-200">
                  <div>
                    <p className="text-[10px] uppercase text-zinc-500 dark:text-zinc-400">Chemins réels</p>
                    <p className="font-mono break-all">PDF: {debugInfo.paths.pdf ?? "—"}</p>
                    <p className="font-mono break-all">MusicXML: {debugInfo.paths.musicxml ?? "—"}</p>
                    <p className="font-mono break-all">TAB: {debugInfo.paths.tabTxt ?? "—"}</p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase text-zinc-500 dark:text-zinc-400">Tailles</p>
                    <p>PDF: {formatBytes(debugInfo.sizes.pdf)}</p>
                    <p>MusicXML: {formatBytes(debugInfo.sizes.musicxml)}</p>
                    <p>TAB: {formatBytes(debugInfo.sizes.tabTxt)}</p>
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
                </div>
              )}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
